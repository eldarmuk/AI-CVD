"""Bounded fitting helpers. All fitted populations carry a training-fold scope."""
import copy
import numpy as np
import torch
from torch.nn import functional as F
from .models import FrozenRisk, MaskedModel, masked_huber
from .projection import project, mask_observed
from .data import aligned_values


def make_model(recipe, config, device):
    m = config['model']
    return MaskedModel(recipe, m['value_width'], m['process_width'], m['decoder_width']).to(device)


def view_for(batch, scaler, task, device, generator=None):
    if batch['scope'] != scaler['scope']:
        raise ValueError('Scaler/fold mismatch')
    if any(batch[key] != scaler[key] for key in ('task_identifier', 'run_sha256', 'export_sha256')) or scaler['task_identifier'] != task.identifier:
        raise ValueError('Scaler/canonical source mismatch')
    x = torch.as_tensor(batch['X'], dtype=torch.float32, device=device)
    view = project(x, scaler['mean'], scaler['scale'], task)
    if generator is not None:
        A = mask_observed(view.M, generator)
        view = project(x, scaler['mean'], scaler['scale'], task, A)
    return view


def ssl_step(model, optimizer, view, gradient_clip=1.0, delta=1.0):
    optimizer.zero_grad(set_to_none=True)
    loss = masked_huber(model(view), view, delta)
    if view.A.any():
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip, error_if_nonfinite=True)
        optimizer.step()
    return float(loss.detach()), int(view.A.sum())


def embed_refs(model, store, refs, scope, operation, scaler, device, batch_size):
    scope.authorize([r['patient'] for r in refs], operation)
    risk = FrozenRisk(model).to(device).eval()
    embeddings, labels, ids, episodes = [], [], [], []
    for start in range(0, len(refs), batch_size):
        selected = refs[start:start + batch_size]
        batch = store.batch(selected, scope, operation, labels=True)
        view = view_for(batch, scaler, store.task, device)
        embeddings.append(risk.embeddings(view).cpu())
        labels.extend(batch['y'].tolist()); ids.extend(batch['sample_ids']); episodes.extend(batch['episodes'])
    aligned_values([r['sample_id'] for r in refs], ids, labels)
    return risk, torch.cat(embeddings), torch.tensor(labels), ids, episodes


def fit_head(risk, embeddings, labels, weights, patients, scope, l2=1e-6, iterations=50):
    scope.authorize(patients, 'head')
    if len(embeddings) != len(labels) or len(weights) != len(labels) or len(patients) != len(labels):
        raise ValueError('Head alignment mismatch')
    if set(labels.tolist()) != {0, 1}:
        raise ValueError('Head requires both classes in fit patients')
    # Full-batch convex head optimization only. Encoder stays frozen and is copied back unchanged.
    before = {k: v.detach().clone() for k, v in risk.encoder.state_dict().items()}
    head = copy.deepcopy(risk.head).cpu().double()
    x, y, w = embeddings.detach().cpu().double(), labels.cpu().double(), torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(x).all() or not torch.isfinite(w).all() or (w <= 0).any():
        raise ValueError('Invalid head data/weights')
    # Scaling is fitted only on head-training embeddings, then algebraically absorbed.
    wn = w / w.sum()
    mean = (x * wn[:, None]).sum(0)
    scale = ((x - mean).square() * wn[:, None]).sum(0).sqrt().clamp_min(1e-6)
    z = (x - mean) / scale
    with torch.no_grad():
        head.weight.zero_(); head.bias.fill_(torch.logit((y * wn).sum().clamp(1e-9, 1 - 1e-9)))
    optimizer = torch.optim.LBFGS(head.parameters(), max_iter=iterations, line_search_fn='strong_wolfe')
    def closure():
        optimizer.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(head(z).squeeze(-1), y, reduction='none') * wn).sum() + l2 * head.weight.square().sum()
        loss.backward()
        return loss
    initial = float(closure().detach())
    optimizer.step(closure)
    final = float(closure().detach())
    with torch.no_grad():
        weight = head.weight / scale
        bias = head.bias - (weight * mean).sum()
        risk.head.weight.copy_(weight.to(risk.head.weight)); risk.head.bias.copy_(bias.to(risk.head.bias))
    if any(not torch.equal(before[k], v) for k, v in risk.encoder.state_dict().items()):
        raise AssertionError('Frozen encoder changed during head fitting')
    return {'initial_objective': initial, 'final_objective': final, 'l2': l2,
            'embedding_mean': mean.tolist(), 'embedding_scale': scale.tolist(),
            'scope': scope.identifier, 'encoder_unchanged': True,
            'weighting': 'inverse_window_inclusion_probability', 'iterations_cap': iterations}


def development_metrics(labels, scores, weights):
    """IP-weighted development estimates, never labeled natural-stream results."""
    y, p, w = map(lambda a: np.asarray(a, dtype=float), (labels, scores, weights))
    if not (len(y) == len(p) == len(w)) or not np.isfinite(p).all() or (w <= 0).any():
        raise ValueError('Invalid predictions')
    order = np.argsort(-p, kind='stable'); y, p, w = y[order], p[order], w[order]
    positive = np.cumsum(w*y); total = np.cumsum(w)
    ends = np.r_[np.flatnonzero(np.diff(p)), len(p)-1]
    if positive[-1] == 0:
        ap = None
    else:
        recall = positive[ends] / positive[-1]
        ap = float(np.sum(np.diff(np.r_[0, recall]) * positive[ends] / total[ends]))
    bounded = np.clip(p, 1e-12, 1-1e-12)
    return {'ip_weighted_average_precision': ap,
            'ip_weighted_log_loss': float(np.sum(-w*(y*np.log(bounded)+(1-y)*np.log1p(-bounded)))/w.sum()),
            'ip_weighted_prevalence': float(np.sum(w*y)/w.sum()),
            'samples': len(y), 'positive_windows': int(y.sum()),
            'interpretation': 'sampled inner-training-fold development estimate; not natural-stream clinical utility'}
