"""One bounded R0 fit; canonical TRAIN only. Never invoked by verification."""
import hashlib
import json
import platform
import subprocess
import time
from importlib.metadata import version
import numpy as np
import torch
from src.architecture_study.config import ROOT, grouped_folds, fingerprint
from src.architecture_study.data import TrainingStore, BlockSampler, fit_scaler, supervised_refs, write_json, digest
from src.architecture_study.training import make_model, view_for, ssl_step, embed_refs
from src.architecture_study.models import masked_huber
from src.architecture_study.artifacts import save_checkpoint, load_checkpoint
from src.architecture_study.head_convergence import solve
from src.ai_cvd.features import FEATURE_NAMES
from src.architecture_study.projection import PRIMITIVES, PROCESS_NAMES
from .config import source_hashes, seal, verify_seal


def verify_run(output, spec, c):
    verify_seal(output)
    meta = json.loads((output/'provenance.json').read_text())
    if meta['spec_sha256'] != fingerprint(spec) or meta['config_sha256'] != fingerprint(c) or meta['source_sha256'] != source_hashes():
        raise ValueError('Final config/implementation changed')
    if tuple(meta['feature_order']) != FEATURE_NAMES or tuple(meta['primitives']) != PRIMITIVES or tuple(meta['process_channels']) != PROCESS_NAMES:
        raise ValueError('Projection changed')
    scaler = json.loads((output/'scaler.json').read_text())
    return load_checkpoint(output/'R0-risk.pt','R0',c,scaler,'risk'), scaler


def train(spec,c,task,output,resume=False):
    torch.set_num_threads(spec['cpu_threads']); torch.use_deterministic_algorithms(True)
    if output.exists():
        if not resume:
            raise FileExistsError('Immutable output exists; use --resume to verify completed work')
        if not (output/'complete.json').exists():
            raise ValueError('Partial final fit: preserve directory; automatic restart/resume is not supported')
        verify_run(output,spec,c)
        store = TrainingStore(ROOT/c['canonical_run'],task)
        if store.run_hash != spec['run_metadata_sha256'] or store.export_hash != spec['train_export_sha256']:
            raise ValueError('Canonical provenance changed')
        print('Verified complete final fit; skipped without changes.',flush=True)
        return
    store = TrainingStore(ROOT/c['canonical_run'],task,c['sampling']['cache_patients'],c['sampling']['cache_megabytes'])
    if store.run_hash != spec['run_metadata_sha256'] or store.export_hash != spec['train_export_sha256']:
        raise ValueError('Canonical provenance mismatch')
    scope = grouped_folds(store.shards,spec['seed'],spec['checkpoint_partition_count'])[spec['checkpoint_partition_index']]
    output.mkdir(parents=True)
    write_json(output/'spec.json',spec); write_json(output/'effective_config.json',c)
    write_json(output/'provenance.json', {'spec_sha256':fingerprint(spec),'config_sha256':fingerprint(c),
        'source_sha256':source_hashes(),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'python':platform.python_version(),'packages':{p:version(p) for p in ('torch','numpy','pandas')},
        'device':'cpu','threads':torch.get_num_threads(),'task_identifier':task.identifier,
        'run_metadata_sha256':store.run_hash,'train_export_sha256':store.export_hash,
        'feature_order':FEATURE_NAMES,'primitives':PRIMITIVES,'process_channels':PROCESS_NAMES,
        'seeds':{'model':spec['seed'],'scaler':spec['seed'],'ssl_sampling':spec['seed']+100,
                 'checkpoint_sampling':spec['seed']+200,'ssl_corruption':spec['seed']+300,'checkpoint_corruption':spec['seed']+400}})
    write_json(output/'patients.json',{'fit':sorted(scope.fit),'held':sorted(scope.held),'scope':scope.identifier})
    r = c['resources']; opt = c['optimization']
    def plan(operation,offset,count):
        sampler = BlockSampler(store,scope,operation,spec['seed']+offset,c['sampling'])
        return [sampler.draw(r['batch_size']) for _ in range(count)]
    scaler_sampler = BlockSampler(store,scope,'scaler',spec['seed'],c['sampling'])
    refs = []
    while len(refs) < r['scaler_windows']:
        refs.extend(scaler_sampler.draw(r['batch_size']))
    refs = refs[:r['scaler_windows']]
    write_json(output/'scaler_samples.json',refs)
    scaler = fit_scaler(store,refs,scope); write_json(output/'scaler.json',scaler)
    sampling = plan('ssl',100,r['ssl_updates']); check = plan('checkpoint',200,r['checkpoint_batches'])
    write_json(output/'ssl_sampling.json',sampling); write_json(output/'checkpoint_samples.json',check)
    head_refs = supervised_refs(store,scope,'head',spec['seed'],r)
    write_json(output/'head_samples.json',head_refs)
    torch.manual_seed(spec['seed']); device = torch.device('cpu')
    model = make_model('R0',c,device)
    optimizer = torch.optim.Adam(model.parameters(),lr=opt['ssl_lr'])
    generator = torch.Generator().manual_seed(spec['seed']+300)
    def checkpoint():
        fixed = torch.Generator().manual_seed(spec['seed']+400)
        losses = []; model.eval()
        with torch.no_grad():
            for refs in check:
                v = view_for(store.batch(refs,scope,'checkpoint'),scaler,task,device,fixed)
                if v.A.any():
                    losses.append(float(masked_huber(model(v),v)))
        if not losses:
            raise ValueError('No checkpoint targets')
        return float(np.mean(losses))
    initial = checkpoint(); best = float('inf'); selected = None; history = []; updates = 0
    exposure_hash = hashlib.sha256(); started = time.perf_counter()
    with open(output/'training_history.jsonl','x') as log:
        for draw,refs in enumerate(sampling,1):
            batch = store.batch(refs,scope,'ssl')
            v = view_for(batch,scaler,task,device,generator)
            exposure_hash.update(json.dumps(batch['sample_ids']).encode()); exposure_hash.update(v.A.numpy().tobytes())
            model.train(); loss,targets = ssl_step(model,optimizer,v,opt['gradient_clip'],opt['huber_delta'])
            updates += int(targets>0)
            row = {'draw':draw,'loss':loss,'targets':targets,'windows':len(refs),
                   'zero_target_windows':int((~v.A.flatten(1).any(1)).sum()),'optimizer_updates':updates}
            if draw % r['checkpoint_every'] == 0 or draw == len(sampling):
                score = checkpoint(); row['checkpoint_loss'] = score
                history.append({'draw':draw,'loss':score})
                if score < best:
                    best = score; selected = draw
                    state = {k:v.detach().clone() for k,v in model.state_dict().items()}
                print(f'R0 draw {draw}/{len(sampling)}; checkpoint_loss={score:.6f}',flush=True)
            log.write(json.dumps(row)+'\n'); log.flush()
    if selected is None or updates == 0:
        raise ValueError('No trained checkpoint')
    model.load_state_dict(state)
    save_checkpoint(output/'R0-ssl.pt',model,'R0',c,scaler,'ssl')
    risk,z,y,ids,episodes = embed_refs(model,store,head_refs,scope,'head',scaler,device,r['batch_size'])
    scope.authorize([r['patient'] for r in head_refs],'head')
    before = {k:v.clone() for k,v in risk.encoder.state_dict().items()}
    weights = [1/r['inclusion_probability'] for r in head_refs]
    fit = solve(z.numpy(),y.numpy(),weights,opt['head_l2'])
    if not fit['stationarity_pass']:
        raise ValueError('Head did not converge; no completed final model')
    with torch.no_grad():
        risk.head.weight.copy_(torch.tensor(fit['weight'])); risk.head.bias.copy_(torch.tensor(fit['bias']))
    if any(not torch.equal(v,risk.encoder.state_dict()[k]) for k,v in before.items()):
        raise ValueError('Encoder changed during head fitting')
    save_checkpoint(output/'R0-risk.pt',risk,'R0',c,scaler,'risk')
    reloaded = load_checkpoint(output/'R0-risk.pt','R0',c,scaler,'risk')
    with torch.no_grad():
        scores = risk.head(z).sigmoid().flatten()
        if not torch.equal(scores,reloaded.head(z).sigmoid().flatten()):
            raise ValueError('Head reload changed predictions')
    np.savez_compressed(output/'head_embeddings.npz',embeddings=z.numpy(),sample_ids=np.asarray(ids),
                        labels=y.numpy(),weights=np.asarray(weights),scores=scores.numpy())
    write_json(output/'head_fit.json',fit)
    write_json(output/'selection.json',{'initial_checkpoint_loss':initial,'history':history,'selected_draw':selected,
        'criterion':spec['selection'],'sample_corruption_sha256':exposure_hash.hexdigest(),
        'ssl_optimizer_updates':updates,'wall_seconds_after_preparation':time.perf_counter()-started,
        'scaler_sha256':digest(output/'scaler.json'),'checkpoint_samples_sha256':digest(output/'checkpoint_samples.json')})
    seal(output,purpose='final R0 TRAIN-only fitting; validation not yet performed')
    verify_run(output,spec,c)
    print('Final R0 training complete: complete.json verified.',flush=True)
