"""Synthetic checks, TRAIN-only loader smoke, and explicitly requested bounded runs."""
import argparse
import hashlib
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import shutil
import time
import numpy as np
import torch
from .config import ROOT, DEFAULT_CONFIG, load_config, grouped_folds, private_output, fingerprint
from .data import TrainingStore, BlockSampler, fit_scaler, supervised_refs, write_json, digest
from .models import parameter_counts, masked_huber
from .projection import project, mask_observed
from .synthetic import tensors
from .training import make_model, view_for, ssl_step, embed_refs, fit_head, development_metrics
from .artifacts import save_checkpoint
from .telemetry import peak_rss


def environment(device):
    return {'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
            'device': str(device), 'hardware': torch.cuda.get_device_name() if device.type == 'cuda' else platform.processor(),
            'threads': torch.get_num_threads()}


def benchmark(c, task, output, device):
    result = {'environment': environment(device), 'scope': 'synthetic only', 'measurements': []}
    for recipe in c['recipes']:
        for batch_size in (16, 32, 64):
            torch.manual_seed(c['seed'])
            model = make_model(recipe, c, device)
            x = tensors(batch_size).to(device)
            v = project(x, np.zeros(6), np.ones(6), task)
            gen = torch.Generator(device=device).manual_seed(c['seed'])
            v = project(x, np.zeros(6), np.ones(6), task, mask_observed(v.M, gen))
            optimizer = torch.optim.Adam(model.parameters(), lr=c['optimization']['ssl_lr'])
            for _ in range(2):
                ssl_step(model, optimizer, v)
            if device.type == 'cuda':
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            losses = [ssl_step(model, optimizer, v)[0] for _ in range(5)]
            if device.type == 'cuda':
                torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            counts = parameter_counts(model)
            import ctypes
            from ctypes import wintypes
            class Memory(ctypes.Structure):
                _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in ('PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage', 'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
            rss = None
            if os.name == 'nt':
                memory = Memory(); memory.cb = ctypes.sizeof(memory)
                kernel = ctypes.WinDLL('kernel32'); kernel.GetCurrentProcess.restype = wintypes.HANDLE
                psapi = ctypes.WinDLL('psapi')
                psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Memory), wintypes.DWORD]
                if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(memory), memory.cb):
                    rss = memory.PeakWorkingSetSize
            result['measurements'].append({'recipe': recipe, 'batch_size': batch_size,
                'seconds_per_update': seconds/5, 'windows_per_second': batch_size*5/seconds,
                'losses': losses, 'parameters': counts,
                'parameter_gradient_adam_bytes': counts['ssl_total']*16,
                'input_bytes': x.numel()*x.element_size(),
                'process_peak_rss_bytes': rss,
                'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None})
            del model, optimizer, v, x
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    write_json(output/'benchmark.json', result)
    print(json.dumps(result, indent=2))


def loader_smoke(c, task, store, scopes, output, device):
    scope = scopes[0]
    refs = []
    load_seconds = []
    start = time.perf_counter()
    for p in sorted(scope.fit)[:2]:
        before_load = time.perf_counter()
        a = store.arrays(p)
        load_seconds.append(time.perf_counter()-before_load)
        for i in np.linspace(0, len(a['end_rows'])-1, 2, dtype=int):
            sid, _ = store.identity(p, int(i), a)
            refs.append({'patient': p, 'index': int(i), 'sample_id': sid})
    scaler = fit_scaler(store, refs, scope)
    batch = store.batch(refs, scope, 'ssl')
    result = {'purpose': 'TRAIN-only loader/forward smoke; no optimizer or labels', 'samples': len(refs),
              'scaler': scaler, 'sample_ids': batch['sample_ids'], 'recipes': {},
              'cold_shard_hash_decode_seconds': load_seconds}
    for recipe in c['recipes']:
        model = make_model(recipe, c, device).eval()
        gen = torch.Generator(device=device).manual_seed(c['seed'])
        view = view_for(batch, scaler, task, device, gen)
        if device.type == 'cuda':
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        forward_start = time.perf_counter()
        with torch.no_grad():
            loss = masked_huber(model(view), view)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        result['recipes'][recipe] = {'finite': bool(torch.isfinite(loss)), 'targets': int(view.A.sum()), 'parameters': parameter_counts(model),
            'forward_seconds': time.perf_counter()-forward_start,
            'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
            'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if device.type == 'cuda' else None}
    result['seconds'] = time.perf_counter()-start
    write_json(output/'loader_smoke.json', result)
    print('TRAIN-only loader smoke complete; private identities/provenance saved in output directory.')


def run(c, task, store, scopes, output, device):
    """Six bounded fits. This entry point is never called by smoke/benchmark commands."""
    r, opt = c['resources'], c['optimization']
    for scope in scopes:
        print(f'Preparing fold {scope.fold}', flush=True)
        folder = output / f'fold-{scope.fold}'; folder.mkdir()
        write_json(folder/'patients.json', {'fit': sorted(scope.fit), 'held': sorted(scope.held), 'scope': scope.identifier})
        sampler = BlockSampler(store, scope, 'scaler', c['seed'] + scope.fold, c['sampling'])
        scaler_refs = []
        while len(scaler_refs) < r['scaler_windows']:
            scaler_refs.extend(sampler.draw(r['batch_size']))
        scaler_refs = scaler_refs[:r['scaler_windows']]
        scaler = fit_scaler(store, scaler_refs, scope)
        write_json(folder/'scaler.json', scaler); write_json(folder/'scaler_samples.json', scaler_refs)
        # One paired plan and corruption seed per fold, shared by R0 and R1.
        sampler = BlockSampler(store, scope, 'ssl', c['seed'] + scope.fold + 100, c['sampling'])
        plan = [sampler.draw(r['batch_size']) for _ in range(r['ssl_updates'])]
        write_json(folder/'ssl_sampling.json', plan)
        checkpoint_sampler = BlockSampler(store, scope, 'checkpoint', c['seed'] + scope.fold + 200, c['sampling'])
        checkpoint_refs = [checkpoint_sampler.draw(r['batch_size']) for _ in range(r['checkpoint_batches'])]
        write_json(folder/'checkpoint_samples.json', checkpoint_refs)
        head_refs = supervised_refs(store, scope, 'head', c['seed'], r)
        assessment_refs = supervised_refs(store, scope, 'assessment', c['seed'], r)
        write_json(folder/'head_samples.json', head_refs); write_json(folder/'assessment_samples.json', assessment_refs)
        write_json(folder/'plan_hashes.json', {name: digest(folder/name) for name in ('patients.json', 'scaler_samples.json', 'ssl_sampling.json', 'checkpoint_samples.json', 'head_samples.json', 'assessment_samples.json')})
        paired_digest = None
        for recipe in c['recipes']:
            torch.manual_seed(c['seed'] + scope.fold)
            model = make_model(recipe, c, device)
            if recipe == 'R0':
                save_checkpoint(folder/'R0-random-ssl.pt', model, recipe, c, scaler, 'ssl')
            optimizer = torch.optim.Adam(model.parameters(), lr=opt['ssl_lr'])
            generator = torch.Generator(device=device).manual_seed(c['seed'] + scope.fold + 300)
            best, best_state, history = float('inf'), None, []
            updates = 0
            started = time.perf_counter()
            draws = []
            def checkpoint_loss():
                check = []
                model.eval()
                fixed = torch.Generator(device=device).manual_seed(c['seed'] + scope.fold + 400)
                with torch.no_grad():
                    for refs_check in checkpoint_refs:
                        v = view_for(store.batch(refs_check, scope, 'checkpoint'), scaler, task, device, fixed)
                        if v.A.any():
                            check.append(float(masked_huber(model(v), v)))
                if not check:
                    raise ValueError('No valid held-training-fold SSL targets; selection infeasible')
                return float(np.mean(check))
            initial_held_loss = checkpoint_loss()
            corruptions = hashlib.sha256()
            for step, refs in enumerate(plan, 1):
                batch = store.batch(refs, scope, 'ssl')
                view = view_for(batch, scaler, task, device, generator)
                corruptions.update(json.dumps(batch['sample_ids']).encode())
                corruptions.update(view.A.detach().cpu().numpy().tobytes())
                model.train()
                loss, targets = ssl_step(model, optimizer, view, opt['gradient_clip'], opt['huber_delta'])
                updates += int(targets > 0)
                draws.append({'draw': step, 'loss': loss, 'targets': targets, 'windows': len(refs),
                              'zero_target_windows': int((~view.A.flatten(1).any(1)).sum()),
                              'no_primitive_windows': int((~view.M.flatten(1).any(1)).sum())})
                if step % r['checkpoint_every'] == 0 or step == len(plan):
                    score = checkpoint_loss()
                    history.append({'draw': step, 'optimizer_updates': updates, 'train_loss': loss, 'inner_ssl_loss': score})
                    print(json.dumps({'fold': scope.fold, 'recipe': recipe, 'draw': step, 'seconds': round(time.perf_counter()-started)}), flush=True)
                    if score < best:
                        best = score; best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if updates == 0 or best_state is None:
                raise ValueError('No SSL updates/checkpoint')
            corruption_hash = corruptions.hexdigest()
            if paired_digest is not None and paired_digest != corruption_hash:
                raise AssertionError('R0/R1 sample/corruption pairing failed')
            paired_digest = corruption_hash
            seconds = time.perf_counter()-started
            write_json(folder/f'{recipe}-telemetry.json', {'draws': draws, 'initial_held_loss': initial_held_loss,
                'ssl_wall_seconds': seconds, 'windows_per_second': sum(d['windows'] for d in draws)/seconds,
                'process_cumulative_peak_rss_bytes': peak_rss(), 'optimizer_updates': updates,
                'unique_samples': len({r['sample_id'] for refs in plan for r in refs}),
                'exposed_patients': len({r['patient'] for refs in plan for r in refs}),
                'checkpoint_criterion': 'minimum mean fixed inner-held SSL batch loss at scheduled checkpoints; earliest tie',
                'selected_draw': min(history, key=lambda h:h['inner_ssl_loss'])['draw']})
            model.load_state_dict(best_state)
            save_checkpoint(folder/f'{recipe}-ssl.pt', model, recipe, c, scaler, 'ssl')
            risk, z, y, ids, episodes = embed_refs(model, store, head_refs, scope, 'head', scaler, device, r['batch_size'])
            fitted = fit_head(risk, z, y, [1/r['inclusion_probability'] for r in head_refs],
                              [r['patient'] for r in head_refs], scope, opt['head_l2'], r['head_iterations'])
            _, held_z, held_y, held_ids, held_episodes = embed_refs(model, store, assessment_refs, scope, 'assessment', scaler, device, r['batch_size'])
            with torch.no_grad():
                scores = risk.head(held_z.to(device)).squeeze(-1).sigmoid().cpu().tolist()
            weights = [1/r['inclusion_probability'] for r in assessment_refs]
            np.savez_compressed(folder/f'{recipe}-embeddings.npz', fit=z.numpy(), held=held_z.numpy(),
                                fit_ids=np.asarray(ids), held_ids=np.asarray(held_ids))
            metrics = development_metrics(held_y.tolist(), scores, weights)
            metrics['unique_linked_episodes'] = len({e for es in held_episodes for e in es})
            write_json(folder/f'{recipe}-development.json', {'history': history, 'head': fitted, 'metrics': metrics,
                       'sample_corruption_sha256': corruption_hash,
                       'sample_ids': held_ids, 'scores': scores, 'labels': held_y.tolist(), 'episodes': held_episodes})
            save_checkpoint(folder/f'{recipe}-risk.pt', risk.cpu(), recipe, c, scaler, 'risk')
    write_json(output/'complete.json', {'status': 'complete', 'scope': 'grouped canonical TRAIN only',
               'caution': 'Development screen only; no natural-stream operational metrics or final architecture decision'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['benchmark', 'loader-smoke', 'run'])
    parser.add_argument('--config', default=str(DEFAULT_CONFIG))
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    c, task = load_config(args.config)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(c['resources']['cpu_threads'])
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA unavailable')
    torch.use_deterministic_algorithms(True)
    output = private_output(args.output)
    write_json(output/'study.json', {'config': c, 'config_sha256': fingerprint(c), 'environment': environment(device),
                                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                                  'implementation_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/'src/architecture_study').glob('*.py'))},
                                  'created_utc': datetime.now(timezone.utc).isoformat(), 'command': args.command})
    shutil.copyfile(args.config, output/'config.toml')
    if args.command == 'benchmark':
        benchmark(c, task, output, device)
    else:
        store = TrainingStore(ROOT/c['canonical_run'], task, c['sampling']['cache_patients'], c['sampling']['cache_megabytes'])
        scopes = grouped_folds(store.shards, c['seed'], c['folds'])
        write_json(output/'canonical_provenance.json', {'run_metadata_sha256': store.run_hash, 'train_export_sha256': store.export_hash})
        (loader_smoke if args.command == 'loader-smoke' else run)(c, task, store, scopes, output, device)


if __name__ == '__main__':
    main()
