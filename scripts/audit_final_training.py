"""Read-only final-fit audit; no fitting, canonical shard reads, or val/test access."""
import argparse
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from src.final_study.config import load,verify_seal
from src.final_study.training import verify_run
from src.architecture_study.config import ROOT,grouped_folds,fingerprint
from src.architecture_study.data import TrainingStore,digest,aligned_values,write_json
from src.architecture_study.artifacts import load_checkpoint
from src.architecture_study.verify_head_convergence import objective_gradient
from src.architecture_study.projection import project
from src.architecture_study.synthetic import tensors


def audit(run):
    spec,c,task = load()
    torch.set_num_threads(spec['cpu_threads']); torch.use_deterministic_algorithms(True)
    marker = verify_seal(run)
    risk,scaler = verify_run(run,spec,c)
    meta = json.loads((run/'provenance.json').read_text())
    if json.loads((run/'spec.json').read_text()) != spec or json.loads((run/'effective_config.json').read_text()) != c:
        raise ValueError('Saved configuration mismatch')
    if subprocess.check_output(['git','rev-parse',meta['git_commit']],cwd=ROOT,text=True).strip() != meta['git_commit']:
        raise ValueError('Unknown training Git commit')
    store = TrainingStore(ROOT/c['canonical_run'],task)  # TRAIN export/metadata only.
    if store.run_hash != spec['run_metadata_sha256'] or store.export_hash != spec['train_export_sha256']:
        raise ValueError('Dataset identity mismatch')
    scope = grouped_folds(store.shards,spec['seed'],spec['checkpoint_partition_count'])[spec['checkpoint_partition_index']]
    expected = {'fit':sorted(scope.fit),'held':sorted(scope.held),'scope':scope.identifier}
    if json.loads((run/'patients.json').read_text()) != expected or scaler['scope'] != scope.identifier:
        raise ValueError('Patient/scaler scope mismatch')
    ssl = load_checkpoint(run/'R0-ssl.pt','R0',c,scaler,'ssl')
    if any(not torch.equal(v,risk.encoder.state_dict()[k]) for k,v in ssl.encoder.state_dict().items()):
        raise ValueError('SSL/risk encoder mismatch')
    seen = {}; reverse = {}; counts = {}
    head_refs = None
    for filename,operation,nested in [('ssl_sampling.json','ssl',True),('checkpoint_samples.json','checkpoint',True),
                                       ('scaler_samples.json','scaler',False),('head_samples.json','head',False)]:
        saved = json.loads((run/filename).read_text())
        batches = saved if nested else [saved]
        if operation == 'ssl' and len(batches) != spec['ssl_draws']:
            raise ValueError('SSL budget mismatch')
        if operation == 'checkpoint' and len(batches) != spec['checkpoint_batches']:
            raise ValueError('Checkpoint budget mismatch')
        population = set(); unique = set(); exposures = 0
        for batch in batches:
            scope.authorize([r['patient'] for r in batch],operation)
            if nested and len(batch) > spec['batch_size']:
                raise ValueError('Batch exceeds frozen cap')
            if len({r['sample_id'] for r in batch}) != len(batch) and nested:
                raise ValueError('Duplicate within sampled batch')
            for r in batch:
                p,i,sid = r['patient'],int(r['index']),r['sample_id']
                if not 0 <= i < store.by_patient[p]['samples'] or not sid.startswith('sample_') or len(bytes.fromhex(sid[7:])) != 32:
                    raise ValueError('Malformed/out-of-bounds sample identity')
                key = (p,i)
                if seen.get(key,sid) != sid or reverse.get(sid,key) != key:
                    raise ValueError('Inconsistent sample identity')
                seen[key] = sid; reverse[sid] = key
                population.add(p); unique.add(sid); exposures += 1
                if operation == 'head':
                    if not 0 < r['inclusion_probability'] <= 1:
                        raise ValueError('Invalid head weights')
                    allowed = {e[1] for e in store.by_patient[p]['events']}
                    if r['episode_group'] != ['negative'] and not set(r['episode_group']) <= allowed:
                        raise ValueError('Episode belongs to another patient/source')
        counts[operation] = {'patients':len(population),'windows_unique':len(unique),'exposures':exposures}
        if operation == 'head': head_refs = saved
        if operation == 'scaler' and exposures != c['resources']['scaler_windows']:
            raise ValueError('Scaler budget mismatch')
    selection = json.loads((run/'selection.json').read_text())
    history = [json.loads(line) for line in (run/'training_history.jsonl').read_text().splitlines()]
    if [r['draw'] for r in history] != list(range(1,spec['ssl_draws']+1)):
        raise ValueError('Missing/reordered training history')
    checks = [{'draw':r['draw'],'loss':r['checkpoint_loss']} for r in history if 'checkpoint_loss' in r]
    if checks != selection['history'] or [r['draw'] for r in checks] != list(range(spec['checkpoint_every'],spec['ssl_draws']+1,spec['checkpoint_every'])):
        raise ValueError('Checkpoint history/schedule mismatch')
    best = min(checks,key=lambda x:x['loss'])
    if best['draw'] != selection['selected_draw'] or selection['criterion'] != spec['selection']:
        raise ValueError('Incorrect checkpoint selection')
    if sum(r['targets']>0 for r in history) != selection['ssl_optimizer_updates'] or sum(r['windows'] for r in history) != counts['ssl']['exposures']:
        raise ValueError('Exposure/update accounting mismatch')
    if selection['scaler_sha256'] != digest(run/'scaler.json') or selection['checkpoint_samples_sha256'] != digest(run/'checkpoint_samples.json'):
        raise ValueError('Selection artifact linkage mismatch')
    fit = json.loads((run/'head_fit.json').read_text())
    with np.load(run/'head_embeddings.npz',allow_pickle=False) as z:
        x,y,w,scores = z['embeddings'],z['labels'],z['weights'],z['scores']
        aligned_values([r['sample_id'] for r in head_refs],z['sample_ids'].tolist(),x)
        np.testing.assert_array_equal(y,[int(r['episode_group']!=['negative']) for r in head_refs])
        np.testing.assert_array_equal(w,[1/r['inclusion_probability'] for r in head_refs])
        objective,gradient = objective_gradient(x,y,w,fit,c['optimization']['head_l2'])
        np.testing.assert_allclose(objective,fit['final_objective'],rtol=1e-10,atol=1e-14)
        np.testing.assert_allclose(gradient,fit['final_gradient_max'],rtol=1e-5,atol=1e-13)
        if gradient > spec['head_stationarity_gate'] or not fit['stationarity_pass'] or not 0 < fit['iterations'] <= spec['head_max_iter']:
            raise ValueError('Head convergence failed')
        np.testing.assert_array_equal(risk.head.weight.detach().numpy(),np.array(fit['weight'],dtype=np.float32))
        np.testing.assert_array_equal(risk.head.bias.detach().numpy(),np.array(fit['bias'],dtype=np.float32))
        with torch.no_grad():
            replay = risk.head(torch.tensor(x)).sigmoid().flatten().numpy()
            np.testing.assert_array_equal(replay,scores)
            np.testing.assert_array_equal(risk.head(torch.tensor(x)).sigmoid().flatten().numpy(),scores)
            view = project(tensors(3),scaler['mean'],scaler['scale'],task)
            reloaded,_ = verify_run(run,spec,c)
            if not torch.equal(risk(view),reloaded(view)) or not torch.equal(risk(view),risk(view)):
                raise ValueError('Full synthetic inference/reload differs')
    counts['head'].update(positive_windows=int(y.sum()),linked_episodes=len({e for r in head_refs for e in r['episode_group'] if e!='negative'}))
    return {'status':'passed','scope':'saved artifacts, TRAIN export only, head numerical replay and synthetic full inference; no canonical patient shards read',
        'git_commit':meta['git_commit'],'artifact_hash_count':len(marker['artifacts_sha256']),
        'complete_sha256':digest(run/'complete.json'),'spec_sha256':fingerprint(spec),
        'fit_patients':len(scope.fit),'checkpoint_holdout_patients':len(scope.held),'counts':counts,
        'initial_held_loss':selection['initial_checkpoint_loss'],'selected':best,'final_held_loss':checks[-1]['loss'],
        'final_train_batch_loss':history[-1]['loss'],'masked_targets':sum(r['targets'] for r in history),
        'zero_target_windows':sum(r['zero_target_windows'] for r in history),'optimizer_updates':selection['ssl_optimizer_updates'],
        'head_gradient':gradient,'head_objective':objective,'head_iterations':fit['iterations'],
        'scaler_unique_rows':scaler['unique_rows'],'scaler_observations':scaler['observations'],
        'wall_seconds_after_preparation':selection['wall_seconds_after_preparation'],
        'whole_stage_seconds_timestamp_estimate':(run/'complete.json').stat().st_mtime-(run/'provenance.json').stat().st_mtime,
        'peak_ram':'unavailable: not recorded','artifact_bytes':sum(p.stat().st_size for p in run.iterdir() if p.is_file()),
        'limitations':'Raw scaler statistics, held-loss inference, masking stream and physiological embedding extraction were not recomputed from private shards; validated through frozen source, manifests and hashes.'}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or not args.output.resolve().is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Use a new ignored private audit output')
    result=audit(args.run); write_json(args.output,result); print(json.dumps(result,indent=2))
