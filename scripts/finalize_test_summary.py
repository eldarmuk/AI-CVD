"""Seal an aggregate paper summary after verifying saved completed analyses only."""
import argparse
import subprocess
from pathlib import Path
import numpy as np
from scripts import test_evaluation as ev
from scripts.audit_test_completion import metadata, console_text
from scripts.test_uncertainty import interval, binomial_interval, REPLICATES, SEED
from src.final_study.config import verify_seal, seal


def finalize(output):
    if output.exists() or not output.resolve().is_relative_to(ev.BASE.resolve()) or output.resolve().is_relative_to(ev.OUTPUT.resolve()):
        raise ValueError('Use a new private summary directory outside the original test run')
    current,e,task,root,blind,patients,marker,log=metadata()
    audit=ev.BASE/'r0-final-v1-test-audit-v1'; uncertainty=ev.OUTPUT/'uncertainty-v1'
    for folder in (audit,uncertainty):
        m=verify_seal(folder)
        if {p.name for p in folder.iterdir() if p.is_file()}!=set(m['artifacts_sha256'])|{'complete.json'}:
            raise ValueError('Unsealed analysis artifact')
    ar=ev.read(audit/'audit_results.json'); ast=ev.read(audit/'started.json')
    us=ev.read(uncertainty/'started.json'); ur=ev.read(uncertainty/'uncertainty_results.json')
    if ar['status']!='passed' or ar['patients']!=current['patient_count'] or ar['windows']!=current['windows']:
        raise ValueError('Audit population/status mismatch')
    for record in (ar,ast,us): ev.check_hash(ev.OUTPUT/'complete.json',record['test_complete_sha256'])
    ev.check_hash(ev.OUTPUT/'test_results.json',ar['test_results_sha256'])
    ev.check_hash(ev.ROOT/'scripts/audit_test_completion.py',ast['source_sha256'])
    ev.check_hash(log,ar['console_sha256'])
    if ar['frozen_spec_sha256']!=ev.SPEC_HASH or ar['engineering_sha256']!=ev.ENGINEERING_HASH or ar['inference_executed'] or ar['fitting_executed']:
        raise ValueError('Audit scientific binding mismatch')
    audit_log=ev.BASE/'r0-final-v1-test-audit-v1.console.log'
    uncertainty_log=ev.BASE/'r0-final-v1-test-uncertainty.console.log'
    if 'TEST RESULT AUDIT COMPLETE' not in console_text(audit_log) or 'TEST UNCERTAINTY COMPLETE' not in console_text(uncertainty_log):
        raise ValueError('Analysis completion message missing')
    if us['source_sha256']!=ev.code_hashes() or us['unit']!='patient' or us['seed']!=ur['seed'] or us['seed']!=SEED or SEED!=101027 or us['replicates']!=ur['replicates'] or ur['replicates']!=REPLICATES or REPLICATES!=200:
        raise ValueError('Uncertainty protocol changed')
    files=sorted(uncertainty.glob('replicate-*.json')); rows=[ev.read(p) for p in files]
    if len(rows)!=200 or [r['replicate'] for r in rows]!=list(range(200)):
        raise ValueError('Missing/misaligned replicates')
    rng=np.random.default_rng(SEED)
    draws=np.array([rng.integers(0,len(patients),len(patients)) for _ in range(REPLICATES)])
    np.testing.assert_array_equal(np.load(uncertainty/'patient_draw_indices.npy'),draws)
    episodes=np.array([len(p['operational']['eligible_episodes']) for p in patients])
    hits=np.array([len(p['operational']['matched_episodes']) for p in patients])
    false=np.array([len(p['operational']['alerts']) for p in patients])-hits
    days=np.array([p['samples'] for p in patients])*task.grid_minutes/1440
    for draw,row in zip(draws,rows):
        w=np.bincount(draw,minlength=len(patients)); n=int(w@episodes)
        if row['episode_sensitivity']!=(float(w@hits/n) if n else None) or row['unmatched_alerts_per_supported_day']!=float(w@false/(w@days)):
            raise ValueError('Cluster operational replicate differs')
    for key in ur['intervals']:
        if interval([r[key] for r in rows])!=ur['intervals'][key]: raise ValueError('Interval arithmetic changed')
    if binomial_interval(int(hits.sum()),int(episodes.sum()))!=ur['episode_binomial_95_supplementary']:
        raise ValueError('Supplementary binomial interval differs')
    result=ev.read(ev.OUTPUT/'test_results.json')
    leads=sorted(a['lead_minutes'] for p in patients for a in p['operational']['alerts'] if a['episode_id'] is not None)
    np.testing.assert_array_equal(np.quantile(leads,[0,.25,.5,.75,1]),result['operational']['all']['lead_minutes_quantiles'])
    summary={'status':'verified_problem10_closed','discrimination':result['discrimination'],
        'operational':result['operational'],'calibration':result['calibration'],'threshold':result['threshold'],
        'alert_policy':result['alert_policy'],'coverage':result['fraction_eligible_supported_time_scored'],
        'lead_minutes_all_detections_sorted':leads,'uncertainty':ur,
        'validation_comparator':result['frozen_validation_comparator'],
        'further_test_access_required':False,'further_scoring_or_model_development_authorized':False,
        'verification_scope':'Saved full audit and uncertainty seals/bindings verified; exact seeded draws, operational bootstrap and interval arithmetic checked; no canonical shard pass, inference or fitting at closure.'}
    inputs=[ev.OUTPUT/'complete.json',ev.OUTPUT/'test_results.json',audit/'complete.json',audit/'audit_results.json',
        uncertainty/'complete.json',uncertainty/'uncertainty_results.json',ev.SPEC,ev.ENGINEERING,
        audit_log,uncertainty_log,ev.ROOT/'docs/problem10_test_results.md',Path(__file__)]
    output.mkdir()
    ev.write_json(output/'verified_summary.json',summary)
    ev.write_json(output/'paper_summary_manifest.json',{
        'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ev.ROOT,text=True).strip(),
        'inputs_sha256':{str(p.resolve().relative_to(ev.ROOT)):ev.digest(p) for p in inputs},
        'no_frozen_artifacts_modified':True,'test_predictions_recomputed':False,
        'method':'Aggregate publication handoff; no new statistical procedure'})
    seal(output,purpose='Problem 10 scientifically closed; immutable aggregate paper handoff')
    verify_seal(output)
    print('PROBLEM 10 CLOSURE VERIFIED AND SEALED; no test scoring or fitting.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True,type=Path)
    finalize(parser.parse_args().output)
