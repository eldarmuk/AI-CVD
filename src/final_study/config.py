import copy
import tomllib
from pathlib import Path
from src.architecture_study.config import ROOT, load_config, fingerprint
from src.architecture_study.data import digest, write_json

DEFAULT = ROOT/'configs/final_studies/r0_final_v1.toml'


def runtime():
    import platform
    import subprocess
    from importlib.metadata import version
    return {'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'python':platform.python_version(),'packages':{p:version(p) for p in ('torch','numpy','pandas')},
            'source_sha256':source_hashes()}


def load(path=DEFAULT):
    with open(path,'rb') as f:
        spec = tomllib.load(f)
    base, task = load_config(ROOT/spec['base_config'])
    if fingerprint(base) != spec['base_config_sha256'] or spec['task_identifier'] != task.identifier:
        raise ValueError('Base/task definition changed')
    if spec['recipe'] != 'R0' or spec['projection'] != base['projection'] or spec['version'] != '1.0.0':
        raise ValueError('Only frozen final R0 is supported')
    # This version supports exactly the reviewed protocol, not a parameter search.
    with open(DEFAULT,'rb') as f:
        approved = tomllib.load(f)
    if spec != approved:
        raise ValueError('Unapproved final specification')
    c = copy.deepcopy(base)
    c.update(recipes=['R0'],seed=spec['seed'],canonical_run=spec['canonical_run'])
    c['resources'].update(ssl_updates=spec['ssl_draws'],checkpoint_every=spec['checkpoint_every'],
        checkpoint_batches=spec['checkpoint_batches'],head_iterations=spec['head_max_iter'],
        batch_size=spec['batch_size'],cpu_threads=spec['cpu_threads'])
    c['final_spec_sha256'] = fingerprint(spec)
    return spec,c,task


def source_hashes():
    paths = [p for package in ('final_study','architecture_study')
             for p in sorted((ROOT/'src'/package).glob('*.py'))]
    paths += [ROOT/'src/ai_cvd'/name for name in ('task.py','dataset.py','features.py','episodes.py')]
    return {str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in paths}


def seal(folder, **extra):
    write_json(folder/'complete.json', {'status':'complete', **extra,
        'artifacts_sha256':{str(p.relative_to(folder)).replace('\\','/'):digest(p)
                           for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix != '.log'}})


def verify_seal(folder):
    import json
    marker = json.loads((folder/'complete.json').read_text())
    if marker['status'] != 'complete':
        raise ValueError('Incomplete artifact')
    for name,h in marker['artifacts_sha256'].items():
        p = (folder/name).resolve()
        if not p.is_relative_to(folder.resolve()) or digest(p) != h:
            raise ValueError(f'Stale artifact: {name}')
    return marker
