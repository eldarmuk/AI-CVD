"""Final R0 commands. No test-scoring command exists in Problem 9."""
import argparse
from pathlib import Path
from src.architecture_study.config import ROOT
from .config import DEFAULT,load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['train','verify','score-validation','develop-validation'])
    parser.add_argument('--config',default=str(DEFAULT))
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--model-run',type=Path)
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args(); spec,c,task = load(args.config)
    output = args.output.resolve()
    if not output.is_relative_to((ROOT/'runs/final_studies').resolve()):
        raise ValueError('Final artifacts must remain in ignored runs/final_studies')
    if args.resume and args.command != 'train':
        raise ValueError('Only completed training runs support verified skip')
    if args.command == 'train':
        from .training import train
        train(spec,c,task,output,args.resume)
    elif args.command == 'verify':
        from .training import verify_run
        verify_run(output,spec,c); print('Final model artifacts verified.')
    else:
        if args.model_run is None:
            parser.error('--model-run is required')
        from .validation import score,develop
        if args.command == 'score-validation':
            score(spec,c,task,args.model_run,output)
        else:
            if args.predictions is None:
                parser.error('--predictions is required')
            develop(spec,c,task,args.model_run,args.predictions,output)


if __name__ == '__main__':
    main()
