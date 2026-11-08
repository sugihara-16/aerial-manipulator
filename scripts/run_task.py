"""Run one shared JSON task, using fresh generation and measured latency."""
import argparse
import json
import traceback
from pathlib import Path
from amsrr_det.task_execution import execute,write


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    spec=None
    try:
        spec=json.loads(args.task.read_text())
        result=execute(spec,args.output)
        return 0 if result['success'] else 2
    except Exception as error:
        args.output.mkdir(exist_ok=True,parents=True)
        result=dict(success=False,reason=str(error),error_type=type(error).__name__,
                    task=spec.get('task') if isinstance(spec,dict) else None,mode=spec.get('execution',{}).get('mode') if isinstance(spec,dict) else None)
        if (args.output/'assembly/result.json').exists():
            result['assembly']=json.loads((args.output/'assembly/result.json').read_text());result['modules']=result['assembly'].get('modules')
        write(args.output/'result.json',result)
        expected=getattr(error,'task_failure',False)
        write(args.output/'status.json',dict(state='finished' if expected else 'error',message='Failed' if expected else str(error),result=result))
        traceback.print_exc()
        return 1


if __name__=='__main__':
    raise SystemExit(main())
