"""Start the local browser interface for the shared task contract."""
import argparse
import errno
import json
import sys
from urllib.request import urlopen
from amsrr_det.task_server import serve


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    url=f'http://127.0.0.1:{args.port}'
    try:
        serve(args.port)
    except OSError as error:
        if error.errno!=errno.EADDRINUSE:
            raise
        try:
            with urlopen(url+'/api/catalog',timeout=2) as response:
                catalog=json.load(response)
            is_console=set(catalog.get('tasks',{}))=={'grasp','valve','walking'} and 'common' in catalog
        except (OSError,ValueError,TypeError,AttributeError):
            is_console=False
        if is_console:
            print(f'Task console is already running: {url}')
            return 0
        print(f'Port {args.port} is in use by another service. Choose another port with --port, for example --port {args.port+1}.',file=sys.stderr)
        return 1
    return 0


if __name__=='__main__':
    raise SystemExit(main())
