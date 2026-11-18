"""Select a dev checkpoint; refuse test metrics even when called accidentally."""
import argparse
import json
from pathlib import Path
from financeqa.common import write_json


def select(paths):
    runs = []
    protocol = None
    for path in paths:
        metrics = json.loads(Path(path).read_text())
        current = metrics['protocol']
        if current['split'] != 'dev':
            raise ValueError('Checkpoint selection requires dev, never test')
        if protocol is not None and current != protocol:
            raise ValueError('Checkpoint selection protocols differ')
        protocol = current
        if metrics['numerical_accuracy'] is None:
            raise ValueError('No scorable dev samples')
        metadata = json.loads((Path(path).parent/'run.json').read_text())
        variant = metadata['variant']
        adapter = metadata['config']['variants'][variant].get('adapter')
        if not adapter:
            raise ValueError('Checkpoint candidates must be adapters')
        runs.append({'adapter':adapter,'metrics':str(path), 'accuracy':metrics['numerical_accuracy'],
                     'n_scorable':metrics['n_scorable']})
    if not runs:
        raise ValueError('No checkpoint candidates')
    return {'selected':max(runs,key=lambda r:r['accuracy']), 'candidates':runs,
            'selection_split':'dev', 'tie_break':'first path supplied'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metrics',nargs='+')
    p.add_argument('--output',default='results/selected_checkpoint.json')
    a = p.parse_args()
    result = select(a.metrics); write_json(a.output,result)
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
