"""Compare all initial weights with the graph seed alone and with the Keras seed."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import StratifiedKFold

import run


def weights(model, cfg, seed, seed_keras):
    if seed_keras:
        run.tf.keras.utils.set_random_seed(seed)
    model._build(**cfg['graph'], batch_size=cfg['batch_size'], epochs_min=cfg['epochs'],
                 graph_seed=seed, suppress_output=True)
    runtime = run.tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    with model.graph_model.as_default(), run.tf.compat.v1.Session(config=runtime) as session:
        variables = run.tf.compat.v1.trainable_variables()
        session.run(run.tf.compat.v1.global_variables_initializer())
        return {v.name: hashlib.sha256(array.tobytes()).hexdigest()
                for v, array in zip(variables, session.run(variables))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path)
    parser.add_argument('--release-archive', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads((run.HERE / 'protocol.json').read_text())
    args.expected_csv_sha256 = cfg['cohort_sha256']
    csv = run.verify_inputs(cfg, args)
    data, groups, labels = run.read_cohort(csv, cfg)
    folds = StratifiedKFold(cfg['folds'], shuffle=True, random_state=cfg['random_seed'])
    train, _ = next(folds.split(np.zeros(len(labels)), labels))
    with TemporaryDirectory() as work:
        run.PRIVATE = Path(work)
        model, encoders, codes = run.encode(data, groups, labels, cfg)
        run.apply_fold_vocab(model, np.flatnonzero(np.isin(groups, train)), encoders, codes)
        model.device = '/device:CPU:0'
        seed = cfg['random_seed'] + 1
        results = {}
        for name, seed_keras in [('published', False), ('all_generators', True)]:
            first, repeat, changed = [weights(model, cfg, s, seed_keras) for s in (seed, seed, seed + 1)]
            results[name] = {'tensors': len(first),
                             'different_with_same_seed': [k for k in first if first[k] != repeat[k]],
                             'different_with_different_seed': [k for k in first if first[k] != changed[k]]}
    run.write_json(args.output, results)


if __name__ == '__main__':
    main()
