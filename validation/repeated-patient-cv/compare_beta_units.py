"""Fit the declared composition comparator under three ways of counting chains."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_repeats import composition, make_folds, measure_baseline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paired-rows', type=Path, required=True)
    parser.add_argument('--source-beta-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'repeat_plan.json').read_text())
    config = json.loads((root.parent / 'patient-grouped-deeptcr/protocol.json').read_text())
    jobs = [(plan['primary_folds'], seed) for seed in plan['patient_partition_seeds']]
    jobs.extend((k, plan['sensitivity_partition_seed']) for k in plan['sensitivity_folds'])
    paths = {'archived_pair_rows': args.paired_rows,
             'paired_cells': args.source_beta_dir / 'paired_cells.csv',
             'all_beta': args.source_beta_dir / 'all_beta.csv'}
    reference = None
    output = {}
    for unit, path in paths.items():
        df = pd.read_csv(path, usecols=['aminoAcid', 'patient_id', 'response_binary'])
        df = df[df.aminoAcid.str.len().le(config['max_length'])]
        patients = df.groupby('patient_id', sort=True).response_binary
        labels = patients.first()
        if not patients.nunique().eq(1).all() or len(labels) != config['patients']:
            raise ValueError('Ambiguous patient outcomes or cohort size')
        if reference is None:
            reference = labels
        elif not reference.equals(labels):
            raise ValueError('Patients or outcomes changed')
        counts = patients.size().to_numpy()
        features = composition(df.aminoAcid.to_numpy(), labels.index.get_indexer(df.patient_id), counts)
        runs = []
        for k, seed in jobs:
            splits = make_folds(labels.to_numpy(), k, seed)
            metrics = measure_baseline(features, labels.to_numpy(), splits, seed)
            runs.append({'folds': k, 'partition_seed': seed, **metrics})
        output[unit] = {'patients': len(labels), 'rows': len(df), 'runs': runs,
                        'primary_median_within_fold_auroc': float(np.median(
                             [entry['within_fold_auroc'] for entry in runs if entry['folds'] == plan['primary_folds']]))}
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
