"""Verify the training cohort against GEO contigs and the publisher's Table S1A.

Reports aggregate counts only. A barcode identifies a cell within a library.
Comparisons retain duplicate records and run separately for every library.
"""
import argparse
from collections import Counter
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import re
import tarfile
from zipfile import ZipFile

import pandas as pd


def sample_id(name):
    match = re.search(r'_(ac\d+)(?:[./_]|$)', name, re.I)
    if not match:
        raise ValueError('Unrecognized GEO filename')
    return match.group(1).lower()


def records(frame, columns):
    return Counter(frame[columns].fillna('').itertuples(index=False, name=None))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('geo-tar', 'geo-manifest', 'clinical', 'contigs-zip', 'paired-csv', 'processed-csv'):
        ap.add_argument('--' + name, type=Path, required=True)
    ap.add_argument('--beta-sensitivity-dir', type=Path, help='Write private per-contig inputs to a new directory')
    args = ap.parse_args()
    if args.beta_sensitivity_dir:
        args.beta_sensitivity_dir.mkdir(mode=0o700, parents=True, exist_ok=False)
    sensitivity = {'paired_cells': [], 'all_beta': []}
    clinical = pd.read_excel(args.clinical, sheet_name='Table S1A', header=1)
    clinical = clinical.set_index('Sample ID', verify_integrity=True)['3mo PET/CT']
    labels = clinical.map({'CR': 1, 'PR': 1, 'PD': 0, 'SD': 0}).dropna()
    paired = pd.read_csv(args.paired_csv)
    processed = pd.read_csv(args.processed_csv)
    counts = Counter()
    diffs = Counter()
    manifest = pd.read_csv(args.geo_manifest, sep='\t')
    expected = manifest.loc[manifest['#Archive/File'] == 'File'].set_index('Name')['Size'].to_dict()
    geo_ids = set()
    sequenced = set()
    beta = ['barcode','v_gene','j_gene','cdr3']
    with ZipFile(args.contigs_zip) as z, tarfile.open(args.geo_tar, 'r:') as tar:
        zip_paths = {sample_id(n): n for n in z.namelist() if n.endswith('/filtered_contig_annotations.csv') and '__MACOSX' not in n}
        members = [m for m in tar.getmembers() if m.isfile()]
        if {m.name: m.size for m in members} != expected:
            raise ValueError('Member names/sizes differ from GEO manifest')
        for m in members:
            sid = sample_id(m.name)
            if sid in geo_ids:
                raise ValueError('Repeated GEO library')
            geo_ids.add(sid)
            data = None
            with tarfile.open(fileobj=tar.extractfile(m), mode='r|gz') as child:
                for part in child:
                    if Path(part.name).name == 'filtered_contig_annotations.csv':
                        data = child.extractfile(part).read()
                        break
            if data is None:
                continue
            sequenced.add(sid)
            if sid in zip_paths and sha256(z.read(zip_paths[sid])).digest() == sha256(data).digest():
                counts['contigs_identical_between_tar_and_zip'] += 1
            if sid not in labels:
                continue
            df = pd.read_csv(BytesIO(data), low_memory=False)
            df = df[df.productive.astype(str).str.lower().eq('true')]
            b = df[df.chain.eq('TRB')]
            a = df[df.chain.eq('TRA')]
            truth = processed[processed.patient_id.eq(sid)]
            p = paired[paired.patient_id.eq(sid)]
            counts['incorrect_binary_labels'] += int((truth.response_binary != labels[sid]).sum())
            expected_text = 'Responder' if labels[sid] else 'Non-responder'
            counts['incorrect_text_labels'] += int((truth.response != expected_text).sum())
            counts['eligible_patients_with_beta'] += int(not b.empty)
            counts['processed_rows'] += len(truth)
            counts['productive_beta_contigs_in_evaluable_libraries'] += len(b)
            counts['processed_unique_cells'] += truth.barcode.nunique()
            counts['processed_cells_with_multiple_rows'] += int((truth.barcode.value_counts() > 1).sum())
            counts['length_exclusions_over_40'] += int(truth.aminoAcid.str.len().gt(40).sum())
            observed = records(truth, ['barcode','vGeneName','jGeneName','aminoAcid'])
            paired_beta = records(p, ['barcode','TRB_v_gene','TRB_j_gene','TRB_cdr3'])
            diffs['paired_csv_to_processed_beta'] += sum((observed - paired_beta).values()) + sum((paired_beta - observed).values())
            aa = a[beta].rename(columns={c: 'alpha_' + c for c in beta[1:]})
            both = aa.merge(b[beta], on='barcode')
            for name, frame in [('paired_alpha_beta_join', both), ('all_productive_beta', b)]:
                tally = records(frame, beta)
                diffs[name + '_missing_from_processed'] += sum((tally - observed).values())
                diffs[name + '_extra_in_processed'] += sum((observed - tally).values())
            paired_cells = b[b.barcode.isin(a.barcode)]
            for name, frame in [('paired_cells', paired_cells), ('all_beta', b)]:
                counts[name + '_beta_contigs'] += len(frame)
                counts[name + '_cells'] += frame.barcode.nunique()
                counts[name + '_eligible_beta_contigs'] += int(frame.cdr3.str.len().le(40).sum())
                if args.beta_sensitivity_dir:
                    sensitivity[name].append(frame[beta].assign(patient_id=sid, response_binary=int(labels[sid])))
            source_pairs = records(both, ['barcode','alpha_v_gene','alpha_j_gene','alpha_cdr3','v_gene','j_gene','cdr3'])
            repo_pairs = records(p,['barcode','TRA_v_gene','TRA_j_gene','TRA_cdr3','TRB_v_gene','TRB_j_gene','TRB_cdr3'])
            diffs['source_alpha_beta_tuples_absent_from_paired_csv'] += sum((source_pairs - repo_pairs).values())
            diffs['paired_alpha_beta_tuples_absent_from_source_join'] += sum((repo_pairs - source_pairs).values())
    counts['geo_ids_missing_clinical'] = len(geo_ids - set(clinical.index))
    counts['clinical_ids_missing_geo'] = len(set(clinical.index) - geo_ids)
    counts['processed_ids_without_evaluable_sequenced_source'] = len(set(processed.patient_id) - (sequenced & set(labels.index)))
    counts.update(geo_libraries=len(geo_ids), clinical_records=len(clinical), evaluable_clinical_records=len(labels), geo_libraries_with_contigs=len(sequenced), zip_contig_files=len(zip_paths), evaluable_without_contigs=len(set(labels.index) - sequenced), processed_patients=processed.patient_id.nunique(), paired_rows=len(paired))
    required_zero = [counts['incorrect_binary_labels'], counts['incorrect_text_labels'],
                     counts['geo_ids_missing_clinical'], counts['clinical_ids_missing_geo'],
                     counts['processed_ids_without_evaluable_sequenced_source'],
                     diffs['paired_csv_to_processed_beta'],
                     diffs['paired_alpha_beta_join_missing_from_processed'],
                     diffs['paired_alpha_beta_join_extra_in_processed'],
                     diffs['source_alpha_beta_tuples_absent_from_paired_csv'],
                     diffs['paired_alpha_beta_tuples_absent_from_source_join']]
    if any(required_zero) or counts['contigs_identical_between_tar_and_zip'] != len(sequenced) or set(zip_paths) != sequenced:
        raise ValueError('Clinical labels, GEO contigs, or paired rows do not agree')
    if args.beta_sensitivity_dir:
        for name, frames in sensitivity.items():
            pd.concat(frames, ignore_index=True).rename(columns={'cdr3':'aminoAcid', 'v_gene':'vGeneName', 'j_gene':'jGeneName'}).to_csv(args.beta_sensitivity_dir / (name + '.csv'), index=False)
    print(json.dumps({'clinical_table_sha256': sha256(args.clinical.read_bytes()).hexdigest(), 'three_month_imaging': clinical.fillna('missing').value_counts().to_dict(), 'counts': dict(counts), 'multiset_differences': dict(diffs)}, indent=2))


if __name__ == '__main__':
    main()
