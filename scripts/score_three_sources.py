#!/usr/bin/env python3
"""Score RAMPAGE, TE-TSS1.0 and single-cell TE-TSS candidates.

The default preprocessing reproduces the supplied source script: 1-based
RAMPAGE summits; floor((start+end)/2) interpreted as a 1-based TE-TSS1.0
center; and a 1-based single-cell summit in the last pipe-delimited BED name
field. These are explicit compatibility defaults, not inferred source-format
facts. Verify the source definitions before selecting another coordinate mode.

Inputs:
  RAMPAGE: headered TSV containing Peak_Chrom, Peak_Summit, Peak_Strand.
  TE-TSS1.0: headerless 9-column TSV: chr,start,end,strand,type,transposon,
             gene_id,transcript_id,sample. Lines beginning with # are ignored.
  Single-cell: headerless 12-column TE-intersection BED, with the first six
               columns for a TSS candidate and the final six for its TE match.
               The fourth field ends in |<summit>.

Windows are 201 bp, negative-strand sequences are reverse complemented, and
only A/C/G/T windows are scored. Retention uses a strict score > threshold.
Existing output filenames and principal columns are retained for downstream
compatibility. An optional reference table audits sequence consistency only;
it does not demonstrate independent evaluation or absence of training overlap.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pyfaidx import Fasta

WINDOW_SIZE = 201
RADIUS = 100


class TETSSClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=15, padding=7),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 128, kernel_size=15, padding=7),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=7, padding=3),
            nn.BatchNorm1d(256), nn.ReLU(), nn.AdaptiveMaxPool1d(1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def reverse_complement(seq):
    return seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]


def seq_to_onehot(seq):
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, base in enumerate(seq):
        arr[{'A': 0, 'C': 1, 'G': 2, 'T': 3}[base], i] = 1.0
    return arr


def numeric_centers(values, base):
    result = pd.to_numeric(values, errors='coerce').astype(float)
    result = result.where(np.isfinite(result) & result.eq(np.floor(result)))
    return result + (1 if base == 0 else 0)


def read_unheaded(path, names):
    try:
        frame = pd.read_csv(path, sep='\t', header=None, comment='#')
    except pd.errors.EmptyDataError:
        return pd.DataFrame(columns=names)
    if frame.shape[1] != len(names):
        raise ValueError(f'{path.name}: expected {len(names)} headerless columns, got {frame.shape[1]}')
    frame.columns = names
    return frame


def read_sources(args):
    rampage = pd.read_csv(args.rampage, sep='\t')
    required = {'Peak_Chrom', 'Peak_Summit', 'Peak_Strand'}
    if required - set(rampage):
        raise ValueError(f'RAMPAGE missing columns: {sorted(required-set(rampage))}')
    reserved = {'chrom_core', 'summit_core', 'strand_core', 'sequence_201bp', 'TETSS_Score', 'invalid_reason'}
    if reserved & set(rampage):
        raise ValueError('RAMPAGE input contains generated scoring columns; provide the original source table')
    rampage = rampage.copy()
    rampage['chrom_core'] = rampage.Peak_Chrom
    rampage['summit_core'] = numeric_centers(rampage.Peak_Summit, args.rampage_summit_base)
    rampage['strand_core'] = rampage.Peak_Strand
    t1 = read_unheaded(args.tetss1, ['chr', 'start', 'end', 'strand', 'type', 'transposon', 'gene_id', 'transcript_id', 'sample'])
    start, end = numeric_centers(t1.start, 1), numeric_centers(t1.end, 1)
    if args.tetss1_coordinates == 'zero-based-half-open':
        # Use the lower of the two middle genomic bases when the width is even.
        centers = np.floor((start+end-1)/2)+1
        centers = centers.where((start >= 0) & (end > start))
    else:
        centers = np.floor((start+end)/2)
        minimum = 1 if args.tetss1_coordinates == 'one-based-inclusive' else 0
        centers = centers.where((start >= minimum) & (end >= start))
    t1['chrom_core'], t1['summit_core'], t1['strand_core'] = t1['chr'], centers, t1.strand
    sources = {'RAMPAGE': rampage, 'TETSS1_0': t1}
    raw_counts = {'RAMPAGE': len(rampage), 'TETSS1_0': len(t1)}
    if args.single_cell:
        names = ['camo_chr', 'camo_start', 'camo_end', 'camo_name', 'camo_score', 'camo_strand',
                 'te_chr', 'te_start', 'te_end', 'te_info', 'te_score', 'te_strand']
        single = read_unheaded(args.single_cell, names)
        raw_counts['CamoTSS'] = len(single)
        last_field = single.camo_name.astype('string').str.rsplit('|', n=1).str[-1]
        single['summit_core'] = numeric_centers(last_field, args.single_cell_summit_base)
        single['chrom_core'], single['strand_core'] = single.camo_chr, single.camo_strand
        keys = ['chrom_core', 'summit_core', 'strand_core', 'camo_name', 'camo_chr', 'camo_start', 'camo_end', 'camo_strand']
        # Match the source grouping rule: consolidate TE annotations for one candidate.
        single = single.groupby(keys, dropna=False).agg(
            te_info=('te_info', lambda x: ','.join(pd.Series(x).dropna().astype(str).unique()))).reset_index()
        sources['CamoTSS'] = single
    return sources, raw_counts


def extract_sequence(fa, chrom, center, strand):
    if pd.isna(center) or not np.isfinite(center) or center != int(center) or center < 1 or strand not in ['+', '-'] or not chrom:
        return '', 'invalid_metadata'
    center = int(center)
    start, end = center-101, center+100  # 0-based half-open extraction.
    if chrom not in fa:
        return '', 'chromosome_not_in_fasta'
    if start < 0 or end > len(fa[chrom]):
        return '', 'window_out_of_bounds'
    seq = str(fa[chrom][start:end]).upper()
    if len(seq) != WINDOW_SIZE:
        return '', 'unexpected_sequence_length'
    if strand == '-':
        seq = reverse_complement(seq)
    if set(seq)-set('ACGT'):
        return seq, 'non_acgt_bases'
    return seq, ''


def predict_dataframe(meta, prefix, model, fa, out_dir, device, batch_size, threshold):
    df = meta.copy().reset_index(drop=True)
    df['chrom_core'] = df.chrom_core.astype('string').fillna('').str.strip()
    df['strand_core'] = df.strand_core.astype('string').fillna('').str.strip()
    values = [extract_sequence(fa, row.chrom_core, row.summit_core, row.strand_core) for row in df.itertuples()]
    df['sequence_201bp'] = [x[0] for x in values]
    reasons = pd.Series([x[1] for x in values], index=df.index, dtype='string')
    invalid = df.loc[reasons.ne('')].copy()
    invalid['invalid_reason'] = reasons.loc[invalid.index]
    valid = df.loc[reasons.eq('')].copy().reset_index(drop=True)
    scores = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(valid), batch_size):
            seqs = valid.sequence_201bp.iloc[i:i+batch_size]
            tensor = torch.from_numpy(np.stack([seq_to_onehot(s) for s in seqs])).to(device)
            probs = torch.sigmoid(model(tensor)).cpu().numpy()
            if not np.isfinite(probs).all():
                raise ValueError(f'{prefix}: model returned nonfinite scores')
            scores.extend(probs)
    valid['TETSS_Score'] = np.asarray(scores, dtype=np.float32)
    high = valid[valid.TETSS_Score > threshold].copy()
    for suffix, table in [('all_preds', valid), ('high_conf_preds', high), ('invalid_sequences', invalid)]:
        table.to_csv(out_dir/f'{prefix}_fixed_{suffix}.tsv', sep='\t', index=False)
    summary = {'input_rows': len(df), 'valid_rows': len(valid), 'retained_rows': len(high),
               'invalid_rows': len(invalid), 'invalid_reason_counts': invalid.invalid_reason.value_counts().to_dict()}
    print(f'{prefix}: input={len(df)}, valid={len(valid)}, retained={len(high)}, invalid={len(invalid)}', flush=True)
    return valid, summary


def audit_reference(rampage, reference_path, out_dir):
    reference = pd.read_csv(reference_path, sep='\t')
    required = {'chrom', 'center_pos', 'strand', 'sequence'}
    if required-set(reference):
        raise ValueError(f'Reference missing columns: {sorted(required-set(reference))}')
    if 'label' in reference:
        reference = reference[pd.to_numeric(reference.label, errors='raise').eq(1)].copy()
    if 'augmented' in reference:
        reference = reference[pd.to_numeric(reference.augmented, errors='raise').eq(0)].copy()
    reference['chrom'] = reference.chrom.astype('string').str.strip()
    reference['strand'] = reference.strand.astype('string').str.strip()
    centers = numeric_centers(reference.center_pos, 1)
    if centers.isna().any() or (centers < 1).any() or not reference.strand.isin(['+', '-']).all():
        raise ValueError('Reference requires 1-based integer center_pos and + or - strand')
    reference['center_pos'] = centers.astype('int64')
    # Disagreeing sequences for the same coordinate cannot be silently collapsed.
    if reference.groupby(['chrom', 'center_pos', 'strand']).sequence.nunique().gt(1).any():
        raise ValueError('Reference has conflicting sequences for one genomic coordinate')
    reference = reference[['chrom', 'center_pos', 'strand', 'sequence']].drop_duplicates(['chrom', 'center_pos', 'strand'])
    merged = rampage.merge(reference, left_on=['chrom_core', 'summit_core', 'strand_core'],
                           right_on=['chrom', 'center_pos', 'strand'], how='inner')
    merged['sequence_exact_match'] = merged.sequence_201bp.astype(str).eq(merged.sequence.astype(str))
    merged.to_csv(out_dir/'RAMPAGE_reference_sequence_audit.tsv', sep='\t', index=False)
    return {'reference_rows': len(reference), 'matched_rows': len(merged),
            'exact_sequence_matches': int(merged.sequence_exact_match.sum()),
            'exact_sequence_match_pct': float(100*merged.sequence_exact_match.mean()) if len(merged) else None,
            'purpose': 'sequence consistency against the supplied reference; not an independence test'}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for flag in ['fasta', 'model', 'rampage', 'tetss1', 'out-dir']:
        p.add_argument('--'+flag, type=Path, required=True)
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument('--single-cell', type=Path, help='12-column candidate-TE intersection BED')
    group.add_argument('--skip-single-cell', action='store_true', help='Explicitly score the two other sources only')
    p.add_argument('--reference-tsv', type=Path, help='Optional sequence-consistency audit; center_pos must be 1-based')
    p.add_argument('--rampage-summit-base', type=int, choices=[0, 1], default=1)
    p.add_argument('--single-cell-summit-base', type=int, choices=[0, 1], default=1)
    p.add_argument('--tetss1-coordinates', choices=['source-midpoint', 'one-based-inclusive', 'zero-based-half-open'], default='source-midpoint',
                   help='Default preserves the source midpoint arithmetic; select another convention only after verifying the input definition')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--batch-size', type=int, default=512)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--torch-threads', type=int, default=0, help='0 uses the runtime default')
    p.add_argument('--hash-fasta', action='store_true', help='Also hash the potentially large reference FASTA')
    args = p.parse_args()
    if not 0 <= args.threshold <= 1 or args.batch_size < 1 or args.torch_threads < 0:
        p.error('Invalid threshold, batch size or thread count')
    inputs = {k: getattr(args, k) for k in ['fasta', 'model', 'rampage', 'tetss1', 'single_cell', 'reference_tsv'] if getattr(args, k)}
    for name, path in inputs.items():
        if not path.is_file():
            p.error(f'{name}: file not found: {path}')
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        p.error('Output directory must be empty; choose a new directory')
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    if args.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA requested but not available')
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    sources, raw_counts = read_sources(args)
    model = TETSSClassifier().to(device)
    state = torch.load(args.model, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    fa = Fasta(str(args.fasta), as_raw=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for prefix, frame in sources.items():
        scored, counts = predict_dataframe(frame, prefix, model, fa, args.out_dir, device, args.batch_size, args.threshold)
        counts['raw_source_rows'] = raw_counts[prefix]
        summary[prefix] = counts
        if prefix == 'RAMPAGE' and args.reference_tsv:
            summary['RAMPAGE_reference_sequence_audit'] = audit_reference(scored, args.reference_tsv, args.out_dir)
    if args.skip_single_cell:
        summary['CamoTSS'] = {'status': 'explicitly_skipped'}
    (args.out_dir/'fixed_prediction_run_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    config = {k: (v.name if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config['inputs'] = {k: {'filename': path.name, 'size_bytes': path.stat().st_size,
                          'sha256': sha256(path) if k != 'fasta' or args.hash_fasta else None} for k, path in inputs.items()}
    config.update(device=str(device), score_rule='sigmoid(logit); retain score > threshold', sequence_length=WINDOW_SIZE,
                  versions={'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__, 'torch': str(torch.__version__)})
    (args.out_dir/'prediction_run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    fa.close()
    print(f'Outputs written to {args.out_dir}')


if __name__ == '__main__':
    main()
