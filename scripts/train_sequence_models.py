#!/usr/bin/env python3
"""Train two sequence models on fixed grouped partitions; never split rows here.

Both modes use the same unaugmented validation set, identical network architecture,
class-balanced sampling, equal total weight per positive locus, and the same number
of training draws per epoch. Input tables must include fixed partition identifiers.
"""
import argparse
import bisect
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

FILES = ['train_summit_centered.tsv', 'train_position_augmented.tsv', 'validation.tsv']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sequence_key(seq):
    rc = seq.translate(str.maketrans('ACGTN', 'TGCAN'))[::-1]
    return min(seq, rc)


def check_table(df, expected_split):
    required = {'chrom', 'start', 'end', 'center_pos', 'anchor_pos', 'strand', 'label', 'sample_type',
                'augmented', 'sequence', 'locus_id', 'split_group', 'split', 'record_id', 'source_id'}
    if required - set(df):
        raise ValueError(f'Missing columns: {sorted(required - set(df))}; regenerate the dataset with the new builder')
    if df.empty or df[list(required)].isna().any().any():
        raise ValueError('Required fields are empty/missing')
    for c in ['label', 'augmented', 'start', 'end', 'center_pos', 'anchor_pos']:
        x = pd.to_numeric(df[c], errors='raise')
        if not np.isfinite(x).all() or (x % 1 != 0).any():
            raise ValueError(f'{c} must contain finite integers')
        df[c] = x.astype('int64')
    if not df.label.isin([0, 1]).all() or not df.augmented.isin([0, 1]).all():
        raise ValueError('Labels and augmented flags must be 0 or 1')
    if df.label.nunique() != 2 or not df.split.eq(expected_split).all():
        raise ValueError('Wrong partition labels or missing positive/negative class')
    if not df.strand.isin(['+', '-']).all() or (df.start < 0).any() or not (df.end-df.start).eq(201).all():
        raise ValueError('Invalid 201-bp windows or strand')
    if not df.center_pos.eq(df.start+101).all() or not df.end.eq(df.center_pos+100).all():
        raise ValueError('Window/center coordinates disagree')
    if not df.sequence.map(lambda x: isinstance(x, str) and len(x) == 201 and not (set(x)-set('ACGTN'))).all():
        raise ValueError('Every sequence must contain exactly 201 uppercase A/C/G/T/N bases')
    if df.assign(_sequence_key=df.sequence.map(sequence_key)).groupby('_sequence_key').label.nunique().gt(1).any():
        raise ValueError('Identical/reverse-complement sequences have conflicting labels')
    if ((df.augmented == 1) & (df.label != 1)).any() or df.record_id.duplicated().any():
        raise ValueError('Invalid augmentation or duplicate record identifiers')
    return df


def check_separation(train, val):
    if set(train.split_group) & set(val.split_group):
        raise ValueError('Training and validation share split_group identifiers')
    if set(train.sequence.map(sequence_key)) & set(val.sequence.map(sequence_key)):
        raise ValueError('Training and validation share identical or reverse-complement sequences')
    for column in ['source_id']:
        if set(train[column]) & set(val[column]):
            raise ValueError(f'Training and validation share {column}')
    # Interval overlap check is strand-independent.
    for chrom, v in val.groupby('chrom'):
        spans = sorted(zip(v.start, v.end))
        starts, ends, maximum = [], [], -1
        for start, end in spans:
            starts.append(start)
            maximum = max(maximum, end)
            ends.append(maximum)
        for row in train[train.chrom.eq(chrom)].itertuples():
            i = bisect.bisect_left(starts, row.end)-1
            if i >= 0 and ends[i] > row.start:
                raise ValueError(f'Training and validation windows overlap on {chrom}')
    if val.augmented.ne(0).any():
        raise ValueError('Validation must not contain augmented sequences')


def load_datasets(directory):
    cfg = json.loads((directory / 'dataset_config.json').read_text())
    if cfg.get('split_method') != 'overlap_and_sequence_grouped_before_augmentation':
        raise ValueError('Expected grouped-before-augmentation dataset configuration')
    tables = {}
    for name in FILES:
        expected = cfg.get('output_sha256', {}).get(name)
        if expected != sha256(directory / name):
            raise ValueError(f'{name}: missing/mismatched dataset checksum; do not mix runs')
        tables[name] = check_table(pd.read_csv(directory/name, sep='\t'), 'val' if name == 'validation.tsv' else 'train')
    base, aug, val = [tables[name] for name in FILES]
    if base.augmented.ne(0).any():
        raise ValueError('Summit-centered training must not contain augmentation')
    if base[base.label.eq(1)].locus_id.duplicated().any():
        raise ValueError('Base positives must contain one summit-centered row per locus')
    columns = ['chrom', 'start', 'end', 'strand', 'label', 'sample_type', 'sequence', 'locus_id', 'split_group', 'source_id']
    rowset = lambda d: set(d[columns].itertuples(index=False, name=None))
    if rowset(base) != rowset(aug[aug.augmented.eq(0)]) or rowset(base[base.label.eq(0)]) != rowset(aug[aug.label.eq(0)]):
        raise ValueError('The two training modes do not share the same base examples')
    if not set(aug[aug.label.eq(1)].locus_id) <= set(base[base.label.eq(1)].locus_id):
        raise ValueError('Augmented rows have unknown positive loci')
    if set(aug[aug.label.eq(1)].locus_id) & set(val[val.label.eq(1)].locus_id):
        raise ValueError('A positive anchor is present in training and validation')
    check_separation(base, val)
    check_separation(aug, val)
    return base, aug, val, cfg


def sampling_weights(df):
    labels = df.label.to_numpy()
    pos_counts = df[df.label.eq(1)].locus_id.value_counts()
    n_pos, n_neg = len(pos_counts), int((labels == 0).sum())
    weights = np.empty(len(df), dtype=np.float64)
    weights[labels == 0] = 0.5/n_neg
    weights[labels == 1] = [0.5/(n_pos*pos_counts[x]) for x in df.loc[df.label.eq(1), 'locus_id']]
    return weights


def train_mode(train, val, mode, out_dir, args, dataset_config, draws):
    # Import PyTorch only when fitting; --validate-only works without it.
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
    except ImportError as e:
        raise RuntimeError('PyTorch is required for training. See environment.yml') from e
    from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device)
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)

    class TETSSClassifier(nn.Module):
        # Parameter names and tensor shapes match src/tetss2/model.py.
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
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1))
        def forward(self, x):
            return self.classifier(self.features(x)).squeeze(1)

    def encode(df):
        # Compact uint8 storage avoids keeping all one-hot matrices in RAM.
        lookup = np.full(256, 4, dtype=np.uint8)
        for i, base in enumerate(b'ACGT'):
            lookup[base] = i
        encoded = np.stack([lookup[np.frombuffer(s.encode('ascii'), dtype=np.uint8)] for s in df.sequence])
        return TensorDataset(torch.from_numpy(encoded), torch.tensor(df.label.to_numpy(), dtype=torch.float32))

    def onehot(x):
        # N is represented as four zero channels.
        return torch.nn.functional.one_hot(x.long(), num_classes=5)[..., :4].permute(0, 2, 1).float()

    weights = sampling_weights(train)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), num_samples=draws, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(encode(train), batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)
    val_loader = DataLoader(encode(val), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    model = TETSSClassifier().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    out_dir.mkdir(parents=True, exist_ok=False)
    train.to_csv(out_dir/'train_split.tsv', sep='\t', index=False)
    val.to_csv(out_dir/'val_split.tsv', sep='\t', index=False)
    pd.DataFrame({'record_id': train.record_id, 'sampling_weight': weights}).to_csv(out_dir/'sampling_weights.tsv', sep='\t', index=False)

    @torch.no_grad()
    def evaluate():
        model.eval()
        probs, labels, total_loss = [], [], 0.0
        for x, y in val_loader:
            x, y = onehot(x.to(device)), y.to(device)
            logits = model(x)
            total_loss += float(criterion(logits, y))*len(y)
            probs.extend(torch.sigmoid(logits).cpu().tolist())
            labels.extend(y.cpu().tolist())
        predictions = np.asarray(probs) > args.threshold
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        metrics = dict(loss=total_loss/len(val), auroc=float(roc_auc_score(labels, probs)),
                       auprc=float(average_precision_score(labels, probs)), accuracy=float(accuracy_score(labels, predictions)),
                       precision=float(precision_score(labels, predictions, zero_division=0)), recall=float(recall_score(labels, predictions, zero_division=0)),
                       tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
        return metrics, probs

    history, best_auroc, best_epoch = [], -1.0, None
    for epoch in range(1, args.epochs+1):
        model.train()
        total_loss, observed = 0.0, 0
        for x, y in train_loader:
            x, y = onehot(x.to(device)), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())*len(y)
            observed += len(y)
        metrics, _ = evaluate()
        history.append(dict(epoch=epoch, train_loss=total_loss/observed, **metrics))
        pd.DataFrame(history).to_csv(out_dir/'train_history.tsv', sep='\t', index=False)
        print(f'{mode} epoch={epoch} train_loss={total_loss/observed:.5f} val_auroc={metrics["auroc"]:.5f}', flush=True)
        if metrics['auroc'] > best_auroc:
            best_auroc, best_epoch = metrics['auroc'], epoch
            torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, out_dir/'best_model.pth')
    model.load_state_dict(torch.load(out_dir/'best_model.pth', map_location=device, weights_only=True))
    metrics, probs = evaluate()
    (out_dir/'final_val_metrics.json').write_text(json.dumps(metrics, indent=2)+'\n')
    val.assign(probability=probs, prediction=(np.asarray(probs)>args.threshold).astype(int)).to_csv(out_dir/'val_predictions.tsv', sep='\t', index=False)
    config = {k: (v.name if isinstance(v, Path) else v) for k, v in vars(args).items()}
    config.update(mode=mode, best_epoch=best_epoch, model_selection_metric='val_auroc', device=str(device),
                  dataset_config=dataset_config, train_sha256=sha256(out_dir/'train_split.tsv'),
                  validation_sha256=sha256(out_dir/'val_split.tsv'), weight_sha256=sha256(out_dir/'best_model.pth'),
                  training_draws_per_epoch=draws, positive_class_total_sampling_weight=float(weights[train.label.to_numpy()==1].sum()),
                  negative_class_total_sampling_weight=float(weights[train.label.to_numpy()==0].sum()),
                  sampling='with replacement; class-balanced; equal total positive-locus weight',
                  software_versions={'python': __import__('platform').python_version(), 'torch': str(torch.__version__),
                                     'numpy': np.__version__, 'pandas': pd.__version__, 'sklearn': __import__('sklearn').__version__},
                  final_val_metrics=metrics)
    (out_dir/'run_config.json').write_text(json.dumps(config, indent=2)+'\n')
    summary = dict(train_rows=len(train), validation_rows=len(val), train_positive_loci=int(train[train.label.eq(1)].locus_id.nunique()),
                   train_counts=train.groupby(['label', 'sample_type', 'augmented']).size().reset_index(name='count').to_dict('records'),
                   validation_counts=val.groupby(['label', 'sample_type']).size().reset_index(name='count').to_dict('records'))
    (out_dir/'split_summary.json').write_text(json.dumps(summary, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--out-root', type=Path)
    p.add_argument('--modes', nargs='+', choices=['summit_centered', 'position_augmented'], default=['summit_centered', 'position_augmented'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--torch-threads', type=int, default=0, help='0: use the PyTorch default')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--validate-only', action='store_true', help='Validate split files without fitting or importing PyTorch')
    a = p.parse_args()
    if a.epochs < 1 or a.batch_size < 1 or a.lr <= 0 or a.weight_decay < 0 or not 0 <= a.threshold <= 1 or a.num_workers < 0 or a.torch_threads < 0:
        p.error('Invalid training settings')
    base, aug, val, cfg = load_datasets(a.dataset_dir)
    print(f'Partitions passed audit: base train={len(base)}, augmented train={len(aug)}, validation={len(val)}')
    if a.validate_only:
        return
    if a.out_root is None:
        p.error('--out-root is required for training')
    for mode in a.modes:
        if (a.out_root/mode).exists():
            p.error(f'Output already exists: {a.out_root/mode}; use a new directory')
    draws = 2*int(base.label.eq(1).sum())
    for mode in a.modes:
        train_mode(base if mode == 'summit_centered' else aug, val, mode, a.out_root/mode, a, cfg, draws)


if __name__ == '__main__':
    main()
