# TETSS2.0

**TETSS2.0** is a deep learning model for predicting TE-derived transcription start site (TE-TSS) activity from DNA sequences.

The Python package name and command-line tool name are **`tetss2`**. The model name shown in documents, figures, and the website is **TETSS2.0**.

---

## Overview

TETSS2.0 is a PyTorch-based convolutional neural network classifier designed to predict whether an input DNA sequence is associated with TE-TSS activity.

The model takes a DNA sequence as input and returns:

* a prediction probability
* a binary prediction label
* the classification threshold used for prediction

By default, TETSS2.0 uses a threshold of `0.5`.

---

## Input requirement

TETSS2.0 expects DNA sequences of exactly **201 bp**.

Allowed bases:

```text
A, C, G, T, N
```

Notes:

* Input sequences are automatically converted to uppercase.
* `N` is allowed but is encoded as an all-zero position in the one-hot representation.
* Sequences shorter or longer than 201 bp are rejected by default.
* The option `--no-length-check` is available only for debugging and is not recommended for normal prediction.

---

## Output

For each input sequence, TETSS2.0 outputs:

| Column                   | Description                              |
| ------------------------ | ---------------------------------------- |
| `tetss2_sequence_length` | Length of the input sequence             |
| `tetss2_probability`     | Predicted probability score              |
| `tetss2_prediction`      | Binary prediction result, `0` or `1`     |
| `tetss2_threshold`       | Threshold used for binary classification |

---

## Installation

### Option 1: Install from a local source directory

If you have downloaded or cloned this package locally, enter the package directory and install it with:

```bash
cd tetss_rampage_package
pip install -e .
```

After installation, check whether the command-line tool is available:

```bash
tetss2 --help
```

### Option 2: Recommended conda environment

We recommend creating a clean conda environment before installation:

```bash

conda create -n tetss2 python=3.9
conda activate tetss2
conda install numpy pandas scikit-learn
pip install torch==1.10.2

pip install -e .

```

A future public release may support:

```bash
pip install tetss2
```

---

## Command-line usage

### 1. Predict a single sequence

```bash
tetss2 predict --sequence ACGTACGTACGTACGT
```

For normal use, the input sequence should be exactly 201 bp:

```bash
tetss2 predict --sequence YOUR_201BP_DNA_SEQUENCE
```

Example output:

```json
{
  "model": "TETSS2.0",
  "sequence": "ACGTACGTACGTACGT",
  "sequence_length": 16,
  "probability": 0.35490313172340393,
  "prediction": 0,
  "threshold": 0.5
}
```

Note: the example above uses a short sequence only to demonstrate the command format. For biological prediction, please use a 201 bp sequence.

---

### 2. Batch prediction from a TSV file

Prepare an input file containing a `sequence` column.

Example input file: `input.tsv`

```tsv
sample_id	sequence
sample1	ACGT...
sample2	TTTT...
sample3	GCGC...
```

Run batch prediction:

```bash
tetss2 predict-file \
  --input input.tsv \
  --output tetss2_predictions.tsv
```

The output file will contain the original columns plus TETSS2.0 prediction results.

Example output:

```tsv
sample_id	sequence	tetss2_sequence_length	tetss2_probability	tetss2_prediction	tetss2_threshold
sample1	ACGT...	201	0.3549	0	0.5
sample2	TTTT...	201	0.8123	1	0.5
sample3	GCGC...	201	0.4471	0	0.5
```

---

### 3. Batch prediction from a CSV file

If your input file is comma-separated, use `--sep ","`:

```bash
tetss2 predict-file \
  --input input.csv \
  --output tetss2_predictions.csv \
  --sep ","
```

---

### 4. Use a custom sequence column name

If the sequence column is not named `sequence`, specify it with `--sequence-column`.

For example, if the input file contains a column named `dna`:

```bash
tetss2 predict-file \
  --input input.tsv \
  --output tetss2_predictions.tsv \
  --sequence-column dna
```

---

## Python API usage

TETSS2.0 can also be used directly in Python.

```python
from tetss2 import TETSS2Predictor

predictor = TETSS2Predictor()

result = predictor.predict("YOUR_201BP_DNA_SEQUENCE")
print(result)
```

Example output:

```python
{
    "model": "TETSS2.0",
    "sequence": "YOUR_201BP_DNA_SEQUENCE",
    "sequence_length": 201,
    "probability": 0.73,
    "prediction": 1,
    "threshold": 0.5,
}
```

---

## Model architecture

TETSS2.0 uses a one-dimensional convolutional neural network for DNA sequence classification.

The model contains:

* one-hot encoding of DNA sequences
* multiple 1D convolutional layers
* batch normalization
* ReLU activation
* max pooling
* adaptive max pooling
* fully connected classification layers

The model outputs a single logit, which is converted to a probability using the sigmoid function.

---

## Model files

The package includes the trained model weight file:

```text
best_model.pth
```

The original training output directory also contains:

```text
best_model.pth
run_config.json
train_history.tsv
final_val_metrics.json
split_summary.json
train_split.tsv
val_split.tsv
```

These files can be used to document model configuration, validation performance, and data splitting information.

---

## Validation performance

Please fill in the following values using `final_val_metrics.json`:

| Metric          | Value |
| --------------- | ----: |
| AUROC           |  TODO |
| AUPRC           |  TODO |
| Accuracy        |  TODO |
| Precision       |  TODO |
| Recall          |  TODO |
| True negatives  |  TODO |
| False positives |  TODO |
| False negatives |  TODO |
| True positives  |  TODO |

To view the metrics file:

```bash
cat final_val_metrics.json
```

---

## Troubleshooting

### MKL threading error

If you see an error similar to:

```text
mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp.so.1
```

try running:

```bash
export MKL_THREADING_LAYER=GNU
```

Then run the prediction command again.

---

### Cannot find sequence column

If you see an error like:

```text
Cannot find sequence column 'sequence'
```

check whether your input file contains a real tab or comma separator.

For a TSV file, you can check tabs with:

```bash
cat -A input.tsv
```

A real tab will appear as:

```text
^I
```

A correct TSV file should look like:

```text
sample_id^Isequence$
sample1^IACGT...$
```

---

## Citation

If you use TETSS2 in your research, please cite:

@software{tetss2,
  title={TETSS2: Deep Learning Model for TE-TSS Prediction},
  year={2026}
}

---

## Contact

For questions or issues, please contact the developer or open an issue in the project repository.

---

## License

License information will be added later.

## Repository

https://github.com/MoriyaaCui/TETSS2.git


## Live Demo

A Gradio demo is available:

```bash
python demo/app.py