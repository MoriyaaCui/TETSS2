# TETSS2

TETSS2 is an integrated resource of human transposable element-derived
transcription start sites (TE-TSSs), combining RAMPAGE-derived evidence,
human records from TE-TSS1.0, and candidates obtained by processing
5′ single-cell RNA-sequencing data.

This repository contains the `tetss2` sequence-prediction toolkit and scripts
for preparing model-development datasets and training sequence models.
The model is referred to as **TETSS2.0** in the predictor interface; the Python
package and command-line program are named **`tetss2`**.

## Repository contents

| Location | Purpose |
| --- | --- |
| `src/tetss2/` | Model architecture, sequence encoding, Python predictor and CLI |
| `src/tetss2/assets/best_model.pth` | Pretrained model parameters bundled with the predictor |
| `scripts/` | Data preparation and model-training scripts |
| `environment.yml` | Environment specification |
| `pyproject.toml` and `MANIFEST.in` | Package installation and model-resource packaging |
| `run_config.json` and `final_val_metrics.json` | Available training configuration and validation metrics |
| `test_tetss2_length_check.py` | Manual sequence-length checking example |

## Sequence model

The predictor uses a PyTorch 1D convolutional neural network to assign a
sequence-based TE-TSS prioritization score to a strand-oriented 201-bp DNA
sequence. It returns a score between 0 and 1 and a binary label at a configurable
threshold (default: 0.5). A sequence-model score is not a direct experimental
measurement of transcriptional activity.

## Data preparation and training scripts

| Script | Purpose |
| --- | --- |
| `scripts/filter_rampage_for_modeling.py` | Filter RAMPAGE records and normalize explicitly specified coordinate conventions |
| `scripts/generate_position_augmented_dataset.py` | Construct positive/background examples, assign fixed partitions and augment training positives |
| `scripts/train_sequence_models.py` | Train summit-centered and position-augmented sequence models on the supplied partitions |

Inputs, outputs and configurable arguments are shown by each script's help:

```bash
python scripts/filter_rampage_for_modeling.py --help
python scripts/generate_position_augmented_dataset.py --help
python scripts/train_sequence_models.py --help
```

For these development scripts, install the additional dependencies:

```bash
python -m pip install numpy pandas scikit-learn pyfaidx
```

The predictor can be used independently of the development scripts.

## Installation (Recommended)

To avoid dependency conflicts and bypass massive GPU driver downloads for standard prediction tasks, we highly recommend installing `tetss2` in a clean Conda environment with a CPU-only version of PyTorch.

**1. Create a clean environment (Python 3.10 or later):**
```bash
conda create -n tetss2_env python=3.11 pip -y
conda activate tetss2_env
```

**2. Install a lightweight, CPU-only PyTorch:**
```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**3. Install TETSS2 directly from GitHub:**
```bash
python -m pip install git+https://github.com/MoriyaaCui/TETSS2.git
```

---

## Input Requirements

To ensure biological and mathematical validity, input data must strictly adhere to the following specifications:
* **Length:** Exactly **201 bp**. Shorter or longer sequences will be rejected by default.
* **Allowed Bases:** `A, C, G, T, N` (case-insensitive; all inputs are automatically converted to uppercase).
* **Unknowns:** `N` bases are allowed and are safely encoded as all-zero vectors in the one-hot representation.

---

## Command-Line Usage (CLI)

Once installed, the `tetss2` command becomes globally available in your terminal.

### 1. Predict a Single Sequence
Use the `predict` mode for a quick evaluation of a 201 bp sequence:

```bash
tetss2 predict --sequence ATGCTCTGTGAGGCCGTGCTTGGCCGAGTCAGGAAGGGCCCCACGCGTGCCGTAGGCCGTAGGTAGGCGGGTAGGCGAGTGGCGGAGGCCCAGGGCCTGCCAGCTGGGTAGGCGAGGGCGGCGGGAGGCCTGGGGCCTGCGCGTGCGGCGGAGGCCTGGAGCCTGCCAGCTGGGCGGGCGAGGGCGGCGGGAGGCCTGGGG
```
The JSON output contains `model`, `sequence`, `sequence_length`,
`probability`, `prediction`, and `threshold`. The score depends on the input
sequence and checkpoint; the default positive label uses score > 0.5.

### 2. High-Throughput Batch Prediction
Use the `predict-file` mode for large-scale screenings from TSV or CSV files. The input file must contain a column named `sequence` (or specify the column name with `--sequence-column`).

**TSV Example:**
```bash
tetss2 predict-file --input input.tsv --output tetss2_predictions.tsv
```

**CSV Example (with custom column name):**
```bash
tetss2 predict-file --input input.csv --output tetss2_predictions.csv --sep "," --sequence-column dna
```

The output file will seamlessly append the original data with new columns: `tetss2_sequence_length`, `tetss2_probability`, `tetss2_prediction`, and `tetss2_threshold`.

---

## Python API Usage

TETSS2.0 can be seamlessly integrated into your custom bioinformatics pipelines or Jupyter Notebooks:

```python
from tetss2 import TETSS2Predictor

# Initialize the predictor (automatically loads the optimized best_model.pth weight file)
predictor = TETSS2Predictor()

# Execute prediction on a valid 201 bp sequence context
seq = "ATGCTCTGTGAGGCCGTGCTTGGCCGAGTCAGGAAGGGCCCCACGCGTGCCGTAGGCCGTAGGTAGGCGGGTAGGCGAGTGGCGGAGGCCCAGGGCCTGCCAGCTGGGTAGGCGAGGGCGGCGGGAGGCCTGGGGCCTGCGCGTGCGGCGGAGGCCTGGAGCCTGCCAGCTGGGCGGGCGAGGGCGGCGGGAGGCCTGGGG"
result = predictor.predict(seq)

print(result)
```

---

## Troubleshooting

* **MKL Threading Error:** If the program crashes on Linux server clusters with an Intel MKL layer conflict message, run `export MKL_THREADING_LAYER=GNU` before executing prediction commands.
* **Parser Column Errors:** If the CLI returns a `Cannot find sequence column` exception, please run `cat -A input.tsv` to verify whether your delimiters are true hardware horizontal tabs (`^I`) or hidden whitespace character blocks.

---

## Citation and contact

Manuscript citation and permanent archive identifiers will be added when
available. For questions about the code, please open an issue in this repository.

