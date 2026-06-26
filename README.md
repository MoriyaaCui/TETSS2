# TETSS2.0

TETSS2.0 is a deep learning model for predicting TE-derived transcription start site (TE-TSS) activity from DNA sequences. 

The Python package and command-line tool are named `tetss2`. The model name used in documents, figures, and the upcoming web server is **TETSS2.0**.

## Overview

TETSS2.0 is a PyTorch-based 1D-Convolutional Neural Network classifier designed to predict whether an input DNA sequence is associated with TE-TSS activity. 

The model takes a strictly 201 bp DNA sequence as input and returns:
* A prediction probability score
* A binary prediction label (0 or 1)
* The classification threshold used (default: 0.5)

---

## Installation (Recommended)

To avoid dependency conflicts and bypass massive GPU driver downloads for standard prediction tasks, we highly recommend installing `tetss2` in a clean Conda environment with a CPU-only version of PyTorch.

**1. Create a clean environment (Python 3.10+ recommended):**
```bash
conda create -n tetss2_env python=3.10 -y
conda activate tetss2_env
```

**2. Install a lightweight, CPU-only PyTorch:**
```bash
pip install torch torchvision --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu)
```

**3. Install TETSS2 directly from GitHub:**
```bash
pip install git+[https://github.com/MoriyaaCui/TETSS2.git](https://github.com/MoriyaaCui/TETSS2.git)
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
**Example Output:**
```json
{
  "model": "TETSS2.0",
  "sequence": "ATGCTCTGTGAGGCCGTGCTT...",
  "sequence_length": 201,
  "probability": 0.0044,
  "prediction": 0,
  "threshold": 0.5
}
```

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

## Citation & Contact

If you use TETSS2 in your research, please cite:
```bibtex
@software{tetss2, 
  title={TETSS2: Deep Learning Model for TE-TSS Prediction}, 
  year={2026} 
}
```
For questions or issues, please open an issue in the project repository.