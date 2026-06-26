import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import torch
import numpy as np
import pandas as pd
from importlib.resources import files

from .model import TETSSClassifier


EXPECTED_SEQUENCE_LENGTH = 201


def clean_sequence(seq: str) -> str:
    seq = str(seq).strip().upper()
    return seq


def validate_sequence(seq: str, expected_length: int = EXPECTED_SEQUENCE_LENGTH) -> None:
    allowed_bases = set("ACGTN")

    invalid_bases = sorted(set(seq) - allowed_bases)
    if invalid_bases:
        raise ValueError(
            f"Invalid bases found: {invalid_bases}. "
            "Only A, C, G, T, and N are allowed."
        )

    if len(seq) != expected_length:
        raise ValueError(
            f"Invalid sequence length: {len(seq)} bp. "
            f"TETSS2.0 expects exactly {expected_length} bp."
        )


def seq_to_onehot(seq: str) -> np.ndarray:
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    arr = np.zeros((4, len(seq)), dtype=np.float32)

    for i, base in enumerate(seq.upper()):
        idx = mapping.get(base)
        if idx is not None:
            arr[idx, i] = 1.0

    return arr


class TETSS2Predictor:
    def __init__(self, model_path=None, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_path is None:
            model_path = files("tetss2").joinpath("assets/best_model.pth")

        self.model = TETSSClassifier().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def predict_proba(self, sequence: str, check_length: bool = True) -> float:
        sequence = clean_sequence(sequence)

        if check_length:
            validate_sequence(sequence)

        x = seq_to_onehot(sequence)
        x = torch.from_numpy(x).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            prob = torch.sigmoid(logits).item()

        return float(prob)

    def predict(
        self,
        sequence: str,
        threshold: float = 0.5,
        check_length: bool = True,
    ) -> dict:
        sequence = clean_sequence(sequence)

        prob = self.predict_proba(sequence, check_length=check_length)
        label = int(prob > threshold)

        return {
            "model": "TETSS2.0",
            "sequence": sequence,
            "sequence_length": len(sequence),
            "probability": prob,
            "prediction": label,
            "threshold": threshold,
        }

    def predict_file(
        self,
        input_path: str,
        output_path: str,
        sequence_column: str = "sequence",
        threshold: float = 0.5,
        sep: str = "\t",
        check_length: bool = True,
    ) -> None:
        df = pd.read_csv(input_path, sep=sep)

        if sequence_column not in df.columns:
            raise ValueError(
                f"Cannot find sequence column '{sequence_column}' in input file. "
                f"Available columns are: {list(df.columns)}"
            )

        probabilities = []
        predictions = []
        sequence_lengths = []

        for idx, seq in enumerate(df[sequence_column], start=1):
            seq = clean_sequence(seq)

            try:
                prob = self.predict_proba(seq, check_length=check_length)
            except ValueError as e:
                raise ValueError(
                    f"Error in row {idx}: {e}"
                )

            pred = int(prob > threshold)

            probabilities.append(prob)
            predictions.append(pred)
            sequence_lengths.append(len(seq))

        df["tetss2_sequence_length"] = sequence_lengths
        df["tetss2_probability"] = probabilities
        df["tetss2_prediction"] = predictions
        df["tetss2_threshold"] = threshold

        df.to_csv(output_path, sep=sep, index=False)