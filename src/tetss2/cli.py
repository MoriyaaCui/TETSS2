import os

os.environ["MKL_THREADING_LAYER"] = "GNU"

import argparse
import json

from .predictor import TETSS2Predictor


def main():
    parser = argparse.ArgumentParser(
        description="TETSS2.0: predict TE-TSS activity from 201 bp DNA sequence."
    )

    subparsers = parser.add_subparsers(dest="command")

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict one DNA sequence."
    )
    predict_parser.add_argument(
        "--sequence",
        required=True,
        help="Input DNA sequence. TETSS2.0 expects exactly 201 bp."
    )
    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold. Default: 0.5"
    )
    predict_parser.add_argument(
        "--no-length-check",
        action="store_true",
        help="Disable 201 bp length check. Mainly for debugging."
    )

    file_parser = subparsers.add_parser(
        "predict-file",
        help="Predict sequences from a TSV or CSV file."
    )
    file_parser.add_argument(
        "--input",
        required=True,
        help="Input file path. The file should contain a sequence column."
    )
    file_parser.add_argument(
        "--output",
        required=True,
        help="Output file path."
    )
    file_parser.add_argument(
        "--sequence-column",
        default="sequence",
        help="Column name containing DNA sequences. Default: sequence"
    )
    file_parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold. Default: 0.5"
    )
    file_parser.add_argument(
        "--sep",
        default="\t",
        help="File separator. Default: tab. Use ',' for CSV."
    )
    file_parser.add_argument(
        "--no-length-check",
        action="store_true",
        help="Disable 201 bp length check. Mainly for debugging."
    )

    args = parser.parse_args()

    if args.command == "predict":
        predictor = TETSS2Predictor()
        result = predictor.predict(
            args.sequence,
            threshold=args.threshold,
            check_length=not args.no_length_check,
        )
        print(json.dumps(result, indent=2))

    elif args.command == "predict-file":
        sep = args.sep
        if sep == "\\t":
            sep = "\t"

        predictor = TETSS2Predictor()
        predictor.predict_file(
            input_path=args.input,
            output_path=args.output,
            sequence_column=args.sequence_column,
            threshold=args.threshold,
            sep=sep,
            check_length=not args.no_length_check,
        )

        print(f"Prediction finished. Results saved to: {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()