import argparse
from pathlib import Path

import pandas as pd


DEFAULT_DATASET_ID = "dhrubangtalukdar/e-commerce-shopper-behavior-amazonshopify-based"
DEFAULT_DATASET_FILE = "e_commerce_shopper_behaviour_and_lifestyle.csv"


def resolve_input_path(input_path: str | None) -> Path:
    if input_path:
        path = Path(input_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        return path

    # Fallback: download dataset the same way as preprocessing notebook.
    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "No input path provided and kagglehub is not installed. "
            "Install kagglehub or pass --input /path/to/e_commerce_data.csv"
        ) from exc

    dataset_dir = Path(kagglehub.dataset_download(DEFAULT_DATASET_ID))
    dataset_file = dataset_dir / DEFAULT_DATASET_FILE
    if not dataset_file.exists():
        raise FileNotFoundError(
            f"Dataset downloaded to {dataset_dir}, but {DEFAULT_DATASET_FILE} was not found"
        )
    return dataset_file.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a small sample_input.csv from the e-commerce dataset"
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to source e-commerce CSV. If omitted, script tries kagglehub download.",
    )
    parser.add_argument(
        "--output",
        default="deployment_module/sample_input.csv",
        help="Where to write the sampled CSV",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1000,
        help="Number of rows to keep",
    )
    args = parser.parse_args()

    if args.rows <= 0:
        raise ValueError("--rows must be greater than 0")

    source_path = resolve_input_path(args.input)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(source_path)
    sampled_df = df.head(args.rows)
    sampled_df.to_csv(output_path, index=False)
    print(df.shape)
    print(f"Source file: {source_path}")
    print(f"Rows in source: {len(df)}")
    print(f"Rows written: {len(sampled_df)}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()