import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Train U-Mamba model.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--params_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    params_path = Path(args.params_path)
    output_path = Path(args.output_path)

    print("=== RUN TRAIN ===")
    print(f"Data path: {data_path}")
    print(f"Params path: {params_path}")
    print(f"Output path: {output_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data path ne obstaja: {data_path}")

    if not params_path.exists():
        raise FileNotFoundError(f"Params file ne obstaja: {params_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print("OK: run_train.py se pravilno zažene.")


if __name__ == "__main__":
    main()