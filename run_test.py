import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Test U-Mamba model.")
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    print("=== RUN TEST ===")
    print(f"Data path: {data_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Data path ne obstaja: {data_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print("OK: run_test.py se pravilno zažene.")


if __name__ == "__main__":
    main()