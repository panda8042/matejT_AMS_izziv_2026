import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Run inference with U-Mamba model.")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)

    print("=== RUN INFERENCE ===")
    print(f"Input path: {input_path}")
    print(f"Model path: {model_path}")
    print(f"Output path: {output_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Input path ne obstaja: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    print("OK: run_inference.py se pravilno zažene.")


if __name__ == "__main__":
    main()