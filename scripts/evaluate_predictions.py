#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def case_id_from_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "")
    return path.stem


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    intersection = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate binary segmentation predictions against ground truth masks."
    )
    parser.add_argument("--pred-dir", required=True, help="Directory with prediction .nii.gz files.")
    parser.add_argument("--gt-dir", required=True, help="Directory with ground-truth .nii.gz files.")
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--out-json", required=True, help="Output summary JSON path.")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    out_csv = Path(args.out_csv)
    out_json = Path(args.out_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pred_files = sorted(
        pred_dir.glob("*.nii.gz"),
        key=lambda p: int(case_id_from_name(p)) if case_id_from_name(p).isdigit() else case_id_from_name(p),
    )

    if not pred_files:
        raise RuntimeError(f"No .nii.gz prediction files found in {pred_dir}")

    rows = []
    dices = []

    for pred_path in pred_files:
        case_id = case_id_from_name(pred_path)
        gt_path = gt_dir / f"{case_id}.nii.gz"

        if not gt_path.exists():
            print(f"WARNING: missing GT for case {case_id}: {gt_path}")
            continue

        gt_img = sitk.ReadImage(str(gt_path))
        pred_img = sitk.ReadImage(str(pred_path))

        gt = sitk.GetArrayFromImage(gt_img) > 0
        pred = sitk.GetArrayFromImage(pred_img) > 0

        if gt.shape != pred.shape:
            raise RuntimeError(
                f"Shape mismatch for case {case_id}: GT {gt.shape}, pred {pred.shape}"
            )

        tp = int(np.logical_and(gt, pred).sum())
        fp = int(np.logical_and(~gt, pred).sum())
        fn = int(np.logical_and(gt, ~pred).sum())
        gt_voxels = int(gt.sum())
        pred_voxels = int(pred.sum())
        dice = dice_score(gt, pred)

        dices.append(dice)
        rows.append({
            "case_id": case_id,
            "dice": dice,
            "gt_voxels": gt_voxels,
            "pred_voxels": pred_voxels,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        })

        print(
            f"{case_id}: Dice={dice:.4f}, GT={gt_voxels}, pred={pred_voxels}, "
            f"FP={fp}, FN={fn}"
        )

    if not rows:
        raise RuntimeError("No valid cases were evaluated.")

    dices_np = np.array(dices, dtype=float)

    summary = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n": int(len(dices)),
        "mean_dice": float(np.mean(dices_np)),
        "std_dice": float(np.std(dices_np)),
        "min_dice": float(np.min(dices_np)),
        "max_dice": float(np.max(dices_np)),
        "best_case": rows[int(np.argmax(dices_np))]["case_id"],
        "worst_case": rows[int(np.argmin(dices_np))]["case_id"],
    }

    with out_csv.open("w", encoding="utf-8") as f:
        f.write("case_id,dice,gt_voxels,pred_voxels,tp,fp,fn\n")
        for r in rows:
            f.write(
                f"{r['case_id']},{r['dice']:.8f},{r['gt_voxels']},"
                f"{r['pred_voxels']},{r['tp']},{r['fp']},{r['fn']}\n"
            )

    out_json.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print()
    print("Summary:")
    print(f"Mean Dice: {summary['mean_dice']}")
    print(f"Std Dice: {summary['std_dice']}")
    print(f"Min Dice: {summary['min_dice']}  case {summary['worst_case']}")
    print(f"Max Dice: {summary['max_dice']}  case {summary['best_case']}")
    print(f"N: {summary['n']}")
    print()
    print(f"Saved CSV: {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
