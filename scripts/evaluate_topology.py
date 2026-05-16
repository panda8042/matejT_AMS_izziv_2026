#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

try:
    from scipy import ndimage as ndi
except Exception as e:
    raise RuntimeError("This script requires scipy.ndimage.") from e


def case_id_from_name(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name.replace(".nii.gz", "")
    return path.stem


def dice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    inter = np.logical_and(gt, pred).sum()
    denom = gt.sum() + pred.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * inter / denom)


def connected_components_count(mask: np.ndarray) -> int:
    if mask.sum() == 0:
        return 0
    structure = np.ones((3, 3, 3), dtype=np.uint8)
    _, n = ndi.label(mask.astype(bool), structure=structure)
    return int(n)


def skeleton_proxy(mask: np.ndarray) -> np.ndarray:
    """
    Approximate 3D centerline/skeleton.

    First tries skimage skeletonization if available.
    If not available, uses a distance-transform local-maximum proxy.
    This is not a perfect vascular skeleton, but is useful as a reproducible
    topology-like clDice approximation.
    """
    mask = mask.astype(bool)
    if mask.sum() == 0:
        return mask

    try:
        try:
            from skimage.morphology import skeletonize_3d
            return skeletonize_3d(mask).astype(bool)
        except Exception:
            from skimage.morphology import skeletonize
            return skeletonize(mask).astype(bool)
    except Exception:
        dist = ndi.distance_transform_edt(mask)
        local_max = dist == ndi.maximum_filter(dist, size=3)
        skel = np.logical_and(local_max, mask)
        return skel.astype(bool)


def cldice_score(gt: np.ndarray, pred: np.ndarray) -> float:
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    if gt.sum() == 0 and pred.sum() == 0:
        return 1.0
    if gt.sum() == 0 or pred.sum() == 0:
        return 0.0

    s_gt = skeleton_proxy(gt)
    s_pred = skeleton_proxy(pred)

    if s_gt.sum() == 0 or s_pred.sum() == 0:
        return 0.0

    tprec = np.logical_and(s_pred, gt).sum() / s_pred.sum()
    tsens = np.logical_and(s_gt, pred).sum() / s_gt.sum()

    if tprec + tsens == 0:
        return 0.0

    return float(2.0 * tprec * tsens / (tprec + tsens))


def variation_of_information(gt: np.ndarray, pred: np.ndarray) -> float:
    """
    VOI for binary segmentations.
    Lower is better. 0 means identical labelings.
    """
    gt = gt.astype(np.uint8).ravel()
    pred = pred.astype(np.uint8).ravel()

    n = gt.size
    if n == 0:
        return 0.0

    contingency = np.zeros((2, 2), dtype=np.float64)
    for g, p in zip(gt, pred):
        contingency[g, p] += 1.0

    contingency /= n
    p_gt = contingency.sum(axis=1)
    p_pred = contingency.sum(axis=0)

    def entropy(p):
        p = p[p > 0]
        return -float(np.sum(p * np.log2(p)))

    h_gt = entropy(p_gt)
    h_pred = entropy(p_pred)

    mi = 0.0
    for i in range(2):
        for j in range(2):
            pij = contingency[i, j]
            if pij > 0 and p_gt[i] > 0 and p_pred[j] > 0:
                mi += pij * np.log2(pij / (p_gt[i] * p_pred[j]))

    voi = h_gt + h_pred - 2.0 * mi
    return float(voi)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation predictions with overlap and topology-like metrics."
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

    for pred_path in pred_files:
        case_id = case_id_from_name(pred_path)
        gt_path = gt_dir / f"{case_id}.nii.gz"

        if not gt_path.exists():
            print(f"WARNING: missing GT for case {case_id}: {gt_path}")
            continue

        gt = sitk.GetArrayFromImage(sitk.ReadImage(str(gt_path))) > 0
        pred = sitk.GetArrayFromImage(sitk.ReadImage(str(pred_path))) > 0

        if gt.shape != pred.shape:
            raise RuntimeError(f"Shape mismatch for case {case_id}: GT {gt.shape}, pred {pred.shape}")

        tp = int(np.logical_and(gt, pred).sum())
        fp = int(np.logical_and(~gt, pred).sum())
        fn = int(np.logical_and(gt, ~pred).sum())

        gt_components = connected_components_count(gt)
        pred_components = connected_components_count(pred)
        betti0_error = abs(pred_components - gt_components)

        row = {
            "case_id": case_id,
            "dice": dice_score(gt, pred),
            "cldice": cldice_score(gt, pred),
            "betti0_error": float(betti0_error),
            "voi": variation_of_information(gt, pred),
            "gt_voxels": int(gt.sum()),
            "pred_voxels": int(pred.sum()),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_components": gt_components,
            "pred_components": pred_components,
        }
        rows.append(row)

        print(
            f"{case_id}: Dice={row['dice']:.4f}, clDice={row['cldice']:.4f}, "
            f"Betti0Err={row['betti0_error']:.0f}, VOI={row['voi']:.6f}, "
            f"GTcomp={gt_components}, Predcomp={pred_components}, FP={fp}, FN={fn}"
        )

    if not rows:
        raise RuntimeError("No valid cases were evaluated.")

    keys = [
        "case_id", "dice", "cldice", "betti0_error", "voi",
        "gt_voxels", "pred_voxels", "tp", "fp", "fn",
        "gt_components", "pred_components"
    ]

    with out_csv.open("w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in keys) + "\n")

    def values(key):
        return np.array([float(r[key]) for r in rows], dtype=float)

    summary = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "n": len(rows),
        "mean_dice": float(values("dice").mean()),
        "std_dice": float(values("dice").std()),
        "mean_cldice": float(values("cldice").mean()),
        "std_cldice": float(values("cldice").std()),
        "mean_betti0_error": float(values("betti0_error").mean()),
        "std_betti0_error": float(values("betti0_error").std()),
        "mean_voi": float(values("voi").mean()),
        "std_voi": float(values("voi").std()),
        "min_dice": float(values("dice").min()),
        "max_dice": float(values("dice").max()),
        "best_dice_case": rows[int(values("dice").argmax())]["case_id"],
        "worst_dice_case": rows[int(values("dice").argmin())]["case_id"],
    }

    out_json.write_text(json.dumps(summary, indent=4), encoding="utf-8")

    print()
    print("Summary:")
    print(f"Mean Dice:         {summary['mean_dice']:.4f} ± {summary['std_dice']:.4f}")
    print(f"Mean clDice:       {summary['mean_cldice']:.4f} ± {summary['std_cldice']:.4f}")
    print(f"Mean Betti0 error: {summary['mean_betti0_error']:.4f} ± {summary['std_betti0_error']:.4f}")
    print(f"Mean VOI:          {summary['mean_voi']:.6f} ± {summary['std_voi']:.6f}")
    print(f"Best Dice case:    {summary['best_dice_case']}")
    print(f"Worst Dice case:   {summary['worst_dice_case']}")
    print()
    print(f"Saved CSV:  {out_csv}")
    print(f"Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
