import argparse
import json
from pathlib import Path


def make_symlink(src: Path, dst: Path, overwrite: bool = False):
    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    dst.symlink_to(src.resolve())


def main():
    parser = argparse.ArgumentParser(
        description="Convert ImageCAS dataset to nnU-Net raw format using symbolic links."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to original ImageCAS folder containing 1-200, 201-400, ... folders.",
    )
    parser.add_argument(
        "--output_path",
        required=True,
        help="Output path for nnU-Net raw dataset, e.g. nnUNet_raw/Dataset501_ImageCAS.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fraction of cases used as test set if no official split is used.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing symlinks.",
    )

    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    images_tr = output_path / "imagesTr"
    labels_tr = output_path / "labelsTr"
    images_ts = output_path / "imagesTs"

    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)
    images_ts.mkdir(parents=True, exist_ok=True)

    img_files = sorted(input_path.glob("*/*.img.nii.gz"), key=lambda p: int(p.name.split(".")[0]))
    label_files = sorted(input_path.glob("*/*.label.nii.gz"), key=lambda p: int(p.name.split(".")[0]))

    img_ids = {p.name.replace(".img.nii.gz", "") for p in img_files}
    label_ids = {p.name.replace(".label.nii.gz", "") for p in label_files}

    missing_labels = sorted(img_ids - label_ids, key=int)
    missing_images = sorted(label_ids - img_ids, key=int)

    if missing_labels:
        raise RuntimeError(f"Images without labels: {missing_labels[:20]}")

    if missing_images:
        raise RuntimeError(f"Labels without images: {missing_images[:20]}")

    case_ids = sorted(img_ids & label_ids, key=int)

    n_total = len(case_ids)
    n_test = int(round(n_total * args.test_fraction))
    n_train = n_total - n_test

    train_ids = case_ids[:n_train]
    test_ids = case_ids[n_train:]

    print(f"Total cases: {n_total}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Test cases: {len(test_ids)}")
    print(f"Output path: {output_path}")

    for old_id in train_ids:
        src_img = input_path / f"{1 + ((int(old_id)-1)//200)*200}-{((int(old_id)-1)//200 + 1)*200}" / f"{old_id}.img.nii.gz"
        src_lbl = input_path / f"{1 + ((int(old_id)-1)//200)*200}-{((int(old_id)-1)//200 + 1)*200}" / f"{old_id}.label.nii.gz"

        new_case = f"imagecas_{int(old_id):04d}"

        dst_img = images_tr / f"{new_case}_0000.nii.gz"
        dst_lbl = labels_tr / f"{new_case}.nii.gz"

        make_symlink(src_img, dst_img, overwrite=args.overwrite)
        make_symlink(src_lbl, dst_lbl, overwrite=args.overwrite)

    for old_id in test_ids:
        src_img = input_path / f"{1 + ((int(old_id)-1)//200)*200}-{((int(old_id)-1)//200 + 1)*200}" / f"{old_id}.img.nii.gz"

        new_case = f"imagecas_{int(old_id):04d}"
        dst_img = images_ts / f"{new_case}_0000.nii.gz"

        make_symlink(src_img, dst_img, overwrite=args.overwrite)

    dataset_json = {
        "channel_names": {
            "0": "CTA"
        },
        "labels": {
            "background": 0,
            "coronary_artery": 1
        },
        "numTraining": len(train_ids),
        "file_ending": ".nii.gz",
        "name": "ImageCAS",
        "description": "ImageCAS coronary artery segmentation dataset converted to nnU-Net format."
    }

    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("Done.")
    print(f"Created nnU-Net dataset at: {output_path}")


if __name__ == "__main__":
    main()