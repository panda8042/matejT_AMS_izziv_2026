import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


def make_symlink(src: Path, dst: Path, overwrite: bool = False):
    if not src.exists():
        raise FileNotFoundError(f"Source file does not exist: {src}")

    if dst.exists() or dst.is_symlink():
        if overwrite:
            dst.unlink()
        else:
            return

    dst.symlink_to(src.resolve())


def case_folder(case_id: str) -> str:
    case_num = int(case_id)
    start = 1 + ((case_num - 1) // 200) * 200
    end = ((case_num - 1) // 200 + 1) * 200
    return f"{start}-{end}"


def col_to_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return idx - 1


def read_xlsx_rows(xlsx_path: Path, sheet_name: str):
    ns = {
        "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
        "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    with ZipFile(xlsx_path) as z:
        shared_strings = []
        if "xl/sharedStrings.xml" in z.namelist():
            root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in root.findall("main:si", ns):
                texts = []
                for t in si.findall(".//main:t", ns):
                    texts.append(t.text or "")
                shared_strings.append("".join(texts))

        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))

        rel_map = {}
        for rel in rels:
            rel_map[rel.attrib["Id"]] = rel.attrib["Target"]

        sheet_path = None
        for sheet in workbook.findall(".//main:sheet", ns):
            name = sheet.attrib["name"]
            rid = sheet.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            target = rel_map[rid]

            if name == sheet_name:
                sheet_path = "xl/" + target.lstrip("/")
                break

        if sheet_path is None:
            available = [s.attrib["name"] for s in workbook.findall(".//main:sheet", ns)]
            raise ValueError(f"Sheet '{sheet_name}' not found. Available sheets: {available}")

        root = ET.fromstring(z.read(sheet_path))
        rows = root.findall(".//main:sheetData/main:row", ns)

        parsed_rows = []
        for row in rows:
            values = []
            for c in row.findall("main:c", ns):
                ref = c.attrib.get("r", "")
                col_idx = col_to_index(ref) if ref else len(values)

                while len(values) < col_idx:
                    values.append(None)

                v = c.find("main:v", ns)
                if v is None:
                    value = None
                else:
                    raw = v.text
                    if c.attrib.get("t") == "s":
                        value = shared_strings[int(raw)]
                    else:
                        value = raw

                values.append(value)

            parsed_rows.append(values)

        return parsed_rows


def load_split_from_xlsx(split_file: Path, split_name: str, sheet_name: str = "v2-latest"):
    rows = read_xlsx_rows(split_file, sheet_name=sheet_name)

    header_row = None
    for row in rows:
        if row and "FileName" in row:
            header_row = row
            break

    if header_row is None:
        raise RuntimeError("Could not find header row with 'FileName'.")

    filename_col = header_row.index("FileName")

    if split_name not in header_row:
        raise RuntimeError(f"Could not find split column '{split_name}'. Header: {header_row}")

    split_col = header_row.index(split_name)

    train_ids = []
    test_ids = []

    header_found = False
    for row in rows:
        if row == header_row:
            header_found = True
            continue

        if not header_found:
            continue

        if len(row) <= max(filename_col, split_col):
            continue

        case_id = row[filename_col]
        split_value = row[split_col]

        if case_id is None or split_value is None:
            continue

        case_id = str(case_id).strip()
        split_value = str(split_value).strip().lower()

        if split_value in ["training", "val", "validation"]:
            train_ids.append(case_id)
        elif split_value == "testing":
            test_ids.append(case_id)

    return sorted(train_ids, key=int), sorted(test_ids, key=int)


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
        "--split_file",
        default=None,
        help="Path to imageCAS_data_split.xlsx. If provided, official split is used.",
    )
    parser.add_argument(
        "--split",
        default="Split-1",
        choices=["Split-1", "Split-2", "Split-3", "Split-4"],
        help="Official split column to use.",
    )
    parser.add_argument(
        "--sheet_name",
        default="v2-latest",
        help="Excel sheet name containing the split definition.",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.2,
        help="Fallback test fraction if no split_file is provided.",
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

    if args.split_file is not None:
        train_ids, test_ids = load_split_from_xlsx(
            Path(args.split_file),
            split_name=args.split,
            sheet_name=args.sheet_name,
        )

        train_ids = [case_id for case_id in train_ids if case_id in case_ids]
        test_ids = [case_id for case_id in test_ids if case_id in case_ids]
    else:
        n_total = len(case_ids)
        n_test = int(round(n_total * args.test_fraction))
        n_train = n_total - n_test

        train_ids = case_ids[:n_train]
        test_ids = case_ids[n_train:]

    # Mini debug subset: use only a few cases for fast pipeline testing
    train_ids = train_ids[:10]
    test_ids = test_ids[:2]

    print(f"Total available cases: {len(case_ids)}")
    print(f"Train cases: {len(train_ids)}")
    print(f"Test cases: {len(test_ids)}")
    print(f"Split: {args.split}")
    print(f"Output path: {output_path}")

    for old_id in train_ids:
        src_img = input_path / case_folder(old_id) / f"{old_id}.img.nii.gz"
        src_lbl = input_path / case_folder(old_id) / f"{old_id}.label.nii.gz"

        new_case = f"imagecas_{int(old_id):04d}"

        dst_img = images_tr / f"{new_case}_0000.nii.gz"
        dst_lbl = labels_tr / f"{new_case}.nii.gz"

        make_symlink(src_img, dst_img, overwrite=args.overwrite)
        make_symlink(src_lbl, dst_lbl, overwrite=args.overwrite)

    for old_id in test_ids:
        src_img = input_path / case_folder(old_id) / f"{old_id}.img.nii.gz"

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
        "name": "ImageCASMini",
        "description": f"ImageCAS coronary artery segmentation dataset converted to nnU-Net format using {args.split}."
    }

    with open(output_path / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    print("Done.")
    print(f"Created nnU-Net dataset at: {output_path}")


if __name__ == "__main__":
    main()