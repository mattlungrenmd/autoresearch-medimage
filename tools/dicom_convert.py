#!/usr/bin/env python3
"""DICOM to PNG converter with proper windowing and modality detection.

Converts DICOM files to PNG images ready for training, handling:
- Window center/width from DICOM tags
- Auto-detection of imaging modality (CT, XR, MR, etc.)
- Batch conversion of entire directories

Usage:
    python tools/dicom_convert.py input.dcm output.png
    python tools/dicom_convert.py dicom_dir/ png_dir/ --recursive
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def apply_windowing(pixel_array, window_center, window_width):
    """Apply DICOM windowing to pixel data."""
    lower = window_center - window_width / 2
    upper = window_center + window_width / 2
    img = np.clip(pixel_array, lower, upper)
    img = ((img - lower) / (upper - lower) * 255).astype(np.uint8)
    return img


def convert_dicom_to_png(dicom_path, output_path=None, window=None):
    """Convert a single DICOM file to PNG.

    Args:
        dicom_path: Path to DICOM file
        output_path: Output PNG path (default: same name with .png)
        window: Optional (center, width) tuple to override DICOM tags
    """
    try:
        import pydicom
    except ImportError:
        print("pydicom is required: pip install pydicom")
        sys.exit(1)

    ds = pydicom.dcmread(dicom_path)
    pixel_array = ds.pixel_array.astype(np.float64)

    # Apply rescale slope/intercept if present
    slope = getattr(ds, "RescaleSlope", 1)
    intercept = getattr(ds, "RescaleIntercept", 0)
    pixel_array = pixel_array * slope + intercept

    # Detect modality
    modality = getattr(ds, "Modality", "unknown")

    # Apply windowing
    if window:
        wc, ww = window
    elif hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        wc = ds.WindowCenter
        ww = ds.WindowWidth
        if isinstance(wc, (list, pydicom.multival.MultiValue)):
            wc = wc[0]
        if isinstance(ww, (list, pydicom.multival.MultiValue)):
            ww = ww[0]
        wc, ww = float(wc), float(ww)
    else:
        # Auto-window based on data range
        wc = (pixel_array.max() + pixel_array.min()) / 2
        ww = pixel_array.max() - pixel_array.min()

    img_array = apply_windowing(pixel_array, wc, ww)

    # Handle photometric interpretation
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        img_array = 255 - img_array

    img = Image.fromarray(img_array)

    if output_path is None:
        output_path = str(Path(dicom_path).with_suffix(".png"))

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    img.save(output_path)

    return {
        "modality": modality,
        "patient_id": getattr(ds, "PatientID", "unknown"),
        "study_desc": getattr(ds, "StudyDescription", ""),
        "rows": ds.Rows,
        "cols": ds.Columns,
        "window": (wc, ww),
    }


def batch_convert(input_dir, output_dir, recursive=False):
    """Convert all DICOM files in a directory to PNG."""
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    pattern = "**/*.dcm" if recursive else "*.dcm"
    dicom_files = list(input_path.glob(pattern))

    # Also check files without extension (common for DICOM)
    if not dicom_files:
        all_files = list(input_path.glob("**/*" if recursive else "*"))
        dicom_files = [f for f in all_files if f.is_file() and not f.suffix]

    print(f"Found {len(dicom_files)} files to convert")

    modalities = {}
    for i, dcm_file in enumerate(dicom_files):
        rel_path = dcm_file.relative_to(input_path)
        out_file = output_path / rel_path.with_suffix(".png")

        try:
            info = convert_dicom_to_png(str(dcm_file), str(out_file))
            mod = info["modality"]
            modalities[mod] = modalities.get(mod, 0) + 1

            if (i + 1) % 100 == 0 or i == len(dicom_files) - 1:
                print(f"  [{i+1}/{len(dicom_files)}] {dcm_file.name} -> {out_file.name} ({mod})")
        except Exception as e:
            print(f"  ERROR: {dcm_file.name}: {e}")

    print(f"\nDone! Converted {sum(modalities.values())} files")
    print(f"Modalities found: {modalities}")


def main():
    parser = argparse.ArgumentParser(description="DICOM to PNG converter")
    parser.add_argument("input", help="Input DICOM file or directory")
    parser.add_argument("output", nargs="?", help="Output PNG file or directory")
    parser.add_argument("--recursive", "-r", action="store_true",
                        help="Recursively process directories")
    parser.add_argument("--window", nargs=2, type=float, metavar=("CENTER", "WIDTH"),
                        help="Override window center and width")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_dir():
        output_dir = args.output or str(input_path) + "_png"
        batch_convert(str(input_path), output_dir, recursive=args.recursive)
    else:
        window = tuple(args.window) if args.window else None
        info = convert_dicom_to_png(str(input_path), args.output, window=window)
        print(f"Converted: {args.input}")
        print(f"  Modality: {info['modality']}")
        print(f"  Size: {info['rows']}x{info['cols']}")
        print(f"  Window: center={info['window'][0]}, width={info['window'][1]}")


if __name__ == "__main__":
    main()
