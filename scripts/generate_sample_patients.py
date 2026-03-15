"""
Generate synthetic CT-ICH demo patients as NIfTI (.nii) files.

Named exactly like CT-ICH dataset (PhysioNet): 049.nii – 055.nii
Volume: 128×128×24 slices  (~1.5 MB per file, ~10 MB total)
Output: data/samples/
"""

from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

try:
    import nibabel as nib
except ImportError:
    print("nibabel not installed — run: pip install nibabel")
    sys.exit(1)

from src.data.mock_data import generate_mock_volume

OUT_DIR = ROOT / "data" / "samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Patient 049–055, each with a different hemorrhage subtype
PATIENTS = [
    ("049", "epidural"),
    ("050", "subdural"),
    ("051", "intraparenchymal"),
    ("052", "intraventricular"),
    ("053", "subarachnoid"),
    ("054", "any"),
    ("055", "normal"),
]

DEPTH, SIZE = 24, 128

print(f"Generating {len(PATIENTS)} CT-ICH demo patients -> {OUT_DIR}\n")

for pid, subtype in PATIENTS:
    fpath = OUT_DIR / f"{pid}.nii"
    volume = generate_mock_volume(subtype=subtype, depth=DEPTH, size=SIZE, seed=int(pid))

    # nibabel expects (H, W, D) — transpose from (D, H, W)
    affine = np.diag([1.0, 1.0, 5.0, 1.0])
    img = nib.Nifti1Image(volume.transpose(1, 2, 0).astype(np.float32), affine)
    img.header.set_data_dtype(np.float32)
    nib.save(img, str(fpath))

    kb = fpath.stat().st_size / 1024
    print(f"  OK {pid}.nii  ({kb:.0f} KB)  [{subtype}]")

print(f"\nDone -- {len(PATIENTS)} files in {OUT_DIR}")
