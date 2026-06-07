# Copyright (c) Meta Platforms, Inc. and affiliates.
# (HOI4D converter contributed externally — not an official Meta dataset)
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

"""Convert HOI4D RGB sequences to WAI format.

This converter handles the RGB-only subset of HOI4D (align_rgb/image.mp4
decoded to JPEG frames). Depth, hand pose, object pose, and segmentation
annotations are not yet included and will be added in future iterations.

HOI4D source layout (after decode_rgb.py):
    <original_root>/
    └── ZY2021080000*/H*/C*/N*/S*/s*/T*/
        └── align_rgb/
            ├── image.mp4
            ├── 00000.jpg
            ├── 00001.jpg
            └── ...

WAI output layout:
    <root>/
    └── ZY2021080000*_H*_C*_N*_S*_s*_T*/
        ├── scene_meta.json
        └── images/
            ├── 00000.jpg -> (symlink to source)
            └── ...

Scene naming convention:
    Original path components joined with "_":
    ZY20210800004/H4/C8/N14/S71/s03/T2  ->  ZY20210800004_H4_C8_N14_S71_s03_T2

Global control:
    FG_MAX_SCENES env var (set by scripts/run.py --max-scenes N) limits the
    number of scenes processed. Takes precedence over cfg.max_scenes.

Usage:
    # Via functional_grasp runner (recommended):
    python scripts/run.py --step hoi4d_convert --max-scenes 5

    # Direct:
    python data_processing/wai_processing/scripts/conversion/hoi4d.py
"""

import os
from pathlib import Path

from argconf import argconf_parse
from natsort import natsorted
from wai_processing.utils.globals import WAI_PROC_CONFIG_PATH
from wai_processing.utils.wrapper import convert_scenes_wrapper

from mapanything.utils.wai.core import store_data
from mapanything.utils.wai.scene_frame import get_scene_names

# HOI4D path is always 7 levels deep: ZY* / H* / C* / N* / S* / s* / T*
_HOI4D_PATH_DEPTH = 7


def scene_name_to_rel_path(scene_name: str) -> str:
    """Convert a flat WAI scene name back to the original HOI4D relative path.

    Args:
        scene_name: Flat scene name, e.g. "ZY20210800004_H4_C8_N14_S71_s03_T2".

    Returns:
        Relative path, e.g. "ZY20210800004/H4/C8/N14/S71/s03/T2".

    Example:
        >>> scene_name_to_rel_path("ZY20210800004_H4_C8_N14_S71_s03_T2")
        'ZY20210800004/H4/C8/N14/S71/s03/T2'
    """
    return scene_name.replace("_", "/", _HOI4D_PATH_DEPTH - 1)


def get_hoi4d_scene_names(cfg) -> list[str]:
    """Discover HOI4D scene names from release.txt, then apply WAI filters.

    Reads the official HOI4D release list and converts each path to a flat
    WAI scene name. Respects FG_MAX_SCENES (env var) and cfg.max_scenes to
    limit the number of scenes for testing.

    Args:
        cfg: Conversion config with fields: release_txt, max_scenes, root,
             scene_filters.

    Returns:
        List of flat scene name strings, e.g. ["ZY20210800004_H4_C8_N14_S71_s03_T2", ...].
    """
    release_txt = Path(cfg.release_txt)
    if not release_txt.exists():
        raise FileNotFoundError(f"release.txt not found: {release_txt}")

    with open(release_txt) as f:
        rel_paths = [line.strip() for line in f if line.strip()]

    # Convert nested path → flat scene name
    scene_names = [p.replace("/", "_") for p in rel_paths]

    # Apply WAI scene_filters first (e.g. skip already-converted scenes),
    # then limit count — so --max-scenes N means N *unprocessed* scenes.
    scene_names = get_scene_names(cfg, scene_names=scene_names)
    max_scenes = int(os.environ.get("FG_MAX_SCENES", cfg.get("max_scenes", 0) or 0))
    if max_scenes > 0:
        scene_names = scene_names[:max_scenes]
    return scene_names


def process_hoi4d_scene(cfg, scene_name: str) -> None:
    """Convert one HOI4D sequence to WAI format (RGB frames only).

    Symlinks decoded JPEG frames into images/ and writes a minimal
    scene_meta.json. Camera intrinsics and extrinsics are omitted
    until depth and calibration data become available.

    Args:
        cfg: Conversion config with fields: original_root, root,
             dataset_name, version.
        scene_name: Flat WAI scene name, e.g. "ZY20210800004_H4_C8_N14_S71_s03_T2".
    """
    rel_path = scene_name_to_rel_path(scene_name)
    src_rgb_dir = Path(cfg.original_root) / rel_path / "align_rgb"
    target_scene_root = Path(cfg.root) / scene_name
    image_dir = target_scene_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Collect decoded frames (sorted: 00000.jpg, 00001.jpg, ...)
    jpg_files = natsorted(src_rgb_dir.glob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(f"No decoded frames found in {src_rgb_dir}. "
                                "Run decode_rgb.py first.")

    wai_frames = []
    for jpg in jpg_files:
        rel_img_path = Path("images") / jpg.name
        target_img_path = target_scene_root / rel_img_path

        # Symlink source frame into WAI images/ dir
        if target_img_path.exists() or target_img_path.is_symlink():
            target_img_path.unlink()
        os.symlink(jpg.resolve(), target_img_path)

        wai_frames.append({
            "frame_name": jpg.stem,          # "00000"
            "file_path": str(rel_img_path),  # "images/00000.jpg"
            "image": str(rel_img_path),
        })

    # Minimal scene_meta — camera fields omitted until calibration is available
    scene_meta = {
        "scene_name": scene_name,
        "dataset_name": cfg.dataset_name,
        "version": cfg.version,
        "camera_convention": "opencv",
        "shared_intrinsics": False,
        "frames": wai_frames,
        "scene_modalities": {},
        "frame_modalities": {
            "image": {"frame_key": "image", "format": "image"},
        },
    }
    store_data(target_scene_root / "scene_meta.json", scene_meta, "scene_meta")


if __name__ == "__main__":
    cfg = argconf_parse(WAI_PROC_CONFIG_PATH / "conversion/hoi4d.yaml")
    target_root_dir = Path(cfg.root)
    target_root_dir.mkdir(parents=True, exist_ok=True)
    convert_scenes_wrapper(process_hoi4d_scene, cfg,
                           get_original_scene_names_func=get_hoi4d_scene_names)
