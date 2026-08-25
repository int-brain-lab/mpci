"""
Lazy loader for ScanImage multiROI tiff series.

In ScanImage multiROI mode each raw frame is a tall composite image
[total_lines, pixels_per_line] containing all planes stacked vertically,
separated by flyback lines.  Each plane occupies a contiguous band of rows
(its *lineIdx* from the FOV metadata).  There is no temporal interleaving:
every raw temporal frame contributes one sample to every plane simultaneously.

Usage example::

    from masknmf.arrays.scanimage_loader import (
        ScanImageTiffSeriesLoader,
        read_fov_line_indices,
        collect_tiff_paths,
    )

    folders = [
        '/data/raw_imaging_data_00',
        '/data/raw_imaging_data_01',
    ]
    meta_json = '/data/raw_imaging_data_00/_ibl_rawImagingData.meta.json'
    line_indices = read_fov_line_indices(meta_json, plane=0)
    file_paths   = collect_tiff_paths(folders)
    loader       = ScanImageTiffSeriesLoader(file_paths, line_indices)
    frames       = loader[0:100]   # shape (100, 512, 512), lazy
"""

from pathlib import Path
import json
from typing import Union

import numpy as np
import tifffile

from mpci.scanimage.io import patch_imaging_meta
from masknmf.arrays.array_interfaces import LazyFrameLoader


def read_fov_line_indices(meta_json_path: str, plane: int) -> list[int]:
    """
    Return 0-indexed row indices for *plane* from an IBL rawImagingData meta JSON.

    The ``FOV[plane]['lineIdx']`` field uses 1-based indices; this function
    converts them to 0-based.

    Parameters
    ----------
    meta_json_path:
        Path to ``_ibl_rawImagingData.meta.json``.
    plane:
        0-indexed plane number.

    Returns
    -------
    list[int]
        0-indexed row indices of this plane within each raw tiff frame.
    """
    with open(meta_json_path) as f:
        meta = patch_imaging_meta(json.load(f))
    fov_list = meta["FOV"]
    if plane >= len(fov_list):
        raise ValueError(f"plane {plane} out of range — meta JSON has {len(fov_list)} FOVs")
    return [idx - 1 for idx in fov_list[plane]["lineIdx"]]


def collect_tiff_paths(folders: list[Path]) -> list[Path]:
    """
    Return a sorted list of tiff file paths from one or more recording folders.

    Files are sorted per-folder by filename, then concatenated in the order
    the folders are supplied so that the temporal sequence is preserved.

    Parameters
    ----------
    folders:
        Ordered list of directory paths (e.g. ``raw_imaging_data_00``,
        ``raw_imaging_data_01``, …).

    Returns
    -------
    list[Path]
        Sorted tiff paths ready to pass to :class:`ScanImageTiffSeriesLoader`.
    """
    paths: list[Path] = []
    for folder in sorted(folders):
        tifs = sorted(folder.glob("*.tif"))
        paths.extend(tifs)
    return paths


def _raw_frame_count(filepath: str) -> int:
    """Return the number of frames in a tiff file."""
    with tifffile.TiffFile(filepath) as tf:
        return len(tf.pages)


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------


class ScanImageTiffSeriesLoader(LazyFrameLoader):
    """
    Lazy plane-specific loader for a ScanImage multiROI tiff series.

    Each raw tiff frame is a composite image that contains **all** planes
    simultaneously at different row ranges.  This class exposes a single plane
    as a ``(T, H, W)`` array-like by slicing the appropriate rows from each
    frame on demand.

    Parameters
    ----------
    file_paths:
        Ordered list of tiff file paths.  Use :func:`collect_tiff_paths` to
        build this from a list of ``raw_imaging_data_XX`` directories.
    line_indices:
        0-indexed row indices that define this plane's pixel band within each
        raw frame.  Use :func:`read_fov_line_indices` to obtain these from an
        ``_ibl_rawImagingData.meta.json`` file, or pass them directly.

        Must be a contiguous ascending range (the class slices
        ``frame[lines[0]:lines[-1]+1, :]``).
    """

    def __init__(self, file_paths: list[Path], line_indices: list[int], memmap: bool = False):
        self._file_paths = list(file_paths)
        self._lines = np.asarray(line_indices, dtype=np.int64)
        self._row_start = int(self._lines[0])
        self._row_end = int(self._lines[-1]) + 1
        self._height = len(self._lines)
        self._memmap = memmap

        # Dtype and pixel width from first frame
        with tifffile.TiffFile(file_paths[0]) as tf:
            page0 = tf.pages[0].asarray()
        self._width = page0.shape[-1]
        self._dtype = page0.dtype

        # Count raw frames per file
        raw_counts = [_raw_frame_count(fp) for fp in file_paths]
        self._n_frames = sum(raw_counts)

        # Frame map: col0=global_idx, col1=local_idx_within_file, col2=file_id
        self._frame_map = np.empty((self._n_frames, 3), dtype=np.int64)
        g = 0
        for file_id, n in enumerate(raw_counts):
            self._frame_map[g : g + n, 0] = np.arange(g, g + n)
            self._frame_map[g : g + n, 1] = np.arange(n)
            self._frame_map[g : g + n, 2] = file_id
            g += n

        # Attempt to memory-map each file as (n_frames, raw_height, raw_width).
        # tifffile.memmap requires contiguous, uncompressed pages; raw ScanImage
        # tiffs often don't satisfy this, so we fall back per-file to imread.
        if memmap:
            self._memmaps: list[np.ndarray | None] = []
            for fp in file_paths:
                try:
                    self._memmaps.append(tifffile.memmap(fp))
                except (ValueError, tifffile.TiffFileError):
                    self._memmaps.append(None)

    # ------------------------------------------------------------------
    # LazyFrameLoader interface
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, int, int]:
        return self._n_frames, self._height, self._width

    @property
    def ndim(self) -> int:
        return 3

    # ------------------------------------------------------------------
    # Informational properties
    # ------------------------------------------------------------------

    @property
    def memmap(self) -> bool:
        return self._memmap

    @property
    def line_indices(self) -> np.ndarray:
        """0-indexed row range for this plane within each raw frame."""
        return self._lines

    # ------------------------------------------------------------------
    # Core read logic
    # ------------------------------------------------------------------

    def _compute_at_indices(self, indices: Union[int, slice, list]) -> np.ndarray:
        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[0])))
        else:
            indices = list(indices)

        rows = self._frame_map[indices, :]

        chunks: list[np.ndarray] = []
        insertion_order = np.zeros(len(rows), dtype=np.int64)

        pos = 0
        for file_id in np.unique(rows[:, 2]):
            mask = rows[:, 2] == file_id
            out_positions = np.where(mask)[0]
            local_idx = rows[mask, 1]

            mm = self._memmaps[file_id] if self._memmap else None
            if mm is not None:
                # Fancy-index the memmap; returns a copied array (not a view).
                raw = mm[local_idx, self._row_start : self._row_end, :]
            else:
                raw = tifffile.imread(self._file_paths[file_id], key=local_idx.tolist())
                if raw.ndim == 2:
                    raw = raw[None]
                raw = raw[:, self._row_start : self._row_end, :]

            chunks.append(raw.astype(self._dtype, copy=False))
            insertion_order[pos : pos + len(out_positions)] = out_positions
            pos += len(out_positions)

        stacked = np.concatenate(chunks, axis=0)
        perm = np.argsort(insertion_order)
        return stacked[perm]


if __name__ == "__main__":
    import numpy as np
    import time

    session_path = Path("/mnt/whiterussian/Subjects/SP072/2025-10-01/001")
    folders = sorted(session_path.glob("raw_imaging_data_??"))

    META = folders[0] / "_ibl_rawImagingData.meta.json"

    print("Collecting tiff paths...")
    fps = collect_tiff_paths(folders)
    print("  %d files" % len(fps))

    print()
    print("Plane 0:")
    lines0 = read_fov_line_indices(META, 0)
    print("  rows %d..%d  (%d lines)" % (lines0[0], lines0[-1], len(lines0)))
    t0 = time.time()
    ld0 = ScanImageTiffSeriesLoader(fps, lines0)
    print("  init: %.2fs  shape=%s  dtype=%s" % (time.time() - t0, ld0.shape, ld0.dtype))

    t0 = time.time()
    f0 = ld0[0:5]
    print("  [0:5] shape=%s  took %.2fs" % (str(f0.shape), time.time() - t0))
    print("  frame0 mean=%.1f  min=%d  max=%d" % (f0[0].mean(), f0[0].min(), f0[0].max()))

    print()
    print("Plane 7:")
    lines7 = read_fov_line_indices(META, 7)
    print("  rows %d..%d  (%d lines)" % (lines7[0], lines7[-1], len(lines7)))
    ld7 = ScanImageTiffSeriesLoader(fps, lines7)
    print("  shape=%s" % str(ld7.shape))
    f7 = ld7[0:5]
    print("  frame0 mean=%.1f  min=%d  max=%d" % (f7[0].mean(), f7[0].min(), f7[0].max()))

    print()
    ops = np.load(session_path.joinpath("suite2p/plane0/ops.npy"), allow_pickle=True).item()
    s2p = int(np.sum(ops["frames_per_folder"][:3]))
    print(
        "Loader frames: %d  s2p 3-folder: %d  match=%s" % (ld0.shape[0], s2p, ld0.shape[0] == s2p)
    )

    print()
    print("Mean image correlation (our 200-frame vs s2p full dataset):")
    chunk = ld0[0:200].astype(np.float32)
    our_mean = chunk.mean(axis=0)
    s2p_mean = ops["meanImg"]
    r = float(np.corrcoef(our_mean.ravel(), s2p_mean.ravel())[0, 1])
    print("  r=%.4f (expected high, ~0.9+ since both are plane 0)" % r)
