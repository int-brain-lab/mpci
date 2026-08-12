"""
Lazy loader for ScanImage multiROI tiff series.

In ScanImage multiROI mode each raw frame is a tall composite image
[total_lines, pixels_per_line] containing all planes stacked vertically,
separated by flyback lines.  Each plane occupies a contiguous band of rows
(its *lineIdx* from the FOV metadata).  There is no temporal interleaving:
every raw temporal frame contributes one sample to every plane simultaneously.

Usage example::

    from mpci.masknmf.io import (
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
from typing import Tuple, Union

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
    fov_list = meta['FOV']
    if plane >= len(fov_list):
        raise ValueError(
            f"plane {plane} out of range — meta JSON has {len(fov_list)} FOVs"
        )
    return [idx - 1 for idx in fov_list[plane]['lineIdx']]


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
        tifs = sorted(folder.glob('*.tif'))
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
    frame on demand.  ``H``/``W`` indices given to ``__getitem__`` are resolved
    to absolute row/column ranges and pushed into the underlying memmap read
    (when ``memmap=True``), rather than reading the full plane band and width
    before subselecting in memory.

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
    memmap:
        Memory-map each file for fast, partial-region reads (default). Falls
        back to ``tifffile.imread`` per-file if a file can't be memory-mapped
        (raw ScanImage tiffs aren't always contiguous/uncompressed); in that
        case the full plane band and width are always read from disk and any
        ``H``/``W`` cropping is applied afterward in memory.
    """

    def __init__(self, file_paths: list[Path], line_indices: list[int], memmap: bool = True):
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
            self._frame_map[g:g + n, 0] = np.arange(g, g + n)
            self._frame_map[g:g + n, 1] = np.arange(n)
            self._frame_map[g:g + n, 2] = file_id
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
    # Intelligent (T, H, W) indexing
    # ------------------------------------------------------------------

    def _resolve_spatial_indexer(self, indexer, size: int):
        """Normalize an H or W indexer (from ``__getitem__``) to an int, slice, or ndarray.

        Negative ints/entries are resolved to positive positions here (rather than left for
        numpy to handle) since row indices get shifted by an absolute offset afterward.
        """
        if indexer is None:
            return slice(0, size, 1)
        if isinstance(indexer, (int, np.integer)):
            idx = int(indexer)
            return idx + size if idx < 0 else idx
        if isinstance(indexer, (slice, range)):
            return slice(*indexer.indices(size))
        if isinstance(indexer, (list, np.ndarray)):
            arr = np.asarray(indexer)
            return np.where(arr < 0, arr + size, arr)
        raise IndexError(f"Invalid spatial index: {indexer!r}")

    def _shift_row_indexer(self, row_indexer):
        """Shift a plane-local row indexer (0..height) to an absolute row in the raw frame."""
        if isinstance(row_indexer, int):
            return row_indexer + self._row_start
        if isinstance(row_indexer, slice):
            # A resolved reverse-order slice's stop may be -1 ("before index 0"); shifting
            # that literally would land on a real, wrong absolute row, so keep it as None.
            stop = None if row_indexer.stop == -1 else row_indexer.stop + self._row_start
            return slice(row_indexer.start + self._row_start, stop, row_indexer.step)
        return row_indexer + self._row_start  # ndarray

    def __getitem__(
        self,
        item: Union[int, list, np.ndarray, slice, range, Tuple[Union[int, np.ndarray, slice, range]]],
    ):
        frame_indexer, item = self._parse_indices(item)
        row_item = item[1] if isinstance(item, tuple) and len(item) > 1 else None
        col_item = item[2] if isinstance(item, tuple) and len(item) > 2 else None

        row_indexer = self._shift_row_indexer(self._resolve_spatial_indexer(row_item, self._height))
        col_indexer = self._resolve_spatial_indexer(col_item, self._width)

        frames = self._compute_at_indices(frame_indexer, row_indexer, col_indexer)
        return frames.astype(self.dtype, copy=False)

    def _compute_at_indices(
        self,
        indices: Union[int, slice, list],
        row_indexer: Union[int, slice, np.ndarray, None] = None,
        col_indexer: Union[int, slice, np.ndarray, None] = None,
    ) -> np.ndarray:
        if row_indexer is None:
            row_indexer = slice(self._row_start, self._row_end, 1)
        if col_indexer is None:
            col_indexer = slice(0, self._width, 1)

        if isinstance(indices, int):
            indices = [indices]
        elif isinstance(indices, slice):
            indices = list(range(*indices.indices(self.shape[0])))
        else:
            indices = list(indices)

        rows = self._frame_map[indices, :]

        # Advanced (array) indexing on more than one axis pairs elements instead of
        # taking the outer product. The frame axis is always advanced here (`local_idx`
        # below), so a fancy row/col index is read in full then applied as its own pass.
        row_is_fancy = isinstance(row_indexer, np.ndarray)
        col_is_fancy = isinstance(col_indexer, np.ndarray)
        read_row = slice(None) if row_is_fancy else row_indexer
        read_col = slice(None) if col_is_fancy else col_indexer

        chunks: list[np.ndarray] = []
        insertion_order = np.zeros(len(rows), dtype=np.int64)

        pos = 0
        for file_id in np.unique(rows[:, 2]):
            mask = rows[:, 2] == file_id
            out_positions = np.where(mask)[0]
            local_idx = rows[mask, 1]

            mm = self._memmaps[file_id] if self._memmap else None
            if mm is not None:
                # Fancy-index the memmap; returns a copied array (not a view), reading only
                # the requested frames/rows/columns rather than the full plane band and width.
                raw = mm[local_idx, read_row, read_col]
            else:
                # Raw TIFF rows can't be partially decoded, so the full plane band and width
                # are always read from disk here; row/col cropping is applied afterward.
                raw = tifffile.imread(self._file_paths[file_id], key=local_idx.tolist())
                if raw.ndim == 2:
                    raw = raw[None]
                raw = raw[:, read_row, read_col]

            if row_is_fancy:
                raw = raw[:, row_indexer, :] if raw.ndim == 3 else raw[:, row_indexer]
            if col_is_fancy:
                raw = raw[..., col_indexer]

            chunks.append(raw.astype(self._dtype, copy=False))
            insertion_order[pos:pos + len(out_positions)] = out_positions
            pos += len(out_positions)

        stacked = np.concatenate(chunks, axis=0)
        perm = np.argsort(insertion_order)
        return stacked[perm]


if __name__ == '__main__':
    import numpy as np
    import time

    session_path = Path('/mnt/whiterussian/Subjects/SP072/2025-10-01/001')
    folders = sorted(session_path.glob('raw_imaging_data_??'))
    
    META = folders[0] / '_ibl_rawImagingData.meta.json'

    print('Collecting tiff paths...')
    fps = collect_tiff_paths(folders)
    print('  %d files' % len(fps))

    print()
    print('Plane 0:')
    lines0 = read_fov_line_indices(META, 0)
    print('  rows %d..%d  (%d lines)' % (lines0[0], lines0[-1], len(lines0)))
    t0 = time.time()
    ld0 = ScanImageTiffSeriesLoader(fps, lines0)
    print('  init: %.2fs  shape=%s  dtype=%s' % (time.time()-t0, ld0.shape, ld0.dtype))

    t0 = time.time()
    f0 = ld0[0:5]
    print('  [0:5] shape=%s  took %.2fs' % (str(f0.shape), time.time()-t0))
    print('  frame0 mean=%.1f  min=%d  max=%d' % (f0[0].mean(), f0[0].min(), f0[0].max()))

    print()
    print('Plane 7:')
    lines7 = read_fov_line_indices(META, 7)
    print('  rows %d..%d  (%d lines)' % (lines7[0], lines7[-1], len(lines7)))
    ld7 = ScanImageTiffSeriesLoader(fps, lines7)
    print('  shape=%s' % str(ld7.shape))
    f7 = ld7[0:5]
    print('  frame0 mean=%.1f  min=%d  max=%d' % (f7[0].mean(), f7[0].min(), f7[0].max()))

    print()
    ops = np.load(session_path.joinpath('suite2p/plane0/ops.npy'), allow_pickle=True).item()
    s2p = int(np.sum(ops['frames_per_folder'][:3]))
    print('Loader frames: %d  s2p 3-folder: %d  match=%s' % (ld0.shape[0], s2p, ld0.shape[0]==s2p))

    print()
    print('Mean image correlation (our 200-frame vs s2p full dataset):')
    chunk = ld0[0:200].astype(np.float32)
    our_mean = chunk.mean(axis=0)
    s2p_mean = ops['meanImg']
    r = float(np.corrcoef(our_mean.ravel(), s2p_mean.ravel())[0,1])
    print('  r=%.4f (expected high, ~0.9+ since both are plane 0)' % r)
