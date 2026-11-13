"""Mesoscope quality control.

The QC is independent of the algorithm used for motion correction and ROI extraction (suite2p or
masknmf), with each algorithm's outputs saved in its own FOV subfolder, e.g. alf/FOV_00/masknmf.

Modules
-------
fov
    QC classes for computing and updating the FOV QC on Alyx.
registration
    Motion registration QC metrics, e.g. for detecting slow Z drift.
task
    Standalone QC tasks.
"""
