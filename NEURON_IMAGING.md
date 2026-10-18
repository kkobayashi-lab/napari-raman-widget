# Neuron imaging workspace

Use the existing conda environment `confocal_control_gui_cellpose4` (Cellpose 4.2.1.1).
Start with `launch_cellpose4_napari.bat` in the workspace root.
The launcher sets PYTHONPATH to all four sibling worktrees, so imports use this
workspace without changing the existing environment's editable installations.
It calls conda through base Python because the installed conda executable fails.

Each component is an independent Git repository on branch `neuron_imaging`:
`cns-control`, `napari-raman-widget`, `raman-control`, and `raman-mda-engine`.
Run Git commands inside the relevant component folder. The parent folder is a
workspace container, not a Git repository. The original folders remain on `main`.
The original workspace's empty top-level .git folder contains no Git history;
the four component repositories contain the actual history.

The snapshot package contains source ZIPs and Git bundles for all four repositories,
workspace files, environment inventories, and a commit manifest. Restore a repository
with `git clone -b neuron_imaging <component>.bundle <component>`.
Existing worktrees depend on their original repository metadata; the bundles provide
an independent backup. Do not delete the original repositories while using worktrees.

Acquisition datasets, calibration outputs, caches, and temporary backups are retained
in the original workspace and excluded from this source package. Hardware drivers,
Micro-Manager configuration, and saved application settings remain machine-local.
The environment inventories record installed versions; they are not a tested clean
installation recipe. The original root requirements.txt is retained as legacy metadata.
