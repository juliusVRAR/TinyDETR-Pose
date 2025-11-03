# Pose Estimation Model — README

Quick steps to get the pose estimation pipeline running.

## Prerequisites
- Docker installed and running
- VS Code + “Dev Containers” extension
- Access to the NAS to fetch the debug dataset
- NVIDIA GPU drivers (recommended; the container/tooling handles CUDA)

## 1) Git: clone and checkout branch
```bash
git clone  pose-estimation
cd pose-estimation
git checkout 6d
```

## 2) Dataset prep
- Download the debug dataset from the NAS and unzip it locally.
```bash
/pc3163/nobackup/cache/jkuehn/bop_datasets_debug.zip
unzip /debug_bop_datasets_debug.zip -d 
```
- Mount the unzipped dataset into the container. Edit start_docker.sh and add a volume mount for the debug dataset. For pose estimation, other mounts are not necessary. Example:
```bash
# inside start_docker.sh docker run ...
-v :/workspace/data/debug \
```

## 3) Docker: build, run, attach
- Build/push the image:
```bash
./build_push.sh
```
- Start the container:
```bash
./start_docker.sh
```
- Attach from VS Code using the “Dev Containers” extension (Open Folder in Container).

## 4) Build Deformable Attention ops
Inside the container:
```bash
cd models/ops
python setup.py build install
python test.py # unit test (should see all checking is True)
```
Ensure test.py passes without errors.

## 5) Debug helpers (buggy code locations)
- data_utils/pose_dataset.py — Class MoasicDetection
- data_utils/data_augment.py — function random_affine_single()

Both files have a global variable DEBUG. Set it to True to save debug images under debug/. Example:
```python
# at top of file
DEBUG = True
```

## 6) Run/Debug from VS Code
Use the launch configuration:
- Debug LW-DETR tiny (single proc)

Set breakpoints as needed and start debugging from VS Code.

## Notes
- If ops build fails, ensure the container has CUDA and compiler toolchain available (provided by the image).
- Confirm your dataset is visible in the container at /workspace/data/debug (or your chosen mount path).