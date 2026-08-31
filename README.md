# TinyDETR-Pose: Towards End-to-End Real-Time Single-Stage 6DoF Object Pose Estimation with Lightweight Transformers

Quick steps to get the pose estimation pipeline running.

## Prerequisites
- Docker installed and running
- VS Code + “Dev Containers” extension
- Access to the NAS to fetch the debug dataset
- NVIDIA GPU drivers (recommended; the container/tooling handles CUDA)

## 1) Git: clone and checkout branch
```bash
git clone https://github.com/juliusVRAR/TinyDETR-Pose.git  
cd TinyDETR-Pose
```

## 2) Dataset prep
- Download the debug dataset from the NAS and unzip it locally.
```bash
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