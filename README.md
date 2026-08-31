# TinyDETR-Pose 🤏
## Towards End-to-End Real-Time Single-Stage 6DoF Object Pose Estimation with Lightweight Transformers
Real-time 6DoF object pose estimation on resource-constrained hardware remains challenging, as accurate correspondence-based and refinement pipelines typically rely on non-differentiable PnP/RANSAC stages or costly iterative refinement, while recent foundation-model-based approaches incur inference costs that are prohibitive for edge deployment. We present TinyDETR-Pose, a lightweight, end-to-end, single-stage framework that jointly detects objects and regresses their full 6D pose in a single forward pass. Built on the efficient LW-DETR architecture, TinyDETR-Pose formulates detection and pose estimation as a set-prediction problem and attaches dedicated MLP heads for rotation, monocular depth, and projected object center regression to each decoder query, eliminating the need for PnP, NMS (non-maximum suppression), or iterative pose refinement. Object symmetries are handled through a ADD-S loss applied uniformly to all objects, without the need for object-specific loss schedules or separate geodesic/ADD supervision. In addition, predictions are assigned to ground truth using a symmetry-safe Hungarian matcher based on class and 2D spatial cues, yielding stable assignment under symmetry and depth ambiguity. On YCB-V, TinyDETR-Pose achieves a comparable ADD-S AUC of 85.9, while requiring up to 72.7% fewer parameters than other DETR-based single-stage pose-estimation approaches. Due to its compact design, TinyDETR-Pose runs in real time and achieves an inference latency of only ~4.5 ms per frame on an NVIDIA Jetson Nano using TensorRT, demonstrating that accurate end-to-end transformer-based 6D pose estimation can be made practical for edge deployment. 

Link to preprint:
https://arxiv.org/abs/2608.15297
### Accepted at ECCV Workshops 2026 : 11th Workshop on Recovering 6D Object Pose (R6D)

#### Steps to get the pose estimation pipeline running.
## Prerequisites
- NVIDIA GPU drivers (recommended; the container/tooling handles CUDA)

## 1) Git: clone and checkout branch
```bash
git clone https://github.com/juliusVRAR/TinyDETR-Pose.git  
cd TinyDETR-Pose
```

## 2) Dataset prep
- We use the Pose Estimation Transformer (PoET) dataloader.
- Download the YCB-Video dataset and preprocess it with data_utils/data_annotation/ycbv2poet.py

## 3) Get pretrained weights 
To reproduce the exps. you need the pretrained backbone as well as pretrained weights on Objects365 (tiny)
- [Pretrained detector weights](https://github.com/Atten4Vis/LW-DETR#3-preparation) 
- You need the backbone weights: caev2_tiny_300e_objects365 and the LW-DETR weights: LWDETR_tiny_30e_objects365

## 4) Build Deformable Attention ops
Inside the container:
```bash
cd models/ops
python setup.py build install
python test.py # unit test (should see all checking is True)
```
Ensure test.py passes without errors.

## 5) Install and run training:
- Build the dockerfile
- Place the dataset into workspace/data/datasets
- Place the pretrained weights into workspace/data/weights 
- To reproduce the exps. you need the pretrained backbone as well as pretrained weights on Objects365 (tiny)
- Training and Evaluation scripts are placed in scripts/pose/0_tiny/
- For example will start a training on 4 GPUs for 60 epochs :
```bash
bash -c "/workspace/LWDETR/scripts/pose/0_tiny/train.sh"
```
## 6) Pretrained pose estimation weights:
- You can find the pretrained YCB-Video weights on [HF 🤗](https://huggingface.co/JuliusVRAR/TinyDETR-Pose/tree/main)

