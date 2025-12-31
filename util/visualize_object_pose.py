import cv2
import numpy as np
from torch import is_tensor
import torch
import copy
import os
import matplotlib
from typing import Dict, List, Optional, Union, Tuple, Sequence
from util.utils import save_annotated_image

from torchvision.utils import draw_bounding_boxes
from torchvision.io import write_png
import random
from pathlib import Path
import trimesh
from PIL import Image

class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        self.palette = [self.hex2rgb(c) for c in matplotlib.colors.TABLEAU_COLORS.values()]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


colors = Colors()  # create instance for 'from utils.plots import colors'


def draw_bbox_2d(img, box, label, score, conf = 0.6, thickness=1, gt=False):

    cls_id = int(label)
    if score < conf:
        return
    x0, y0 = int(box[0]), int(box[1])
    x1, y1 = int(box[0] + box[2]), int(box[1] + box[3])

    if gt:
        color = (0, 255, 0)
    else:
        color = colors(cls_id)

    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    if gt:
        cv2.putText(img, str(label), ((x0+x1)//2, (y0+y1)//2), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=thickness)
    else:
        cv2.putText(img, str(label), (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness=thickness)
    return img


def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)
    box = [tuple(kpt) for kpt in box]
    #front??? to check
    cv2.line(img, box[0], box[1], colour, thickness)
    cv2.line(img, box[1], box[2], colour, thickness)
    cv2.line(img, box[2], box[3], colour, thickness)
    cv2.line(img, box[0], box[3], colour, thickness)
    #back
    cv2.line(img, box[4], box[5], colour, thickness)
    cv2.line(img, box[5], box[6], colour, thickness)
    cv2.line(img, box[6], box[7], colour, thickness)
    cv2.line(img, box[4], box[7], colour, thickness)
    #sides
    cv2.line(img, box[0], box[4], colour, thickness)
    cv2.line(img, box[1], box[5], colour, thickness)
    cv2.line(img, box[2], box[6], colour, thickness)
    cv2.line(img, box[3], box[7], colour, thickness)

    return img


def project_3d_2d(pts_3d, rotation_vec, translation_vec, camera_matrix):
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    xformed_3d = np.matmul(pts_3d, rotation_mat.T) + translation_vec
    xformed_3d[:,:3] = xformed_3d[:,:3]/xformed_3d[:,2:3]
    projected_2d = np.matmul(xformed_3d, camera_matrix.reshape((3, 3)).T)[:, :2]
    #projected_2d_corrected = copy.deepcopy(projected_2d)
    #projected_2d_corrected = cv2.undistortPoints(projected_2d_corrected, camera_matrix.reshape(3,3), np.array([0.04112172, -0.4798174, 0.0, 0.0, 1.890084 ]))
    #projected_2d_corrected = np.squeeze(projected_2d_corrected)
    #projected_2d_corrected[:, 0] = camera_matrix[0] * projected_2d_corrected[:, 0] + camera_matrix[2]
    #projected_2d_corrected[:, 1] = camera_matrix[4] * projected_2d_corrected[:, 1] + camera_matrix[5]

    return projected_2d


def draw_6d_pose(img, data_list , camera_matrix, class_to_cuboid=None, conf = 0.6, class_to_model=None, gt=True, out_dir=None, id=None, img_cuboid=None, img_mask=None, img_2dod=None):

    if is_tensor(img):
        img_temp = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img_temp = np.asarray(img_temp, dtype=np.uint8)
        img_temp = np.ascontiguousarray(img_temp)
        if img_mask is None:
            img_mask = copy.deepcopy(img_temp)
        if img_cuboid is None:
            img_cuboid = copy.deepcopy(img_temp)
        if img_2dod is None:
            img_2dod = copy.deepcopy(img_temp)

    for pose in data_list:
        #Rotation matrix is recovered using the formula given in the article
        #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
        pose_type = "gt" if gt else "pred"
        if pose['missing_det'] and not gt:
            continue
        score = pose['score'] if pose_type == "pred" else 1.0
        if score < conf:
            continue
        rotation, translation, bbox, xy,  label = \
            np.array(pose['rotation_{}'.format(pose_type)]), np.array(pose['translation_{}'.format(pose_type)]), \
                        pose['bbox_{}'.format(pose_type)], pose['xy_{}'.format(pose_type)], pose['category_id']
        if gt:
            colour = (0, 255, 0)
        else:
            colour = colors(label)

        img_cuboid = cv2.circle(img_cuboid, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cad_model_2d = project_3d_2d(class_to_model[label], rotation, translation, camera_matrix)
        cad_model_2d = cad_model_2d.astype(np.int32)
        cad_model_2d[:, 0][cad_model_2d[:, 0] >= img.shape[2]] = img.shape[2] - 1
        cad_model_2d[:, 1][cad_model_2d[:, 1] >= img.shape[1]] = img.shape[1] - 1
        cad_model_2d[cad_model_2d < 0] = 0
        img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
        img_mask = cv2.circle(img_mask, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cuboid_corners_2d = project_3d_2d(class_to_cuboid[label], rotation, translation, camera_matrix)
        img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=colour)

        img_2dod = draw_bbox_2d(img_2dod, bbox, label, score, conf=0.6, thickness=2, gt=gt)

    outfile_pose = os.path.join(out_dir, "vis_pose", "{:012}_{}_pose.png".format(id, pose_type))
    outfile_mask = os.path.join(out_dir, "vis_pose", "{:012}_{}_mask.png".format(id, pose_type))
    outfile_2d_od = os.path.join(out_dir, "vis_pose", "{:012}_{}_2d_od.png".format(id, pose_type))
    if not gt:
        cv2.imwrite(outfile_pose, img_cuboid)
        cv2.imwrite(outfile_mask, img_mask)
        cv2.imwrite(outfile_2d_od, img_2dod)
    return img_cuboid, img_mask, img_2dod

def plot_one_box(x, im, im_cuboid=None, im_mask=None, color=None, label=None, line_thickness=3, orig_shape=None, pose=None,
                 cad_models=None, camera_matrix=None, block_x=None, block_y=None, cls_names=None):
    # Plots one bounding box on image 'im' using OpenCV
    assert im.data.contiguous, 'Image not contiguous. Apply np.ascontiguousarray(im) to plot_on_box() input image.'
    tl = line_thickness or round(0.002 * (im.shape[0] + im.shape[1]) / 2) + 1  # line/font thickness
    color = color
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(im, c1, c2, color, thickness=tl*2//3, lineType=cv2.LINE_AA)
    if label:
        if len(label.split(' ')) > 1:
            label = label.split(' ')[-1]
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=tl / 6, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(im, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im, label, (c1[0], c1[1] - 2), 0, tl / 6, [225, 255, 255], thickness=tf//2, lineType=cv2.LINE_AA)

    if cls_names is not None:  #This block is enabled from demo.py. Have to be enabled for
        score = x[4] * x[-2]
        text = '{} : {:.1f}%'.format(cls_names[int(x[-1])], score * 100)
        txt_color = color
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        txt_bk_color = (np.array(color) * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            im,
            (c1[0], c1[1] + 1),
            (c1[0] + txt_size[0] + 1, c1[1] + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(im, text, (c1[0], c1[1] + txt_size[1]), font, 0.4, txt_color, thickness=1)


    
    plot_object_pose(im, im_cuboid, im_mask, pose, cad_models, camera_matrix, color, label, block_x, block_y, orig_shape=orig_shape)


def plot_object_pose(im, im_cuboid, im_mask, pose, cad_models, camera_matrix, color, label, block_x, block_y, orig_shape=None):

    img_2dod = copy.deepcopy(im)
    rotation = pose['rotation_vec']
    translation = pose['translation_vec']
    xy = pose['xy']

    img_cuboid = cv2.circle(im_cuboid, (int(xy[0])+block_x, int(xy[1])+block_y), 3, (0, 0, 255), -1)
    cad_model_2d = project_3d_2d(pts_3d=cad_models.class_to_model[int(label)],
                                 rotation_vec=rotation, translation_vec=translation, camera_matrix=camera_matrix)
    cad_model_2d = cad_model_2d.astype(np.int32)
    cad_model_2d[:, 0][cad_model_2d[:,0] >= orig_shape[1]] = orig_shape[1] - 1
    cad_model_2d[:, 1][cad_model_2d[:,1] >= orig_shape[0]] = orig_shape[0] - 1
    cad_model_2d[cad_model_2d < 0] = 0
    cad_model_2d[:, 0] += block_x
    cad_model_2d[:, 1] += block_y

    im_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = color
    img_mask = cv2.circle(im_mask, (int(xy[0])+block_x, int(xy[1])+block_y), 3, (0, 0, 255), -1)

    cuboid_corners_2d = project_3d_2d(pts_3d=cad_models.models_corners[int(label)],
                                rotation_vec=rotation, translation_vec=translation, camera_matrix=camera_matrix
    )
    cuboid_corners_2d[:, 0] += block_x
    cuboid_corners_2d[:, 1] += block_y
    img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=color)

    #img_2dod = draw_bbox_2d(img_2dod, bbox, label, score, conf=0.6, thickness=2)

    return img_mask, img_cuboid, img_2dod


def project_to_so3(R: np.ndarray) -> np.ndarray:
    # SVD projection to nearest rotation matrix
    U, S, Vt = np.linalg.svd(R)
    R_proj = U @ Vt
    if np.linalg.det(R_proj) < 0:
        U[:, -1] *= -1
        R_proj = U @ Vt
    return R_proj

def euler_to_mat(roll, pitch, yaw):
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [ 0,   0, 1]])
    Ry = np.array([[ cp, 0, sp],
                   [  0, 1,  0],
                   [-sp, 0, cp]])
    Rx = np.array([[1,  0,   0],
                   [0, cr, -sr],
                   [0, sr,  cr]])
    return Rz @ Ry @ Rx

def quat_to_mat(q):
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)],
    ], dtype=np.float32)

def denormalize_pose(pos_norm: np.ndarray,
                     rot_norm: np.ndarray,
                     trans_min: np.ndarray = None,
                     trans_max: np.ndarray = None,
                     trans_mean: np.ndarray = None,
                     trans_std: np.ndarray = None,
                     rot_mode: str = 'matrix_minmax',  # 'matrix_minmax'|'euler_minmax'|'quat_minmax'
                     euler_range: tuple = (-np.pi, np.pi)) -> tuple[np.ndarray, np.ndarray]:
    # Translation
    if trans_min is not None and trans_max is not None:
        pos = trans_min + pos_norm * (trans_max - trans_min)
    elif trans_mean is not None and trans_std is not None:
        pos = pos_norm * trans_std + trans_mean
    else:
        raise ValueError("Provide translation min/max or mean/std to denormalize.")

    # Rotation
    if rot_mode == 'matrix_minmax':
        R = 2.0 * rot_norm - 1.0  # back to [-1,1]
        R = project_to_so3(R)
    elif rot_mode == 'euler_minmax':
        lo, hi = euler_range
        angles = lo + rot_norm * (hi - lo)  # map [0,1] -> [lo,hi]
        R = euler_to_mat(angles[0], angles[1], angles[2])
    elif rot_mode == 'quat_minmax':
        q = 2.0 * rot_norm - 1.0
        R = quat_to_mat(q)
    else:
        raise ValueError(f"Unknown rot_mode: {rot_mode}")
    return pos.astype(np.float32), R.astype(np.float32)

def _corners_from_obj_info(obj_info: dict) -> np.ndarray:
    """
    Build 8 corners (8,3) from object AABB in object coordinates.
    Corner ordering is lexicographic over x,y,z for consistent edge drawing.
    """
    min_x, min_y, min_z = obj_info["min_x"], obj_info["min_y"], obj_info["min_z"]
    sx, sy, sz = obj_info["size_x"], obj_info["size_y"], obj_info["size_z"]
    max_x, max_y, max_z = min_x + sx, min_y + sy, min_z + sz
    xs = [min_x, max_x]
    ys = [min_y, max_y]
    zs = [min_z, max_z]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)  # (8,3)
    return corners

# Edge pairs for the cube with the above corner ordering
_EDGE_PAIRS = [
    (0,1), (2,3), (4,5), (6,7),   # along z
    (0,2), (1,3), (4,6), (5,7),   # along y
    (0,4), (1,5), (2,6), (3,7)    # along x
]

def _corners_from_obj_info(obj_info: dict) -> np.ndarray:
    """
    Build 8 corners (8,3) from object AABB in object coordinates.
    Corner order is lexicographic over x, y, z.
    """
    min_x, min_y, min_z = obj_info["min_x"], obj_info["min_y"], obj_info["min_z"]
    sx, sy, sz = obj_info["size_x"], obj_info["size_y"], obj_info["size_z"]
    max_x, max_y, max_z = min_x + sx, min_y + sy, min_z + sz
    xs = [min_x, max_x]
    ys = [min_y, max_y]
    zs = [min_z, max_z]
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)  # (8,3)
    return corners

_EDGE_PAIRS = [
    (0,1), (2,3), (4,5), (6,7),   # along z
    (0,2), (1,3), (4,6), (5,7),   # along y
    (0,4), (1,5), (2,6), (3,7)    # along x
]

def visualize_object_keypoints(
    cam: dict,
    targets: dict,
    obj_infos_by_label: Union[Dict[int, dict], Sequence[dict]],
    image: Optional[np.ndarray] = None,
    draw_2d_boxes: bool = True,
    use_depth_scale: bool = False,
    label_offset: int = 0,                # set to 1 if labels are 1-based
    box_color: Tuple[int,int,int] = (0,255,0),
    thickness: int = 2,
) -> Tuple[np.ndarray, List[Optional[np.ndarray]]]:
    """
    Draw oriented 3D bounding boxes selected by per-target labels.

    Inputs:
      - cam: {'fx','fy','cx','cy','width','height', ('depth_scale' optional)}
      - targets:
          'relative_position': (N,3) torch tensor (object translation in camera frame)
          'relative_rotation': (N,3,3) torch tensor (object->camera rotation)
           'labels':(N,) torch tensor of ints
          'boxes': (N,4) torch tensor [x1,y1,x2,y2] (optional)
      - obj_infos_by_label:
          dict: {label_int: obj_info_dict}
          or sequence: list/tuple where obj_info = seq[label - label_offset]
        obj_info_dict must have keys: min_x,min_y,min_z,size_x,size_y,size_z
      - image: optional HxWx3 uint8 BGR image; if None, a blank canvas is created.
      - draw_2d_boxes: draw 2D boxes if present.
      - use_depth_scale: multiply translations by cam['depth_scale'] if True.
      - label_offset: use 1 if labels start at 1; 0 if they start at 0.

    Assumptions:
      - relative_rotation maps object coords to camera coords (R_cam_obj).
      - relative_position is in the same linear units as the object sizes (or scaled via depth_scale).
      - Camera frame: X right, Y down, Z forward.

    Returns:
      - img_out: image with drawn 3D boxes
      - projections: list length N of (8,2) arrays (pixel coords) or None if not drawable
    """
    if cam is not None:
        fx, fy = float(cam['fx']), float(cam['fy'])
        cx, cy = float(cam['cx']), float(cam['cy'])
        W, H = int(cam['width']), int(cam['height'])
        depth_scale = float(cam.get('depth_scale', 1.0))

    if image is None:
        img_out = np.zeros((H, W, 3), dtype=np.uint8)
    else:
        img_out = image.copy()
    
    
    # Extract tensors
    pos = targets['relative_position']; rot = targets['relative_rotation']; labels = targets['labels']
    assert pos.ndim == 2 and pos.size(-1) == 3, "relative_position must be (N,3)"
    assert rot.ndim == 3 and rot.shape[1:] == (3,3), "relative_rotation must be (N,3,3)"
    assert labels.ndim == 1 and labels.shape[0] == pos.shape[0], "labels must be (N,)"
    N = pos.shape[0]

    pos_np = pos.detach().cpu().numpy().astype(np.float32)   # (N,3)
    rot_np = rot.detach().cpu().numpy().astype(np.float32)   # (N,3,3)
    labels_np = labels.detach().cpu().numpy().astype(int)    # (N,)
    if use_depth_scale:
        pos_np *= depth_scale

    if 'intrinsics' in targets:
        Ks = targets['intrinsics']  # (N,4)

        if 'keypoints_2d' in targets:
            kpts = targets['keypoints_2d']
            if hasattr(kpts, 'cpu'):
                kpts_np = kpts.cpu().numpy()
            else:
                kpts_np = np.asarray(kpts)
            for i, pts in enumerate(kpts_np):
                for (u,v) in pts.astype(int):
                    if 0 <= u < img_out.shape[1] and 0 <= v < img_out.shape[0]:
                        cv2.circle(img_out, (u,v), 5, (0,255), -1)
                        cv2.putText(img_out, str(labels_np[i]), (int(u)+4, int(v)-4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
            # do not alter camera matrix; return image and keypoints
            return img_out, kpts

        # project centers (example)
        pts_3d = pos  # using object centers
        proj = []
        for i, p in enumerate(pts_3d):
            if Ks[i].shape[0] == 9:
                fx, fy, cx, cy = Ks[i][0], Ks[i][4], Ks[i][2], Ks[i][5]
            else:
                fx, fy, cx, cy = Ks[i].tolist()
            x, y, z = p.tolist()
            if z <= 0:
                proj.append([-1, -1])
                continue
            u = fx * x / z + cx
            v = fy * y / z + cy
            proj.append([u, v])
            cv2.circle(img_out, (int(u), int(v)), 5, (0,0,255), -1)
            cv2.putText(img_out, str(labels_np[i]), (int(u)+4, int(v)-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)
        projections = torch.tensor(proj)
        return img_out, projections
    
    else:
        Ks = torch.tensor([[cam['fx'], cam['fy'], cam['cx'], cam['cy']]], 
                        dtype=torch.float32).repeat(rot.shape[0],1)


    # Optional: draw 2D boxes
    if draw_2d_boxes and ('boxes' in targets):
        # img_out = save_annotated_image(image=img_out, 
        #                            targets=targets, 
        #                            is_bbbox_coords_normalized=True, 
        #                            is_corrected_bbx_coords=False,
        #                            output_path="Test_here.png")
        img_out= img_out.detach().cpu().numpy() 
        img_out = img_out.transpose(1, 2, 0)  # to HWC
        img_out = img_out[..., ::-1]
        img_out = img_out*255
    # Label -> obj_info resolver
    def get_info(label: int) -> Optional[dict]:
        if isinstance(obj_infos_by_label, dict):
            return obj_infos_by_label.get(str(label), None)
        # sequence: index by (label - label_offset)
        idx = label - label_offset
        if 0 <= idx < len(obj_infos_by_label):
            return obj_infos_by_label[idx]
        return None

    projections: List[Optional[np.ndarray]] = []

    for i in range(N):
        info = get_info(labels_np[i])
        if info is None:
            continue
            
        t = pos_np[i].reshape(3, 1)     # (3,1)
        R = rot_np[i]                   # (3,3)


        # Draw center
        if t[2,0] > 1e-6:
            uc = int(round(fx * t[0,0]/t[2,0] + cx))
            vc = int(round(fy * t[1,0]/t[2,0] + cy))
            cv2.circle(img_out, (uc, vc), 10, (0, 0, 255), -1)

            # Optional: label text
            cv2.putText(img_out, str(labels_np[i]), (uc+4, vc-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1, cv2.LINE_AA)

    return img_out, projections

class YCBVVisualizer:
    def __init__(self, models_path):
        """
        Initialize YCB-V visualizer
        
        Args:
            models_path: Path to YCB-V models folder containing PLY files
                        (e.g., '/path/to/ycbv/models')
        """
        self.models_path = Path(models_path)
        self.models = {}
        
    def load_model(self, obj_id):
        """Load 3D model for given object ID"""
        if obj_id in self.models:
            return self.models[obj_id]
        
        model_path = self.models_path / f'obj_{obj_id:06d}.ply'
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        mesh = trimesh.load(str(model_path))
        self.models[obj_id] = mesh
        return mesh
    
    def get_3d_bbox(self, mesh):
        """Compute 3D bounding box corners from mesh"""
        vertices = mesh.vertices
        min_coords = vertices.min(axis=0)
        max_coords = vertices.max(axis=0)
        
        # 8 corners of bounding box
        corners = np.array([
            [min_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], max_coords[1], max_coords[2]],
            [min_coords[0], max_coords[1], max_coords[2]]
        ])
        return corners
    
    def project_points(self, points_3d, K, R, t):
        """Project 3D points to 2D image coordinates"""
        # Transform to camera coordinates
        points_cam = (R @ points_3d.T + t.reshape(3, 1)).T
        
        # Project to image
        points_2d = (K @ points_cam.T).T
        points_2d = points_2d[:, :2] / points_2d[:, 2:3]
        
        return points_2d
    
    def draw_3d_bbox(self, img, corners_2d, color=(0, 255, 0), thickness=2):
        """Draw 3D bounding box on image"""
        corners_2d = corners_2d.astype(int)
        
        # Draw bottom face
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[(i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw top face
        for i in range(4, 8):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[4 + (i + 1) % 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        # Draw vertical edges
        for i in range(4):
            pt1 = tuple(corners_2d[i])
            pt2 = tuple(corners_2d[i + 4])
            cv2.line(img, pt1, pt2, color, thickness)
        
        return img
    

    def visualize_single_image(self, 
                               img: np.ndarray, 
                               annotations: dict, 
                               K: Optional[np.ndarray] = None, 
                               show_mesh=False, 
                               sample_points=1000,
                               conf_threshold: Optional[float] = None,
                               scores: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Visualize a single image with 3D models and bounding boxes
        
        Args:
            img: Input image (numpy array, RGB or BGR, or torch tensor)
            annotations: List of dictionaries with keys:
                        - 'obj_id': object ID
                        - 'cam_R_m2c': 3x3 rotation matrix (list, array, or tensor)
                        - 'cam_t_m2c': 3x1 translation vector (list, array, or tensor)
            K: 3x3 camera intrinsic matrix (list, array, or tensor)
            show_mesh: If True, overlay CAD model vertices on the image
            sample_points: Number of mesh points to sample per object (if show_mesh=True)
        
        Returns:
            Visualized image with 3D bounding boxes and optionally CAD models
        """
        # Convert torch tensors to numpy if needed
        if hasattr(img, 'cpu'):  # Check if it's a torch tensor
            img = img.cpu().numpy()


        # Convert image to RGB if needed
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        vis_img = img.copy()
        labels = annotations['labels']
        rel_pos = annotations['relative_position']
        rel_rot = annotations['relative_rotation']
        # optional per-object intrinsics
        intrinsics_rows = annotations.get('intrinsics', None)
        
         # normalize tensor types
        if hasattr(labels, "cpu"):
            labels_np = labels.cpu().numpy()
        else:
            labels_np = np.asarray(labels)
        if hasattr(rel_pos, "cpu"):
            rel_pos_np = rel_pos.cpu().numpy()
        else:
            rel_pos_np = np.asarray(rel_pos)
        if hasattr(rel_rot, "cpu"):
            rel_rot_np = rel_rot.cpu().numpy()
        else:
            rel_rot_np = np.asarray(rel_rot)

        N = rel_pos_np.shape[0]

        # Build per-object K list
        if intrinsics_rows is not None:
            if hasattr(intrinsics_rows, "cpu"):
                intrinsics_rows = intrinsics_rows.cpu().numpy()
            assert intrinsics_rows.shape[0] == N and intrinsics_rows.shape[1] == 4, "intrinsics must be (N,4)"
            Ks_list = []
            for i in range(N):
                fx, fy, cx, cy = intrinsics_rows[i]
                K_i = np.array([[fx, 0,  cx],
                                [0,  fy, cy],
                                [0,  0,   1]], dtype=np.float32)
                Ks_list.append(K_i)
            
            for label, rot, trans, K, score in zip(annotations["labels"], annotations["relative_rotation"], annotations["relative_position"], Ks_list, scores if scores is not None else [1.0]*N):
                if score > conf_threshold:
                    obj_id = label
                    R = np.array(rot).reshape(3, 3)
                    trans = trans*1000.0 # In the dataset the translation is in meters, convert to mm.
                    t = np.array(trans).flatten()
                    
                    try:
                        mesh = self.load_model(obj_id)
                    except FileNotFoundError:
                        print(f"Warning: Model for obj_id {obj_id} not found, skipping...")
                        continue
                    
                    # Generate random color for this object
                    np.random.seed(obj_id)  # Consistent color per object
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    
                    # Get 3D bounding box
                    bbox_3d = self.get_3d_bbox(mesh)
                    
                    # Transform bbox to camera coordinates and project to 2D
                    bbox_2d = self.project_points(bbox_3d, K, R, t)
                    
                    # Draw bounding box
                    vis_img = self.draw_3d_bbox(vis_img, bbox_2d, color, 2)
                    
                    # Overlay CAD model if requested
                    if show_mesh:
                        vertices = mesh.vertices.copy()
                        
                        # Sample vertices for visualization
                        if len(vertices) > sample_points:
                            indices = np.random.choice(len(vertices), sample_points, replace=False)
                            vertices = vertices[indices]
                        
                        # Project vertices
                        vertices_2d = self.project_points(vertices, K, R, t)
                        
                        # Draw mesh points
                        for pt in vertices_2d:
                            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                                cv2.circle(vis_img, tuple(pt.astype(int)), 1, color, -1)
                    
                    # Add label
                    center_2d = bbox_2d.mean(axis=0).astype(int)
                    cv2.putText(vis_img, f'obj_{obj_id}_{score:.2f}', tuple(center_2d), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
                
            return vis_img

            
        else:
            # Ensure K is numpy array (handle tensor)
            if hasattr(K, 'cpu'):
                K = K.cpu().numpy()
            K = np.array(K).reshape(3, 3)
            
            for label, rot, trans, score in zip(annotations["labels"], annotations["relative_rotation"], annotations["relative_position"], scores if scores is not None else [1.0]*N):
                if score > conf_threshold:
                    obj_id = label
                    R = np.array(rot).reshape(3, 3)
                    trans = trans*1000.0 # In the dataset the translation is in meters, convert to mm.
                    t = np.array(trans).flatten()
                    
                    try:
                        mesh = self.load_model(obj_id)
                    except FileNotFoundError:
                        print(f"Warning: Model for obj_id {obj_id} not found, skipping...")
                        continue
                    
                    # Generate random color for this object
                    np.random.seed(obj_id)  # Consistent color per object
                    color = tuple(np.random.randint(100, 255, 3).tolist())
                    
                    # Get 3D bounding box
                    bbox_3d = self.get_3d_bbox(mesh)
                    
                    # Transform bbox to camera coordinates and project to 2D
                    bbox_2d = self.project_points(bbox_3d, K, R, t)
                    
                    # Draw bounding box
                    vis_img = self.draw_3d_bbox(vis_img, bbox_2d, color, 2)
                    
                    # Overlay CAD model if requested
                    if show_mesh:
                        vertices = mesh.vertices.copy()
                        
                        # Sample vertices for visualization
                        if len(vertices) > sample_points:
                            indices = np.random.choice(len(vertices), sample_points, replace=False)
                            vertices = vertices[indices]
                        
                        # Project vertices
                        vertices_2d = self.project_points(vertices, K, R, t)
                        
                        # Draw mesh points
                        for pt in vertices_2d:
                            if 0 <= pt[0] < img.shape[1] and 0 <= pt[1] < img.shape[0]:
                                cv2.circle(vis_img, tuple(pt.astype(int)), 1, color, -1)
                    
                    # Add label
                    center_2d = bbox_2d.mean(axis=0).astype(int)
                    cv2.putText(vis_img, f'obj_{obj_id}_{score:.2f}', tuple(center_2d), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
            
            return vis_img


def save_image_with_bboxes(img: torch.Tensor,
                           boxes: torch.Tensor,
                           labels: torch.Tensor,
                           out_path: str,
                           clamp: bool = False):
    """
    img: CHW tensor in [0,1] or [0,255]
    boxes: (N,4) xyxy in pixel coords (float or int)
    labels: (N,) int tensor
    """
    if img.dim() != 3:
        raise ValueError("img must be CHW")
    # Convert to uint8 RGB
    im = img.clone()
    if clamp:
        im = im.clamp(0,1)
    if im.max() <= 1.0:
        im = (im * 255.0).to(torch.uint8)
    else:
        im = im.to(torch.uint8)

    if boxes.numel() == 0:
        write_png(im, str(out_path))
        return

    b = boxes.clone()
    if b.dtype != torch.int64 and b.dtype != torch.int32:
        b = b.round().to(torch.int64)

    lab_str = [str(int(l.item())) for l in labels]

    drawn = draw_bounding_boxes(
        im,
        b,
        labels=lab_str,
        colors=[tuple(random.sample(range(255), 3)) for _ in range(len(b))],
        width=2
    )
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    write_png(drawn, str(out_path))

    
def project_3d_2d(pts_3d, rotation_vec, translation_vec, camera_matrix):
    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    xformed_3d = np.matmul(pts_3d, rotation_mat.T) + translation_vec
    xformed_3d[:,:3] = xformed_3d[:,:3]/xformed_3d[:,2:3]
    projected_2d = np.matmul(xformed_3d, camera_matrix.reshape((3, 3)).T)[:, :2]
    #projected_2d_corrected = copy.deepcopy(projected_2d)
    #projected_2d_corrected = cv2.undistortPoints(projected_2d_corrected, camera_matrix.reshape(3,3), np.array([0.04112172, -0.4798174, 0.0, 0.0, 1.890084 ]))
    #projected_2d_corrected = np.squeeze(projected_2d_corrected)
    #projected_2d_corrected[:, 0] = camera_matrix[0] * projected_2d_corrected[:, 0] + camera_matrix[2]
    #projected_2d_corrected[:, 1] = camera_matrix[4] * projected_2d_corrected[:, 1] + camera_matrix[5]

    return projected_2d
#
def draw_cuboid_2d(img, cuboid_corners, colour = (0, 255, 0), thickness = 2):
    box = np.copy(cuboid_corners).astype(np.int32)
    box = [tuple(kpt) for kpt in box]
    #front??? to check
    cv2.line(img, box[0], box[1], colour, thickness)
    cv2.line(img, box[1], box[2], colour, thickness)
    cv2.line(img, box[2], box[3], colour, thickness)
    cv2.line(img, box[0], box[3], colour, thickness)
    #back
    cv2.line(img, box[4], box[5], colour, thickness)
    cv2.line(img, box[5], box[6], colour, thickness)
    cv2.line(img, box[6], box[7], colour, thickness)
    cv2.line(img, box[4], box[7], colour, thickness)
    #sides
    cv2.line(img, box[0], box[4], colour, thickness)
    cv2.line(img, box[1], box[5], colour, thickness)
    cv2.line(img, box[2], box[6], colour, thickness)
    cv2.line(img, box[3], box[7], colour, thickness)

    return img

def draw_6d_pose(img, 
                 data_list , 
                 camera_matrix, 
                 class_to_cuboid=None, 
                 conf = 0.6, 
                 class_to_model=None, 
                 gt=True, 
                 out_dir=None, 
                 id=None, 
                 img_cuboid=None, 
                 img_mask=None, 
                 img_2dod=None):

    if is_tensor(img):
        img_temp = copy.deepcopy(img).cpu().numpy().transpose(1, 2, 0)
        img_temp = np.asarray(img_temp, dtype=np.uint8)
        img_temp = np.ascontiguousarray(img_temp)
        if img_mask is None:
            img_mask = copy.deepcopy(img_temp)
        if img_cuboid is None:
            img_cuboid = copy.deepcopy(img_temp)
        if img_2dod is None:
            img_2dod = copy.deepcopy(img_temp)

    

    for rotation, translation, bbox, label in zip(data_list['relative_rotation'], data_list['relative_position'], data_list['boxes'], data_list['labels']):
        #Rotation matrix is recovered using the formula given in the article
        #https://towardsdatascience.com/better-rotation-representations-for-accurate-pose-estimation-e890a7e1317f
        pose_type = "gt" if gt else "pred"

        # project centers (example)
        pt_3d = translation  # using object centers
        

        fx, fy, cx, cy = camera_matrix["fx"], camera_matrix["fy"], camera_matrix["cx"], camera_matrix["cy"]
        x, y, z = pt_3d.tolist()
        u = fx * x / z + cx
        v = fy * y / z + cy
        xy = (u, v)
        if gt:
            colour = (0, 255, 0)
        else:
            colour = colors(label)

        img_cuboid = cv2.circle(img_cuboid, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cad_model_2d = project_3d_2d(class_to_model[label], rotation, translation, camera_matrix)
        cad_model_2d = cad_model_2d.astype(np.int32)
        cad_model_2d[:, 0][cad_model_2d[:, 0] >= img.shape[2]] = img.shape[2] - 1
        cad_model_2d[:, 1][cad_model_2d[:, 1] >= img.shape[1]] = img.shape[1] - 1
        cad_model_2d[cad_model_2d < 0] = 0
        img_mask[cad_model_2d[:, 1], cad_model_2d[:, 0]] = colour
        img_mask = cv2.circle(img_mask, (int(xy[0]), int(xy[1])), 3, (0, 0, 255), -1)

        cuboid_corners_2d = project_3d_2d(class_to_cuboid[label], rotation, translation, camera_matrix)
        img_cuboid = draw_cuboid_2d(img=img_cuboid, cuboid_corners=cuboid_corners_2d, colour=colour)

        img_2dod = draw_bbox_2d(img_2dod, bbox, label, 1.0, conf=0.6, thickness=2, gt=gt)

    outfile_pose = os.path.join(out_dir, "vis_pose", "{:012}_{}_pose.png".format(id, pose_type))
    outfile_mask = os.path.join(out_dir, "vis_pose", "{:012}_{}_mask.png".format(id, pose_type))
    outfile_2d_od = os.path.join(out_dir, "vis_pose", "{:012}_{}_2d_od.png".format(id, pose_type))
    if not gt:
        cv2.imwrite(outfile_pose, img_cuboid)
        cv2.imwrite(outfile_mask, img_mask)
        cv2.imwrite(outfile_2d_od, img_2dod)
    return img_cuboid, img_mask, img_2dod

def draw_object_centers(img_input, centers_xy, conf=None, scores=None, radius=4, color=(0, 255, 0)):
    """
    Draw circles for object centers.
    Accepts:
      - torch.Tensor (C,H,W) in [0,1]
      - PIL.Image
      - np.ndarray (H,W,C) uint8 RGB or BGR
    centers_xy: (N,2) pixel coords (u,v)
    Returns BGR uint8 ndarray.
    """
    if centers_xy is None or len(centers_xy) == 0:
        # Just convert and return
        if isinstance(img_input, torch.Tensor):
            img = (img_input.permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8)
        elif isinstance(img_input, Image.Image):
            img = np.array(img_input)
        else:
            img = img_input
        return img[..., ::-1].copy()  # BGR
    # Normalize input to RGB uint8
    if isinstance(img_input, torch.Tensor):
        img = (img_input.permute(1,2,0).numpy().clip(0,1) * 255).astype(np.uint8)
    elif isinstance(img_input, Image.Image):
        img = np.array(img_input)
    else:
        img = img_input
    # If grayscale expand
    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    # Assume current img is RGB; convert to BGR for OpenCV drawing
    img_bgr = img[..., ::-1].copy()
    for (u, v) in centers_xy.tolist():
        if conf is not None and scores is not None:
            idx = centers_xy.tolist().index([u, v])
            if scores[idx] < conf:
                continue
        cv2.circle(img_bgr, (int(round(u)), int(round(v))), radius, color, -1, lineType=cv2.LINE_AA)
    return img_bgr