from __future__ import print_function, division, absolute_import

import os
import shutil
import json

from scipy import spatial
import numpy as np
from scipy.linalg import logm
import numpy.linalg as LA

class PoseEvaluator(object):
    def __init__(self, models, classes, model_info, model_symmetry, depth_scale=0.1):
        self.models = models
        self.classes = classes
        self.models_info = model_info
        self.model_symmetry = model_symmetry

        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}
        self.depth_scale = depth_scale

        self.reset()

    def reset(self):
        self.poses_pred = {}
        self.poses_gt = {}
        self.poses_img = {}
        self.camera_intrinsics = {}
        self.num = {}

        for cls in self.classes:
            self.num[cls] = 0.
            self.poses_pred[cls] = []
            self.poses_gt[cls] = []
            self.poses_img[cls] = []
            self.camera_intrinsics[cls] = []

    def _prepare_output_dir(self, output_path, subdir):
        """Helper to prepare output directory"""
        output_dir = os.path.join(output_path, subdir)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        return output_dir

    def _write_log_lines(self, log_file, lines):
        """Batch write multiple lines to reduce I/O operations"""
        log_file.write('\n'.join(lines) + '\n')

    def _compute_metrics_for_class(self, cls_name, cls_poses_pred, cls_poses_gt, 
                                   model_pts, eval_method, thresholds):
        """Optimized metric computation for a single class"""
        n_poses = len(cls_poses_gt)
        if n_poses == 0:
            return None
        
        # Pre-allocate error array
        errors = np.empty(n_poses, dtype=np.float32)
        
        # Compute all errors
        for j, (pose_pred, pose_gt) in enumerate(zip(cls_poses_pred, cls_poses_gt)):
            if eval_method == 'adi':
                errors[j] = self.calc_adi(model_pts, pose_pred, pose_gt)
            else:  # 'add'
                errors[j] = self.calc_add(model_pts, pose_pred, pose_gt)
        
        # Vectorized threshold comparison
        results = {}
        for thresh_name, thresh_value in thresholds.items():
            if isinstance(thresh_value, np.ndarray):
                # For 'mean' threshold (array of values)
                results[thresh_name] = np.sum(errors[:, None] < thresh_value, axis=0)
            else:
                # For single threshold values
                results[thresh_name] = np.sum(errors < thresh_value)
        
        return results

    def _evaluate_pose_generic(self, output_path, metric_name, eval_method_func):
        """Generic evaluation function to reduce code duplication"""
        output_dir = self._prepare_output_dir(output_path, metric_name.lower())
        
        log_file = open(os.path.join(output_dir, f"{metric_name.lower()}.log"), 'w')
        json_file = open(os.path.join(output_dir, f"{metric_name.lower()}.json"), 'w')

        # Use references instead of deep copies (data is not modified)
        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        models = self.models

        log_lines = [
            f'\n* {"-" * 100} *',
            f' {metric_name:^}',
            f'* {"-" * 100} *',
            ''
        ]
        self._write_log_lines(log_file, log_lines)

        n_classes = len(self.classes)
        count_all = np.zeros(n_classes, dtype=np.float32)
        
        # Pre-define thresholds
        dx = 0.0001
        thresholds_dict = {
            '0.02': 0.02,
            '0.05': 0.05,
            '0.10': 0.10,
            'mean': np.arange(0, 0.1, dx, dtype=np.float32)
        }
        num_thresh = len(thresholds_dict['mean'])
        
        count_correct = {
            '0.02': np.zeros(n_classes, dtype=np.float32),
            '0.05': np.zeros(n_classes, dtype=np.float32),
            '0.10': np.zeros(n_classes, dtype=np.float32),
            'mean': np.zeros((n_classes, num_thresh), dtype=np.float32)
        }

        results = {
            "thresholds": [0.02, 0.05, 0.10]
        }

        self.classes = sorted(self.classes)
        num_valid_class = len(self.classes)
        
        # Compute metrics for all classes
        for i, cls_name in enumerate(self.classes):
            cls_poses_pred = poses_pred[cls_name]
            cls_poses_gt = poses_gt[cls_name]
            model_pts = models[cls_name]['pts']
            n_poses = len(cls_poses_gt)
            count_all[i] = n_poses
            
            if n_poses == 0:
                continue
            
            # Determine evaluation method
            if metric_name == 'Metric ADD(-S)':
                eval_method = 'adi' if self.model_symmetry[cls_name] else 'add'
            elif metric_name == 'Metric ADD-S':
                eval_method = 'adi'
            else:  # ADD
                eval_method = 'add'
            
            # Compute metrics
            cls_results = self._compute_metrics_for_class(
                cls_name, cls_poses_pred, cls_poses_gt, 
                model_pts, eval_method, thresholds_dict
            )
            
            if cls_results:
                for key in count_correct:
                    if key == 'mean':
                        count_correct[key][i] = cls_results[key]
                    else:
                        count_correct[key][i] = cls_results[key]
            
            results[cls_name] = {
                "threshold": {k: v[i].tolist() if isinstance(v[i], np.ndarray) else float(v[i]) 
                             for k, v in count_correct.items()}
            }

        # Compute accuracies
        from scipy.integrate import simpson
        sum_acc = {'mean': 0.0, '0.02': 0.0, '0.05': 0.0, '0.10': 0.0}
        
        for i, cls_name in enumerate(self.classes):
            if count_all[i] == 0:
                continue
            
            log_lines = [f"** {cls_name} **"]
            
            # Compute AUC
            area = simpson(count_correct['mean'][i] / count_all[i], dx=dx) / 0.1
            acc_mean = area * 100
            sum_acc['mean'] += acc_mean
            
            # Compute threshold accuracies
            accuracies = {}
            for thresh in ['0.02', '0.05', '0.10']:
                acc = 100.0 * count_correct[thresh][i] / count_all[i]
                sum_acc[thresh] += acc
                accuracies[thresh] = float(acc)
            
            log_lines.extend([
                f'threshold=[0.0, 0.10], area: {acc_mean:.2f}',
                f'threshold=0.02, correct poses: {int(count_correct["0.02"][i])}, all poses: {int(count_all[i])}, accuracy: {accuracies["0.02"]:.2f}',
                f'threshold=0.05, correct poses: {int(count_correct["0.05"][i])}, all poses: {int(count_all[i])}, accuracy: {accuracies["0.05"]:.2f}',
                f'threshold=0.10, correct poses: {int(count_correct["0.10"][i])}, all poses: {int(count_all[i])}, accuracy: {accuracies["0.10"]:.2f}',
                ''
            ])
            
            self._write_log_lines(log_file, log_lines)
            
            results[cls_name]["accuracy"] = {
                'n_poses': float(count_all[i]),
                **accuracies,
                'auc': float(acc_mean)
            }

        # Write summary
        summary_lines = [
            "=" * 30,
            f"---------- {metric_name} performance over {num_valid_class} classes -----------",
            "** iter 1 **",
            f'threshold=[0.0, 0.10], area: {sum_acc["mean"] / num_valid_class:.2f}',
            f'threshold=0.02, mean accuracy: {sum_acc["0.02"] / num_valid_class:.2f}',
            f'threshold=0.05, mean accuracy: {sum_acc["0.05"] / num_valid_class:.2f}',
            f'threshold=0.10, mean accuracy: {sum_acc["0.10"] / num_valid_class:.2f}',
            "=" * 30
        ]
        self._write_log_lines(log_file, summary_lines)
        
        results["accuracy"] = {
            k: float(v / num_valid_class) for k, v in sum_acc.items()
        }

        log_file.close()
        json.dump(results, json_file, indent=2)
        
        json_file.close()

    def evaluate_pose_adds(self, output_path):
        """Evaluate 6D pose by ADD(-S) metric"""
        self._evaluate_pose_generic(output_path, 'Metric ADD(-S)', None)

    def evaluate_pose_adi(self, output_path):
        """Evaluate 6D pose by ADD-S metric"""
        self._evaluate_pose_generic(output_path, 'Metric ADD-S', None)

    def evaluate_pose_add(self, output_path):
        """Evaluate 6D pose by ADD Metric"""
        self._evaluate_pose_generic(output_path, 'Metric ADD', None)

    def calculate_class_avg_translation_error(self, output_path):
        """Calculate average translation error in meters"""
        output_dir = self._prepare_output_dir(output_path, "avg_t_error")
        
        log_file = open(os.path.join(output_dir, "avg_t_error.log"), 'w')
        json_file = open(os.path.join(output_dir, "avg_t_error.json"), 'w')

        log_lines = [
            f'\n* {"-" * 100} *',
            ' Metric Average Translation Error in Meters',
            f'* {"-" * 100} *',
            ''
        ]
        self._write_log_lines(log_file, log_lines)

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        
        all_errors = []
        avg_translation_errors = {}
        
        for cls in self.classes:
            cls_poses_pred = poses_pred[cls]
            cls_poses_gt = poses_gt[cls]
            
            if not cls_poses_pred:
                avg_translation_errors[cls] = None
                continue
            
            # Vectorized translation error calculation
            t_preds = np.array([pose[:3, 3] for pose in cls_poses_pred])
            t_gts = np.array([pose[:3, 3] for pose in cls_poses_gt])
            errors = np.linalg.norm(t_preds - t_gts, axis=1)
            
            avg_error = float(np.mean(errors))
            avg_translation_errors[cls] = avg_error
            all_errors.extend(errors)
            
            log_file.write(f"Class: {cls} \t\t {avg_error}\n")
        
        total_avg_error = float(np.mean(all_errors)) if all_errors else 0.0
        log_file.write(f"All:\t\t\t\t\t {total_avg_error}\n")
        avg_translation_errors["mean"] = [total_avg_error]

        log_file.close()
        json.dump(avg_translation_errors, json_file, indent=2)
        json_file.close()

    def calculate_class_avg_rotation_error(self, output_path):
        """Calculate average rotation error in degrees"""
        output_dir = self._prepare_output_dir(output_path, "avg_rot_error")
        
        log_file = open(os.path.join(output_dir, "avg_rot_error.log"), 'w')
        json_file = open(os.path.join(output_dir, "avg_rot_error.json"), 'w')

        log_lines = [
            f'\n* {"-" * 100} *',
            ' Metric Average Rotation Error in Degrees',
            f'* {"-" * 100} *',
            ''
        ]
        self._write_log_lines(log_file, log_lines)

        poses_pred = self.poses_pred
        poses_gt = self.poses_gt
        
        all_errors = []
        avg_rotation_errors = {}

        for cls in self.classes:
            cls_pose_pred = poses_pred[cls]
            cls_pose_gt = poses_gt[cls]
            
            if not cls_pose_pred:
                avg_rotation_errors[cls] = None
                continue
            
            errors = []
            for pose_est, pose_gt in zip(cls_pose_pred, cls_pose_gt):
                rot_est = pose_est[:3, :3]
                rot_gt = pose_gt[:3, :3]
                rot = rot_est @ rot_gt.T
                trace = np.clip(np.trace(rot), -1.0, 3.0)
                angle_diff = np.degrees(np.arccos(0.5 * (trace - 1.0)))
                errors.append(angle_diff)
            
            avg_error = float(np.mean(errors))
            avg_rotation_errors[cls] = avg_error
            all_errors.extend(errors)
            
            log_file.write(f"Class: {cls} \t\t {avg_error}\n")
        
        total_avg_error = float(np.mean(all_errors)) if all_errors else 0.0
        log_file.write(f"All:\t\t\t\t\t {total_avg_error}\n")
        avg_rotation_errors["mean"] = [total_avg_error]

        log_file.close()
        json.dump(avg_rotation_errors, json_file, indent=2)
        json_file.close()

    # Optimized transformation functions
    def transform_pts(self, pts, rot, t):
        """Vectorized 3D point transformation"""
        return (rot @ pts.T + t.reshape(3, 1)).T

    def project_pts(self, pts, rot, t, K):
        """Vectorized 2D projection"""
        if K.shape == (9,):
            K = K.reshape(3, 3)
        pts_t = rot @ pts.T + t.reshape(3, 1)
        pts_c_t = K @ pts_t
        return (pts_c_t[:2] / pts_c_t[2]).T

    def calc_add(self, pts, pose_pred, pose_gt):
        """Optimized ADD calculation"""
        pts_est = self.transform_pts(pts, pose_pred[:3, :3], pose_pred[:, 3])
        pts_gt = self.transform_pts(pts, pose_gt[:3, :3], pose_gt[:, 3])
        return np.linalg.norm(pts_est - pts_gt, axis=1).mean()

    def calc_adi(self, pts, pose_pred, pose_gt):
        """Optimized ADI calculation"""
        pts_pred = self.transform_pts(pts, pose_pred[:3, :3], pose_pred[:, 3])
        pts_gt = self.transform_pts(pts, pose_gt[:3, :3], pose_gt[:, 3])
        
        nn_index = spatial.cKDTree(pts_pred)
        nn_dists, _ = nn_index.query(pts_gt, k=1)
        return nn_dists.mean()

    # Keep other helper methods
    def se3_mul(self, RT1, RT2):
        """Concat 2 RT transform"""
        R1, T1 = RT1[:3, :3], RT1[:3, 3:4]
        R2, T2 = RT2[:3, :3], RT2[:3, 3:4]
        
        RT_new = np.zeros((3, 4), dtype=np.float32)
        RT_new[:3, :3] = R1 @ R2
        RT_new[:3, 3] = (R1 @ T2 + T1).ravel()
        return RT_new

    def proj(self, pts, pose_pred, pose_gt, K):
        """Average re-projection error in 2D"""
        proj_pred = self.project_pts(pts, pose_pred[:3, :3], pose_pred[:, 3], K)
        proj_gt = self.project_pts(pts, pose_gt[:3, :3], pose_gt[:, 3], K)
        return np.linalg.norm(proj_pred - proj_gt, axis=1).mean()

    def calc_rotation_error(self, rot_pred, r_gt):
        """Calculate angular geodesic rotation error"""
        temp = logm(rot_pred.T @ r_gt)
        rd_rad = LA.norm(temp, 'fro') / np.sqrt(2)
        return rd_rad * 180.0 / np.pi