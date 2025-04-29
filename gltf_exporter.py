import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import sys

try:
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Attributes, Accessor, BufferView, Buffer,
        PERSPECTIVE, ORTHOGRAPHIC, Asset, Camera
    )
    PYGLTFLIB_AVAILABLE = True
except ImportError:
    GLTF2 = None # Define GLTF2 as None if import fails
    PYGLTFLIB_AVAILABLE = False
    # Define dummy classes for type hinting if pygltflib is not available
    class GLTF2: pass
    class Scene: pass
    class Node: pass
    class Mesh: pass
    class Primitive: pass
    class Attributes: pass
    class Accessor: pass
    class BufferView: pass
    class Buffer: pass
    class Asset: pass
    class Camera: pass
    PERSPECTIVE = "PERSPECTIVE"


# Coordinate system transformation (OpenCV world -> glTF world)
# X_glTF =  X_OpenCV
# Y_glTF =  Z_OpenCV
# Z_glTF = -Y_OpenCV
OPENCV_TO_GLTF_ROT = np.array([
    [1, 0, 0],
    [0, 0, 1],
    [0, -1, 0]
], dtype=np.float32)

# Camera local coordinate transformation (OpenCV camera -> glTF camera)
# X_glTF =  X_OpenCV
# Y_glTF = -Y_OpenCV
# Z_glTF = -Z_OpenCV
CAMERA_CONVENTION_ROT = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, -1]
], dtype=np.float32)


def export_scene_to_gltf(
    filename: str,
    results: Dict[str, Any],
    image_paths: List[str],
    image_dimensions: Dict[int, Tuple[int, int]],
    generator_name: str = "Pointgram"
) -> Tuple[bool, str]:
    """
    Constructs and writes calibration results (from PyCOLMAP) to a GLTF file.

    Args:
        filename: Path to save the GLTF file.
        results: Dictionary containing calibration data from PyCOLMAP.
                 Expects 'poses', 'points_3d', 'point_ids', 'registered_indices', 'intrinsics'.
        image_paths: List of original image file paths.
        image_dimensions: Dictionary mapping image index to (width, height).
        generator_name: Name to use in the GLTF asset generator field.

    Returns:
        A tuple (success: bool, message: str).
    """
    if not PYGLTFLIB_AVAILABLE:
         return False, "pygltflib library is required but not found. Install with 'pip install pygltflib'."

    if not results or not isinstance(results, dict):
         return False, "Invalid or empty 'results' dictionary provided."

    required_keys = ['poses', 'points_3d', 'point_ids', 'registered_indices', 'intrinsics']
    missing_keys = [key for key in required_keys if key not in results]
    if missing_keys:
        return False, f"Results dictionary is missing required keys: {', '.join(missing_keys)}"

    if not isinstance(results.get('intrinsics'), dict):
         return False, "Results 'intrinsics' dictionary is missing or invalid."

    try:
        poses = results['poses']
        points3d_list_opencv = results['points_3d']
        point_ids = results['point_ids']
        registered_indices = results['registered_indices']
        intrinsics_source = results['intrinsics']

        # Convert points and transform coordinate system
        points3d_opencv = np.array(points3d_list_opencv, dtype=np.float32)
        if points3d_opencv.size > 0:
            if points3d_opencv.ndim != 2 or points3d_opencv.shape[1] != 3:
                return False, f"Invalid shape for points_3d numpy array: {points3d_opencv.shape}. Expected (N, 3)."
            points3d_gltf = (OPENCV_TO_GLTF_ROT @ points3d_opencv.T).T
        else:
            print("Warning: No 3D points found in results to export.")
            points3d_gltf = np.empty((0, 3), dtype=np.float32)

        gltf = GLTF2(); gltf.asset = Asset(version="2.0", generator=generator_name)
        gltf.scene = 0; scene = Scene(nodes=[]); gltf.scenes.append(scene)

        img_idx_to_gltf_cam_idx: Dict[int, int] = {} # Map original image index to gltf.cameras index
        znear = 0.01; zfar = 1000.0

        # Convert keys to int just in case they were loaded as strings from JSON
        registered_indices_int = [int(k) for k in registered_indices]

        if not isinstance(intrinsics_source, dict):
             return False, "'intrinsics' data is missing or not a dictionary."

        # Create Camera Definitions
        for img_idx in registered_indices_int:
             if img_idx not in poses:
                 print(f"Warning: Pose missing for registered camera index {img_idx}. Skipping camera definition.")
                 continue

             K_matrix = None
             intrinsic_data = intrinsics_source.get(img_idx)
             if intrinsic_data and 'K' in intrinsic_data:
                  K_matrix = np.array(intrinsic_data['K'], dtype=np.float32)
             else:
                  print(f"Warning: Missing intrinsic data or 'K' matrix for camera index {img_idx}. Skipping camera definition.")
                  continue

             dims = image_dimensions.get(img_idx)

             # Validate K and dimensions before proceeding
             if K_matrix is None or K_matrix.shape != (3, 3):
                 print(f"Warning: Invalid K matrix shape for camera index {img_idx}. Skipping camera definition.")
                 continue
             if np.isnan(K_matrix).any() or np.isinf(K_matrix).any():
                  print(f"Warning: NaN or Inf found in K matrix for camera index {img_idx}. Skipping camera definition.")
                  continue
             if dims is None or len(dims) != 2 or dims[0] <= 0 or dims[1] <= 0:
                 print(f"Warning: Invalid or missing dimensions for camera index {img_idx}. Skipping camera definition.")
                 continue

             img_w, img_h = dims
             fx, fy = K_matrix[0, 0], K_matrix[1, 1]
             # Calculate yfov and aspectRatio from intrinsics and dimensions
             if abs(fy) < 1e-9: yfov = np.pi / 2
             else: yfov = 2 * np.arctan( (img_h / 2.0) / fy )
             if abs(fx) < 1e-9 or abs(img_h * fx) < 1e-9: aspect_ratio = img_w / img_h if img_h > 0 else 1.0
             else: aspect_ratio = (img_w * fy) / (img_h * fx)
             yfov = np.clip(yfov, 1e-6, np.pi - 1e-6)
             aspect_ratio = max(1e-6, aspect_ratio)

             cam_def_name = f"CameraDef_{img_idx}"
             if 0 <= img_idx < len(image_paths):
                 try: cam_def_name = f"{os.path.splitext(os.path.basename(image_paths[img_idx]))[0]}_Def"
                 except Exception: pass

             camera = Camera(type=PERSPECTIVE, name=cam_def_name)
             camera.perspective = {"aspectRatio": aspect_ratio, "yfov": yfov, "znear": znear, "zfar": zfar}
             gltf.cameras.append(camera)

             current_gltf_cam_idx = len(gltf.cameras) - 1
             img_idx_to_gltf_cam_idx[img_idx] = current_gltf_cam_idx

        # Create Camera Nodes (Referencing the definitions)
        num_exported_cameras = 0; exported_cam_indices = []
        for img_idx in registered_indices_int:
            if img_idx not in img_idx_to_gltf_cam_idx or img_idx not in poses:
                continue

            pose_dict = poses[img_idx]
            R_c2w_cv = np.array(pose_dict['R'], dtype=np.float32)
            t_c2w_cv = np.array(pose_dict['t'], dtype=np.float32).flatten()

            if R_c2w_cv.shape != (3, 3):
                print(f"Warning: Invalid R matrix shape {R_c2w_cv.shape} for camera {img_idx}. Skipping node."); continue
            if t_c2w_cv.shape != (3,):
                print(f"Warning: Invalid t vector shape {t_c2w_cv.shape} for camera {img_idx}. Skipping node."); continue
            if np.isnan(R_c2w_cv).any() or np.isinf(R_c2w_cv).any() or np.isnan(t_c2w_cv).any() or np.isinf(t_c2w_cv).any():
                print(f"Warning: NaN/Inf in pose for camera {img_idx}. Skipping node."); continue

            try:
                # Calculate GLTF pose: Convert OpenCV camera pose to GLTF world pose
                # R_c2w_gltf = R_world_gltf_from_world_cv @ R_c2w_cv @ R_camera_cv_from_camera_gltf
                R_c2w_gltf = OPENCV_TO_GLTF_ROT @ R_c2w_cv @ CAMERA_CONVENTION_ROT
                # C_gltf = R_world_gltf_from_world_cv @ C_cv
                C_gltf = OPENCV_TO_GLTF_ROT @ t_c2w_cv

                T_cam_to_world_gltf = np.eye(4, dtype=np.float32)
                T_cam_to_world_gltf[:3, :3] = R_c2w_gltf
                T_cam_to_world_gltf[:3, 3] = C_gltf

                if np.isnan(T_cam_to_world_gltf).any() or np.isinf(T_cam_to_world_gltf).any():
                     print(f"Warning: NaN/Inf in final transform matrix for camera {img_idx}. Skipping node."); continue

                node_matrix = T_cam_to_world_gltf.flatten(order='F').tolist() # glTF expects column-major

                cam_node_name = f"CameraNode_{img_idx}"
                if 0 <= img_idx < len(image_paths):
                    try: cam_node_name = os.path.splitext(os.path.basename(image_paths[img_idx]))[0]
                    except Exception: pass

                gltf_camera_index = img_idx_to_gltf_cam_idx[img_idx]

                node = Node( name=cam_node_name, camera=gltf_camera_index, matrix=node_matrix )
                gltf.nodes.append(node); scene.nodes.append(len(gltf.nodes) - 1)
                num_exported_cameras += 1; exported_cam_indices.append(img_idx)

            except (ValueError, TypeError, KeyError, IndexError, np.linalg.LinAlgError) as cam_err:
                print(f"Warning: Skipping camera node creation for index '{img_idx}' due to error: {cam_err}"); continue

        # Create Empty Nodes for Points
        num_exported_points = 0; exported_point_indices = []
        if points3d_gltf.size > 0:
             if len(point_ids) != len(points3d_gltf):
                 print(f"Warning: Mismatch between number of point IDs ({len(point_ids)}) and 3D points ({len(points3d_gltf)}). Using sequential IDs.")
                 point_ids_to_use = list(range(len(points3d_gltf)))
             else:
                 point_ids_to_use = point_ids

             for i, point_gltf in enumerate(points3d_gltf):
                 if np.isnan(point_gltf).any() or np.isinf(point_gltf).any(): continue
                 point_id = point_ids_to_use[i]
                 node = Node(name=f"Point_{point_id}", translation=point_gltf.tolist())
                 gltf.nodes.append(node); scene.nodes.append(len(gltf.nodes) - 1)
                 num_exported_points += 1; exported_point_indices.append(point_id)

        # Summary and Save
        export_summary = (f"Exported {len(gltf.cameras)} camera definitions, {num_exported_cameras} camera nodes (Indices: {sorted(exported_cam_indices)}) "
                          f"and {num_exported_points} points (COLMAP IDs: {sorted(exported_point_indices)}) to GLTF.")
        print(export_summary)
        if num_exported_cameras == 0 and num_exported_points == 0:
            print("Warning: No valid camera nodes were exported.")
            print("Warning: No valid points were exported.")
            return False, "Export failed: No valid cameras or points could be processed."
        elif num_exported_cameras == 0:
            print("Warning: No valid camera nodes were exported.")
        elif num_exported_points == 0:
             print("Warning: No valid points were exported.")


        gltf.save(filename)
        return True, f"Scene successfully exported to {filename}.\n{export_summary}"

    except ImportError: return False, "Export failed: pygltflib not found."
    except FileNotFoundError: return False, f"Export failed: Directory for '{filename}' does not exist or insufficient permissions."
    except PermissionError: return False, f"Export failed: Permission denied writing to '{filename}'."
    except Exception as e: import traceback; print("--- GLTF Export Error ---"); traceback.print_exc(); print("--- End Traceback ---"); return False, f"Export failed: An unexpected error occurred: {e}" 