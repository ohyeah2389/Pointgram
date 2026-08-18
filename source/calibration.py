import os
import tempfile
import shutil
from typing import Optional, Dict, Any

import numpy as np

from PySide6.QtWidgets import QApplication, QMessageBox

try:
    import pycolmap

    print("PyCOLMAP found.")
    PYCOLMAP_AVAILABLE = True
except ImportError:
    print("WARNING: PyCOLMAP not found. COLMAP integration will not work.")
    print("Please install it: pip install pycolmap")
    pycolmap = None
    PYCOLMAP_AVAILABLE = False


class CalibrationMixin:
    def _populate_colmap_database(self, db_path: str) -> bool:
        """
        Creates and populates the COLMAP database using PyCOLMAP bindings,
        including manually writing matches and two-view geometries based
        on point_data. Requires image dimensions to be pre-loaded/cached.
        """
        if not PYCOLMAP_AVAILABLE:
            print("Error: Cannot populate COLMAP database, PyCOLMAP not available.")
            return False

        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)

        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Database Error",
                    f"Could not remove existing database file:\n{db_path}\n{e}",
                )
                return False

        db = None
        try:
            db = pycolmap.Database.open(db_path)
            keypoints_per_image = {}
            self.keypoint_maps = {idx: {} for idx in range(len(self.image_paths))}
            sorted_set_ids = sorted(self.point_data.keys())

            for img_idx in range(len(self.image_paths)):
                keypoints_per_image[img_idx] = []
                current_keypoint_idx = 0
                for set_id in sorted_set_ids:
                    if img_idx in self.point_data[set_id]:
                        coords, _ = self.point_data[set_id][img_idx]
                        keypoints_per_image[img_idx].append(
                            (coords.x(), coords.y(), 1.0, 0.0)
                        )
                        self.keypoint_maps[img_idx][set_id] = current_keypoint_idx
                        current_keypoint_idx += 1

            with pycolmap.DatabaseTransaction(db):
                SIMPLE_PINHOLE_MODEL_ID = pycolmap.CameraModelId.SIMPLE_PINHOLE
                camera_ids = {}
                image_ids = {}

                for img_idx, img_path in enumerate(self.image_paths):
                    img_basename = os.path.basename(img_path)
                    width, height = self.image_dimensions.get(img_idx, (0, 0))
                    if width <= 0 or height <= 0:
                        raise ValueError(
                            f"Missing or invalid dimensions ({width}x{height}) for image {img_idx}: {img_basename}. Run dimension loading first."
                        )

                    focal_length = 1.2 * max(width, height)
                    cx = width / 2.0
                    cy = height / 2.0
                    params = np.array([focal_length, cx, cy], dtype=np.float64)
                    camera = pycolmap.Camera(
                        model=SIMPLE_PINHOLE_MODEL_ID,
                        width=width,
                        height=height,
                        params=params,
                    )
                    cam_id = db.write_camera(camera, use_camera_id=False)
                    camera_ids[img_idx] = cam_id

                    image = pycolmap.Image(name=img_basename, camera_id=cam_id)
                    img_id = db.write_image(image, use_image_id=False)
                    image_ids[img_idx] = img_id

                    keypoints = keypoints_per_image.get(img_idx, [])
                    num_keypoints = len(keypoints)
                    if num_keypoints > 0:
                        db.write_keypoints(
                            img_id, np.array(keypoints, dtype=np.float32)
                        )
                        db.write_descriptors(
                            img_id,
                            pycolmap.FeatureDescriptors(
                                type=pycolmap.FeatureExtractorType.SIFT,
                                data=np.zeros((num_keypoints, 128), dtype=np.uint8),
                            ),
                        )

                num_pairs_matched = 0
                img_indices = list(range(len(self.image_paths)))
                for i in range(len(img_indices)):
                    for j in range(i + 1, len(img_indices)):
                        idx1 = img_indices[i]
                        idx2 = img_indices[j]

                        image_id1 = image_ids[idx1]
                        image_id2 = image_ids[idx2]

                        common_sets = set(self.keypoint_maps[idx1].keys()) & set(
                            self.keypoint_maps[idx2].keys()
                        )

                        if common_sets:
                            matches_list = []
                            for set_id in common_sets:
                                kp_idx1 = self.keypoint_maps[idx1][set_id]
                                kp_idx2 = self.keypoint_maps[idx2][set_id]
                                matches_list.append((kp_idx1, kp_idx2))

                            if matches_list:
                                num_pairs_matched += 1
                                matches_array = np.array(matches_list, dtype=np.uint32)
                                db.write_matches(image_id1, image_id2, matches_array)

                                geometry = pycolmap.TwoViewGeometry()
                                geometry.config = (
                                    pycolmap.TwoViewGeometryConfiguration.CALIBRATED
                                )
                                geometry.inlier_matches = matches_array
                                db.write_two_view_geometry(
                                    image_id1, image_id2, geometry
                                )

            print(
                f"Manual matches/geometries written for {num_pairs_matched} image pairs."
            )
            return True

        except Exception as e:
            QMessageBox.critical(
                self,
                "Database Error",
                f"PyCOLMAP error creating/populating database/matches:\n{e}",
            )
            print(f"ERROR: PyCOLMAP database error: {e}")
            import traceback

            traceback.print_exc()
            return False
        finally:
            if db is not None:
                db.close()

    def run_calibration(self):
        """
        Prepares data and runs the COLMAP pipeline using PyCOLMAP bindings.
        Uses manually specified points/matches instead of feature extraction/matching.
        """
        if not PYCOLMAP_AVAILABLE:
            QMessageBox.critical(
                self,
                "PyCOLMAP Error",
                "PyCOLMAP library is not installed or could not be imported. Calibration is disabled.",
            )
            self.statusBar().showMessage(
                "Calibration failed: PyCOLMAP not available.", 5000
            )
            return

        self.statusBar().showMessage("Starting calibration with PyCOLMAP...")
        QApplication.processEvents()

        num_images = len(self.image_paths)
        num_point_sets = len(self.point_data)
        if num_images < 2:
            QMessageBox.warning(
                self, "Calibration Error", "Need at least 2 images loaded."
            )
            self.statusBar().showMessage(
                "Calibration failed: Need at least 2 images.", 5000
            )
            return
        if num_point_sets < 3:
            QMessageBox.warning(
                self,
                "Calibration Error",
                f"Need at least 3 point sets defined (found {num_point_sets}).",
            )
            self.statusBar().showMessage(
                "Calibration failed: Insufficient point sets.", 5000
            )
            return
        if not self.point_data:
            QMessageBox.warning(self, "Calibration Error", "No point data defined.")
            self.statusBar().showMessage("Calibration failed: No point data.", 5000)
            return

        self.calibration_results = None
        self.reprojection_errors = {}
        self.update_image_list()
        self.update_point_set_list()
        self.redraw_markers_and_errors_for_current_image()
        QApplication.processEvents()

        colmap_base_dir = tempfile.gettempdir()
        if self.current_save_path:
            project_dir = os.path.dirname(os.path.abspath(self.current_save_path))
            if os.path.isdir(project_dir):
                colmap_base_dir = project_dir
            else:
                print(
                    f"Warning: Project directory '{project_dir}' not found. Using system temp."
                )
        else:
            print("Warning: No project path set. Using system temp directory for COLMAP.")

        colmap_work_dir = os.path.join(colmap_base_dir, "colmap_py_work")
        if os.path.exists(colmap_work_dir):
            try:
                shutil.rmtree(colmap_work_dir)
            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Cleanup Error",
                    f"Could not remove old directory:\n{colmap_work_dir}\n{e}",
                )
                self.statusBar().showMessage("Calibration failed: Cleanup error.", 5000)
                return
        try:
            os.makedirs(colmap_work_dir)
        except OSError as e:
            QMessageBox.critical(
                self,
                "Directory Error",
                f"Could not create directory:\n{colmap_work_dir}\n{e}",
            )
            self.statusBar().showMessage(
                "Calibration failed: Directory creation error.", 5000
            )
            return

        database_path_abs = os.path.join(colmap_work_dir, "database.db")
        image_copy_dir_abs = os.path.join(colmap_work_dir, "images")
        sparse_output_path_abs = os.path.join(colmap_work_dir, "sparse")

        self.statusBar().showMessage("Verifying image dimensions...")
        QApplication.processEvents()
        for idx in range(len(self.image_paths)):
            if idx not in self.image_dimensions:
                if not self._load_dimensions_for_image(idx):
                    QMessageBox.critical(
                        self,
                        "Dimension Error",
                        f"Failed to load dimensions for image {idx}: {self.image_paths[idx]}. Cannot proceed.",
                    )
                    self.statusBar().showMessage(
                        "Calibration failed: Image dimension error.", 5000
                    )
                    shutil.rmtree(colmap_work_dir, ignore_errors=True)
                    return

        self.statusBar().showMessage("PyCOLMAP: Populating database & matches...")
        QApplication.processEvents()
        if not self._populate_colmap_database(database_path_abs):
            self.statusBar().showMessage(
                "Calibration failed: Error populating database/matches.", 5000
            )
            shutil.rmtree(colmap_work_dir, ignore_errors=True)
            return

        self.statusBar().showMessage("PyCOLMAP: Preparing image directory...")
        QApplication.processEvents()
        os.makedirs(image_copy_dir_abs, exist_ok=True)
        try:
            for orig_path in self.image_paths:
                basename = os.path.basename(orig_path)
                dest_path_abs = os.path.join(image_copy_dir_abs, basename)
                if not os.path.exists(dest_path_abs):
                    shutil.copy2(orig_path, dest_path_abs)
        except Exception as copy_e:
            QMessageBox.critical(
                self,
                "Image Copy Error",
                f"Failed to copy images for PyCOLMAP.\n{copy_e}",
            )
            shutil.rmtree(colmap_work_dir, ignore_errors=True)
            return

        self.statusBar().showMessage(
            "PyCOLMAP: Running incremental mapping... (This may take time)"
        )
        QApplication.processEvents()
        os.makedirs(sparse_output_path_abs, exist_ok=True)

        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.min_num_matches = 3
        pipeline_options.mapper.init_min_num_inliers = 3
        pipeline_options.mapper.init_min_tri_angle = 1.0
        pipeline_options.mapper.abs_pose_min_num_inliers = 3
        pipeline_options.mapper.abs_pose_max_error = 24.0
        pipeline_options.mapper.filter_min_tri_angle = 0.0

        try:
            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path_abs,
                image_path=image_copy_dir_abs,
                output_path=sparse_output_path_abs,
                options=pipeline_options,
            )
        except Exception as e:
            error_text = str(e)
            if (
                "init_min_num_inliers" in error_text
                or "options.Check()" in error_text
                or "no initial pair" in error_text.lower()
            ):
                friendly_message = (
                    "Calibration failed because PyCOLMAP could not find a valid initial "
                    "image pair. Add more matching points between overlapping images "
                    "so that each pair shares at least three correspondences, then try again."
                )
                status_message = "Calibration failed: Not enough matched points."
            else:
                friendly_message = f"Error during incremental mapping:\n{e}"
                status_message = "Calibration failed: PyCOLMAP mapping error."

            QMessageBox.critical(self, "PyCOLMAP Error", friendly_message)
            self.statusBar().showMessage(status_message, 5000)
            import traceback

            traceback.print_exc()
            return

        self.statusBar().showMessage("PyCOLMAP: Parsing results...")
        QApplication.processEvents()

        if reconstructions is None or not isinstance(reconstructions, dict):
            QMessageBox.warning(
                self,
                "Calibration Result",
                "PyCOLMAP mapping did not return a result dictionary.",
            )
            self.statusBar().showMessage(
                "Calibration finished: No reconstruction dictionary.", 5000
            )
            return

        if not reconstructions:
            QMessageBox.warning(
                self,
                "Calibration Result",
                "PyCOLMAP mapping finished, but did not produce any reconstruction models.",
            )
            self.statusBar().showMessage(
                "Calibration finished: No reconstruction models found.", 5000
            )
            return

        largest_rec_id = -1
        max_reg_images = -1
        largest_rec = None
        for rec_id, rec in reconstructions.items():
            num_reg = rec.num_reg_images()
            if num_reg > max_reg_images:
                max_reg_images = num_reg
                largest_rec_id = rec_id
                largest_rec = rec

        if largest_rec_id == -1 or largest_rec is None or max_reg_images < 2:
            QMessageBox.warning(
                self,
                "Calibration Result",
                f"PyCOLMAP mapping finished, but the largest model had only {max_reg_images} registered images. Need at least 2.",
            )
            self.statusBar().showMessage(
                "Calibration finished: Insufficient registered images.", 5000
            )
            return

        rec = largest_rec

        self.calibration_results = {
            "intrinsics": {},
            "poses": {},
            "points_3d": [],
            "point_ids": [],
            "registered_indices": [],
        }
        registered_image_ids_map = {img.image_id: img for img in rec.images.values()}

        original_idx_to_colmap_id: Dict[int, int] = {}
        colmap_id_to_original_idx: Dict[int, int] = {}
        db_read = None
        try:
            db_read = pycolmap.Database.open(database_path_abs)
            all_db_images = db_read.read_all_images()
            name_to_id_map = {img.name: img.image_id for img in all_db_images}
            for idx, img_path in enumerate(self.image_paths):
                basename = os.path.basename(img_path)
                if basename in name_to_id_map:
                    colmap_id = name_to_id_map[basename]
                    original_idx_to_colmap_id[idx] = colmap_id
                    colmap_id_to_original_idx[colmap_id] = idx
                else:
                    print(
                        f"Warning: Could not find image name {basename} in database during result parsing."
                    )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Result Parsing Error",
                f"Failed to read image names from database for mapping:\n{e}",
            )
            self.statusBar().showMessage(
                "Calibration finished with parsing errors.", 5000
            )
            return
        finally:
            if db_read is not None:
                db_read.close()

        for img_idx, colmap_image_id in original_idx_to_colmap_id.items():
            if colmap_image_id in registered_image_ids_map:
                image = registered_image_ids_map[colmap_image_id]
                camera = rec.cameras[image.camera_id]

                if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    if len(camera.params) == 3:
                        f, cx, cy = camera.params
                        K_matrix = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
                        self.calibration_results["intrinsics"][img_idx] = {
                            "K": K_matrix
                        }
                    else:
                        print(
                            f"Warning: Unexpected number of params for SIMPLE_PINHOLE camera {image.camera_id} (image {img_idx}). Expected 3, got {len(camera.params)}. Skipping intrinsics."
                        )
                        continue
                else:
                    print(
                        f"Warning: Skipping intrinsics for image {img_idx}. Unsupported camera model ID: {camera.model}"
                    )

                try:
                    pose_w2c = image.cam_from_world()
                    R_w2c = pose_w2c.rotation.matrix()
                    t_w2c = pose_w2c.translation

                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c

                    self.calibration_results["poses"][img_idx] = {
                        "R": R_c2w.tolist(),
                        "t": t_c2w.tolist(),
                    }
                    self.calibration_results["registered_indices"].append(img_idx)

                except AttributeError as e:
                    print(
                        f"  ERROR accessing pose attribute for img_idx {img_idx}: {e}"
                    )
                    print(f"    Attributes available: {dir(image)}")
                    print(f"  Skipping pose extraction for image {img_idx}.")
                    continue
                except Exception as e:
                    print(
                        f"  UNEXPECTED ERROR during pose extraction for img_idx {img_idx}: {e}"
                    )
                    import traceback

                    traceback.print_exc()
                    continue

        points_list = []
        point_ids_list = []
        keypoint_idx_to_set_id_map: Dict[int, Dict[int, int]] = {}
        if hasattr(self, "keypoint_maps") and self.keypoint_maps:
            for img_idx, set_map in self.keypoint_maps.items():
                keypoint_idx_to_set_id_map[img_idx] = {
                    v: k for k, v in set_map.items()
                }
        else:
            print(
                "WARNING: Cannot map Point3D IDs to Set IDs - self.keypoint_maps missing or empty."
            )

        colmap_point3D_id_to_set_id: Dict[int, int] = {}
        num_mapped_points = 0
        for p3d_id, point3d in rec.points3D.items():
            points_list.append(point3d.xyz.tolist())

            found_set_id = None
            if keypoint_idx_to_set_id_map:
                for track_el in point3d.track.elements:
                    img_id = track_el.image_id
                    kp_idx = track_el.point2D_idx

                    original_img_idx = colmap_id_to_original_idx.get(img_id)

                    if (
                        original_img_idx is not None
                        and original_img_idx in keypoint_idx_to_set_id_map
                    ):
                        set_id = keypoint_idx_to_set_id_map[original_img_idx].get(
                            kp_idx
                        )
                        if set_id is not None:
                            found_set_id = set_id
                            break

            if found_set_id is not None:
                colmap_point3D_id_to_set_id[p3d_id] = found_set_id
                point_ids_list.append(found_set_id)
                num_mapped_points += 1
            else:
                point_ids_list.append(None)

        self.calibration_results["points_3d"] = points_list
        self.calibration_results["point_ids"] = point_ids_list

        print(
            f"Mapped {num_mapped_points} COLMAP 3D points back to original Set IDs."
        )

        self.statusBar().showMessage("Calculating reprojection errors...")
        QApplication.processEvents()
        self.reprojection_errors = {}

        num_errors_calculated = 0
        for img_idx, colmap_image_id in original_idx_to_colmap_id.items():
            if colmap_image_id in registered_image_ids_map:
                image = registered_image_ids_map[colmap_image_id]
                camera = rec.cameras[image.camera_id]
                pose_w2c = image.cam_from_world()

                for point2D in image.points2D:
                    observed_xy = point2D.xy
                    p3d_id = point2D.point3D_id

                    if point2D.has_point3D and p3d_id in rec.points3D:
                        set_id = colmap_point3D_id_to_set_id.get(p3d_id)

                        if set_id is not None:
                            point3D = rec.points3D[p3d_id]
                            world_point = point3D.xyz

                            point_in_camera_coords = pose_w2c * world_point

                            if point_in_camera_coords[2] > 1e-6:
                                if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                                    try:
                                        f, cx, cy = camera.params
                                        X, Y, Z = point_in_camera_coords
                                        u = f * (X / Z) + cx
                                        v = f * (Y / Z) + cy
                                        projected_xy = np.array(
                                            [u, v], dtype=np.float64
                                        )

                                        error_vec = observed_xy - projected_xy
                                        dx, dy = error_vec[0], error_vec[1]
                                        magnitude = np.linalg.norm(error_vec)

                                        if set_id not in self.reprojection_errors:
                                            self.reprojection_errors[set_id] = {}
                                        self.reprojection_errors[set_id][img_idx] = {
                                            "dx": dx,
                                            "dy": dy,
                                            "magnitude": magnitude,
                                        }
                                        num_errors_calculated += 1
                                    except ZeroDivisionError:
                                        print(
                                            f"Warning: Skipping projection for point {p3d_id} (Set {set_id}) in image {img_idx} due to Z=0."
                                        )
                                    except Exception as proj_err:
                                        print(
                                            f"Warning: Error during manual projection for point {p3d_id} (Set {set_id}) in image {img_idx}: {proj_err}"
                                        )

                                else:
                                    print(
                                        f"Warning: Reprojection calculation skipped for unsupported camera model ID {camera.model} (img_idx {img_idx}, Set {set_id})"
                                    )

        print(
            f"Calculated {num_errors_calculated} reprojection errors for mapped points."
        )

        self.statusBar().showMessage("Updating UI with results...")
        QApplication.processEvents()
        self.update_image_list()
        self.update_point_set_list()
        self.redraw_markers_and_errors_for_current_image()

        QMessageBox.information(
            self,
            "Calibration Successful",
            f"PyCOLMAP processing complete.\n"
            f"Found {len(reconstructions)} model(s).\n"
            f"Processed largest model with {max_reg_images} registered images and {rec.num_points3D()} points.\n"
            f"Results stored and reprojection errors calculated for observed points.",
        )
        self.statusBar().showMessage("PyCOLMAP calibration successful.", 8000)
