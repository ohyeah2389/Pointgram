import os
from typing import Tuple

from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
from PySide6.QtCore import QStandardPaths

from gltf_exporter import export_scene_to_gltf, PYGLTFLIB_AVAILABLE


class ExportSceneMixin:
    def _do_export(self, filename: str) -> Tuple[bool, str]:
        """Performs the export logic using self.calibration_results and point set names."""
        results_to_export = self.calibration_results
        generator_name = "Pointgram (PyCOLMAP)"

        if not PYGLTFLIB_AVAILABLE:
            msg = "Export failed: pygltflib not found."
            self.statusBar().showMessage(msg, 5000)
            return False, msg
        if not results_to_export:
            msg = "Export failed: No calibration results available."
            self.statusBar().showMessage(msg, 3000)
            return False, msg

        required_keys = [
            "intrinsics",
            "poses",
            "points_3d",
            "point_ids",
            "registered_indices",
        ]
        if not isinstance(results_to_export, dict) or not all(
            k in results_to_export for k in required_keys
        ):
            missing = [k for k in required_keys if k not in results_to_export]
            msg = f"Export failed: Calibration results incomplete/invalid. Missing keys: {missing}"
            self.statusBar().showMessage(
                "Export failed: Invalid calibration results structure.", 3000
            )
            return False, msg
        if not isinstance(results_to_export["intrinsics"], dict):
            msg = "Export failed: Calibration 'intrinsics' data is not a dictionary."
            self.statusBar().showMessage(msg, 3000)
            return False, msg

        missing_dims = False
        indices_to_check = results_to_export.get("registered_indices", [])
        if not indices_to_check and results_to_export.get("poses"):
            indices_to_check = list(results_to_export["poses"].keys())

        for img_idx in indices_to_check:
            if img_idx not in self.image_dimensions:
                if not self._load_dimensions_for_image(img_idx):
                    msg = f"Cannot export: Image dimensions missing for registered image {img_idx} and could not be loaded."
                    self.statusBar().showMessage(msg, 5000)
                    missing_dims = True

        if missing_dims:
            return (
                False,
                "Export failed: Image dimensions missing for one or more registered images.",
            )

        if not filename.lower().endswith(".gltf"):
            filename += ".gltf"

        self.statusBar().showMessage(f"Exporting scene to {filename}...")
        QApplication.processEvents()

        success, message = export_scene_to_gltf(
            filename=filename,
            results=results_to_export,
            image_paths=self.image_paths,
            image_dimensions=self.image_dimensions,
            point_set_names=self.point_set_names,
            generator_name=generator_name,
        )

        if success:
            self.statusBar().showMessage(message, 8000)
        else:
            self.statusBar().showMessage(f"Export failed: {message}", 8000)

        return success, message

    def export_scene_as(self):
        """Exports the calibrated scene (cameras, points) from PyCOLMAP to a GLTF file."""
        if not self.calibration_results:
            QMessageBox.warning(
                self,
                "Export Error",
                "No calibration results available. Run calibration first.",
            )
            return

        suggested_name = "scene_colmap.gltf"
        if self.current_save_path:
            base = os.path.splitext(os.path.basename(self.current_save_path))[0]
            suggested_name = f"{base}_scene_colmap.gltf"

        default_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.DocumentsLocation
        )
        if self.current_save_path:
            default_dir = os.path.dirname(self.current_save_path)

        dialog = QFileDialog(
            self, "Export Scene As GLTF", default_dir, "GLTF Files (*.gltf)"
        )
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.selectFile(suggested_name)

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            success, message = self._do_export(filename)
            if success:
                QMessageBox.information(self, "Export Successful", message)
            else:
                QMessageBox.critical(self, "Export Error", message)
