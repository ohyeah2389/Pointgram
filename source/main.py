import sys
import json
import os
from typing import Dict, Tuple, List, Optional, Any
import argparse

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QGraphicsLineItem,
)
from PySide6.QtGui import QPixmap, QIcon
from PySide6.QtCore import QPointF, Slot, QSize

from graphics_widgets import CrosshairMarker
from ui_panel import UiPanelMixin, resource_path
from point_interaction import PointInteractionMixin
from calibration import CalibrationMixin, PYCOLMAP_AVAILABLE
from export_scene import ExportSceneMixin

class MainWindow(
    QMainWindow, UiPanelMixin, PointInteractionMixin, CalibrationMixin, ExportSceneMixin
):
    """Main Pointgram application window."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Pointgram")
        self.setWindowIcon(QIcon(resource_path(os.path.join('icons', 'logo.ico'))))
        self.setGeometry(100, 100, 1400, 900)

        # Data storage
        self.image_paths: List[str] = []
        self.current_image_index: int = -1
        # { set_idx: {img_idx: (QPointF, marker_item), ... }, ... }
        self.point_data: Dict[int, Dict[int, Tuple[QPointF, CrosshairMarker]]] = {}
        self.point_set_names: Dict[int, str] = {}
        self.active_point_set_index: int = -1
        self._next_point_set_id: int = 0
        self.thumbnails: Dict[str, Optional[QPixmap]] = {}
        self.thumbnail_size: QSize = QSize(64, 64)
        self.current_save_path: Optional[str] = None
        # {img_idx: (width, height)}
        self.image_dimensions: Dict[int, Tuple[int, int]] = {}
        self.calibration_results: Optional[Dict[str, Any]] = (
            None  # Stores results from PyCOLMAP
        )
        # {set_id: {img_idx: {'dx': float, 'dy': float, 'magnitude': float}}}
        self.reprojection_errors: Dict[int, Dict[int, Dict[str, float]]] = {}
        # {img_idx: {set_id: keypoint_idx}} - Map for COLMAP keypoint ordering
        self.keypoint_maps: Dict[int, Dict[int, int]] = {}
        # Stores QGraphicsLineItem objects for error arrows currently in scene
        self.error_arrow_items: List[QGraphicsLineItem] = []

        # UI Setup
        self._pycolmap_available = PYCOLMAP_AVAILABLE
        self._setup_ui()
        self.view.set_tool_mode(tool="add_move")
        self._update_window_title()

        if not self._pycolmap_available:
            QMessageBox.critical(
                self,
                "Dependency Error",
                "PyCOLMAP library not found.\n"
                "Calibration functionality will be disabled.\n"
                "Please install it: pip install pycolmap",
            )
            # Consider disabling calibration UI elements here if needed

    def _update_window_title(self):
        """Updates the window title to include the current project file path."""
        base_title = "Pointgram"
        if self.current_save_path:
            filename = os.path.basename(self.current_save_path)
            self.setWindowTitle(f"{base_title} - [{filename}]")
        else:
            self.setWindowTitle(f"{base_title} - [Untitled]")

    # --- Save/Load Functionality ---
    @Slot()
    def save_points(self):
        """Saves the current point data to the existing file, or prompts for a new one."""
        if not self.current_save_path:
            self.save_points_as()
        else:
            self._write_points_to_file(self.current_save_path)

    @Slot()
    def save_points_as(self):
        """Prompts the user for a filename and saves the point data."""
        dialog = QFileDialog(self, "Save Point Data", ".", "JSON Files (*.json)")
        dialog.setAcceptMode(QFileDialog.AcceptSave)

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            if not filename.lower().endswith(".json"):
                filename += ".json"

            if self._write_points_to_file(filename):
                self.current_save_path = filename
                self._update_window_title()

    def _write_points_to_file(self, filename: str) -> bool:
        """Writes the point data, names, and image list to the specified JSON file."""
        if not self.point_data and not self.image_paths:
            self.statusBar().showMessage("Nothing to save.", 3000)
            return False

        data_to_save = {
            "image_paths": self.image_paths,
            "point_data": {},
            "point_set_names": {}, # Add names dictionary
            "image_dimensions": {},
        }

        # Convert QPointF to list [x, y] and keys to strings for JSON
        for set_id, observations in self.point_data.items():
            str_set_id = str(set_id)
            data_to_save["point_data"][str_set_id] = {}
            for img_idx, (point, marker) in observations.items():
                str_img_idx = str(img_idx)
                data_to_save["point_data"][str_set_id][str_img_idx] = [
                    point.x(),
                    point.y(),
                ]

        # Convert point set name keys to strings
        for set_id, name in self.point_set_names.items():
            str_set_id = str(set_id)
            data_to_save["point_set_names"][str_set_id] = name

        # Convert image dimension keys to strings
        for img_idx, dims in self.image_dimensions.items():
            str_img_idx = str(img_idx)
            data_to_save["image_dimensions"][str_img_idx] = [dims[0], dims[1]]

        try:
            with open(filename, "w") as f:
                json.dump(data_to_save, f, indent=4)
            self.statusBar().showMessage(f"Points saved to {filename}", 3000)
            return True
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save file:\n{e}")
            self.statusBar().showMessage(f"Error saving file: {e}", 5000)
            return False

    @Slot()
    def open_project(self):
        """Opens a file dialog to load a saved project file."""
        # TODO: Add check for unsaved changes before proceeding

        dialog = QFileDialog(self, "Open Project File", ".", "JSON Files (*.json)")
        dialog.setFileMode(QFileDialog.ExistingFile)

        if dialog.exec():
            filename = dialog.selectedFiles()[0]
            if self._load_data_from_file(filename):
                self.current_save_path = filename
                self._update_window_title()

                # Update UI Lists AFTER loading data
                self.update_image_list()
                self.update_point_set_list()

                # Display the first image
                if self.image_paths:
                    current_row = self.image_list_widget.currentRow()
                    if current_row == 0:
                        self.display_image(0)  # Force display if already selected
                    else:
                        self.image_list_widget.setCurrentRow(0)
                else:
                    self.clear_scene_and_pixmap()

    def _load_data_from_file(self, filename: str) -> bool:
        """Loads project data (images, points, names) from the specified JSON file."""
        try:
            with open(filename, "r") as f:
                loaded_data = json.load(f)
        except FileNotFoundError:
            QMessageBox.critical(self, "Load Error", f"File not found:\n{filename}")
            return False
        except json.JSONDecodeError as e:
            QMessageBox.critical(
                self, "Load Error", f"Error parsing JSON file:\n{filename}\n{e}"
            )
            return False
        except Exception as e:
            QMessageBox.critical(
                self, "Load Error", f"Could not read file:\n{filename}\n{e}"
            )
            return False

        # Basic Format Validation
        if (
            not isinstance(loaded_data, dict)
            or "image_paths" not in loaded_data
            or "point_data" not in loaded_data
            or not isinstance(loaded_data["image_paths"], list)
            or not isinstance(loaded_data["point_data"], dict)
        ):
            QMessageBox.critical(
                self,
                "Load Error",
                "Invalid project file format (missing image_paths or point_data).",
            )
            return False

        self._reset_state()  # Reset current state before loading

        self.image_paths = loaded_data["image_paths"]
        # Pre-load thumbnails
        for i, path in enumerate(self.image_paths):
            if not os.path.exists(path):
                # Don't warn for synthetic paths used in testing/examples
                if not path.startswith("synthetic_images/"):
                    QMessageBox.warning(
                        self,
                        "Load Warning",
                        f"Image path not found:\n{path}\nPoints associated with this image might not display correctly.",
                    )
            self.load_thumbnail(path)

        # Load Image Dimensions (optional field in JSON)
        loaded_dims = loaded_data.get("image_dimensions")
        if isinstance(loaded_dims, dict):
            print("Loading image dimensions from project file...")
            parsed_count = 0
            for img_idx_str, dims_list in loaded_dims.items():
                try:
                    img_idx = int(img_idx_str)
                    if isinstance(dims_list, list) and len(dims_list) == 2:
                        width, height = int(dims_list[0]), int(dims_list[1])
                        if width > 0 and height > 0:
                            self.image_dimensions[img_idx] = (width, height)
                            parsed_count += 1
                        else:
                            print(
                                f"Warning: Invalid dimensions [{width},{height}] for image index {img_idx}. Skipping."
                            )
                    else:
                        print(
                            f"Warning: Invalid dimension format '{dims_list}' for image index {img_idx}. Skipping."
                        )
                except (ValueError, TypeError) as e:
                    print(
                        f"Warning: Error parsing image dimension key '{img_idx_str}' or values: {e}. Skipping."
                    )
            print(f"Loaded dimensions for {parsed_count} images.")
        else:
            print(
                "No 'image_dimensions' found in project file or format incorrect. Will attempt to load from files."
            )

        # Load Point Data
        max_set_id = -1
        loaded_point_data = loaded_data["point_data"]
        for set_id_str, observations_dict in loaded_point_data.items():
            try:
                set_id = int(set_id_str)
                max_set_id = max(max_set_id, set_id)
                self.point_data[set_id] = {}
                if not isinstance(observations_dict, dict):
                    print(
                        f"Warning: Invalid format for observations in set {set_id}. Skipping."
                    )
                    continue

                for img_idx_str, coords_list in observations_dict.items():
                    try:
                        img_idx = int(img_idx_str)
                        if not (
                            isinstance(coords_list, list) and len(coords_list) == 2
                        ):
                            print(
                                f"Warning: Invalid coordinate format for point in set {set_id}, img {img_idx}. Skipping."
                            )
                            continue

                        point_coords = QPointF(coords_list[0], coords_list[1])
                        # Markers are created but NOT added to the scene here.
                        # display_image handles adding markers for the visible image.
                        marker = self.create_marker_item(point_coords, set_id)
                        self.point_data[set_id][img_idx] = (point_coords, marker)

                    except (ValueError, TypeError) as e:
                        print(
                            f"Warning: Error processing point data for set {set_id}, img_idx '{img_idx_str}': {e}. Skipping."
                        )
                        continue

            except (ValueError, TypeError) as e:
                print(f"Warning: Invalid set ID '{set_id_str}': {e}. Skipping.")
                continue

        # Load Point Set Names (optional field in JSON)
        loaded_names = loaded_data.get("point_set_names")
        if isinstance(loaded_names, dict):
            print("Loading point set names from project file...")
            parsed_count = 0
            for set_id_str, name in loaded_names.items():
                try:
                    set_id = int(set_id_str)
                    if isinstance(name, str) and name.strip():
                        self.point_set_names[set_id] = name.strip()
                        parsed_count += 1
                    else:
                        print(f"Warning: Invalid name '{name}' for set ID {set_id}. Skipping.")
                except (ValueError, TypeError) as e:
                    print(f"Warning: Error parsing point set name key '{set_id_str}' or value: {e}. Skipping.")
            print(f"Loaded {parsed_count} point set names.")
        else:
            print("No 'point_set_names' found in project file or format incorrect.")

        self._next_point_set_id = max_set_id + 1

        self.statusBar().showMessage(f"Project loaded from {filename}", 3000)
        return True

    def _reset_state(self):
        """Clears all current project data and resets UI elements."""
        self.clear_scene_and_pixmap()

        self.image_paths = []
        self.current_image_index = -1
        self.point_data = {}
        self.point_set_names = {}
        self.active_point_set_index = -1
        self._next_point_set_id = 0
        self.thumbnails = {}
        self.current_save_path = None
        self.image_dimensions = {}
        self.calibration_results = None
        self.reprojection_errors = {}
        self.keypoint_maps = {}

        self.image_list_widget.clear()
        self.point_set_list_widget.clear()
        self._update_window_title()
        self.current_image_index = -1

    # --- Helper to load dimensions (used by calibration) ---
    def _load_dimensions_for_image(self, index: int) -> Optional[Tuple[int, int]]:
        """Loads dimensions for a specific image index if not already cached.
        Checks cache first, then attempts to load from the file.
        """
        if index in self.image_dimensions:
            return self.image_dimensions[index]

        if 0 <= index < len(self.image_paths):
            path = self.image_paths[index]
            # Avoid trying to load synthetic placeholders
            if path.startswith("synthetic_images/"):
                print(
                    f"Warning: Cannot load dimensions for synthetic placeholder image {index}: {path}"
                )
                return None

            pixmap = QPixmap(path)
            if not pixmap.isNull():
                dims = (pixmap.width(), pixmap.height())
                self.image_dimensions[index] = dims  # Cache after loading
                return dims
            else:
                print(
                    f"Warning: Could not load pixmap to get dimensions for image {index} at path {path}"
                )
        return None

    @Slot()
    def run_calibration(self):
        return CalibrationMixin.run_calibration(self)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pointgram")
    parser.add_argument(
        "--project", type=str, help="Path to the .json project file to load."
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run calibration immediately after loading the project.",
    )
    parser.add_argument(
        "--export",
        type=str,
        help="Export calibration results to the specified GLTF file path.",
    )
    parser.add_argument(
        "--no-gui", action="store_true", help="Force command-line mode (no GUI)."
    )

    args = parser.parse_args()

    # Command-Line Execution Path
    if args.project or args.no_gui:
        print("Running in command-line mode...")
        app = QApplication.instance()
        if app is None:
            app = QApplication([])  # Needed for some non-GUI Qt ops

        window = MainWindow()  # Create instance but don't show

        if args.project:
            print(f"Loading project: {args.project}")
            if not window._load_data_from_file(args.project):
                print(f"ERROR: Failed to load project file '{args.project}'. Exiting.")
                sys.exit(1)
            window.current_save_path = args.project
            print("Project loaded successfully.")
        else:
            if not args.no_gui:  # --run or --export requires --project
                print(
                    "ERROR: --project must be specified for command-line processing unless only --no-gui is used."
                )
                sys.exit(1)

        if args.run:
            if not window.image_paths or not window.point_data:
                print(
                    "WARNING: Cannot run calibration - project not loaded or has no images/points. Skipping run."
                )
            else:
                print("Running calibration...")
                window.run_calibration()  # Handles its own status/error printing

        if args.export:
            print(f"Attempting to export results to: {args.export}")
            if not window.calibration_results:
                print("WARNING: No calibration results available to export. Skipping.")
            else:
                success, msg = window._do_export(filename=args.export)
                if success:
                    print(f"Export Successful: {msg}")
                else:
                    print(f"Export FAILED: {msg}")

        print("Command-line processing finished.")
        sys.exit(0)

    # GUI Execution Path
    else:
        print("Starting GUI mode...")
        app = QApplication(sys.argv)
        app.setApplicationName("Pointgram")
        app.setOrganizationName("Pointgram")
        app.setApplicationVersion("0.3")
        app.setStyle("Fusion")

        window = MainWindow()
        window.show()
        sys.exit(app.exec())
