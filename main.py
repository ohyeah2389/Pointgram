import sys
import json
import os
import numpy as np
from typing import Union, Dict, Tuple, List, Optional, Any
import argparse
import tempfile
import shutil

try:
    import pycolmap

    print("PyCOLMAP found.")
    _pycolmap_available = True
except ImportError:
    print("WARNING: PyCOLMAP not found. COLMAP integration will not work.")
    print("Please install it: pip install pycolmap")
    pycolmap = None
    _pycolmap_available = False

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QGraphicsScene,
    QGraphicsPixmapItem,
    QWidget,
    QListWidget,
    QListWidgetItem,
    QHBoxLayout,
    QSplitter,
    QToolBar,
    QMessageBox,
    QGraphicsItem,
    QGraphicsLineItem,
    QLabel,
    QInputDialog,
)
from PySide6.QtGui import (
    QPixmap,
    QAction,
    QKeySequence,
    QPen,
    QColor,
    QActionGroup,
    QIcon,
)
from PySide6.QtCore import Qt, Slot, QPointF, QRectF, QSize, QStandardPaths

from graphics_widgets import ZoomableView, CrosshairMarker
from gltf_exporter import export_scene_to_gltf, PYGLTFLIB_AVAILABLE

# Reprojection error visualization options
ERROR_ARROW_SCALE = 5.0  # Visual scaling factor for error vectors
ERROR_ARROW_COLOR = QColor(Qt.GlobalColor.cyan)
ERROR_ARROW_WIDTH = (
    0  # Cosmetic pen width ensures constant thickness regardless of zoom
)

# Error color gradient stops (magnitude, color)
ERROR_COLOR_STOPS = [
    (0.0, QColor(Qt.GlobalColor.blue)),
    (1.0, QColor(Qt.GlobalColor.green)),
    (3.0, QColor(Qt.GlobalColor.yellow)),
    (6.0, QColor(Qt.GlobalColor.red)),
    (10.0, QColor(Qt.GlobalColor.magenta)),
]

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        # For one-folder builds, this is the folder containing the executable
        base_path = sys._MEIPASS
    except Exception:
        # Not running bundled, use the script's directory
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


def interpolate_color(magnitude: float, stops: List[Tuple[float, QColor]]) -> QColor:
    """Linearly interpolates color based on magnitude between defined stops."""
    if magnitude <= stops[0][0]:
        return stops[0][1]
    if magnitude >= stops[-1][0]:
        return stops[-1][1]

    for i in range(len(stops) - 1):
        mag1, color1 = stops[i]
        mag2, color2 = stops[i + 1]
        if mag1 <= magnitude < mag2:
            factor = (magnitude - mag1) / (mag2 - mag1)
            r = int(color1.red() + (color2.red() - color1.red()) * factor)
            g = int(color1.green() + (color2.green() - color1.green()) * factor)
            b = int(color1.blue() + (color2.blue() - color1.blue()) * factor)
            return QColor(r, g, b)
    # Fallback for magnitudes exactly matching the last stop or exceeding it
    return stops[-1][1]


class MainWindow(QMainWindow):
    """Main application window for manual point calibration."""

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
        self._setup_ui()
        self.view.set_tool_mode(tool="add_move")
        self._update_window_title()

        if not _pycolmap_available:
            QMessageBox.critical(
                self,
                "Dependency Error",
                "PyCOLMAP library not found.\n"
                "Calibration functionality will be disabled.\n"
                "Please install it: pip install pycolmap",
            )
            # Consider disabling calibration UI elements here if needed

    def _setup_ui(self):
        """Creates and arranges the UI widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        # Left: Image List
        self.image_list_widget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(
            self.on_image_selection_changed
        )
        self.image_list_widget.setIconSize(self.thumbnail_size)
        main_splitter.addWidget(self.image_list_widget)

        # Center: Graphics View
        self.scene = QGraphicsScene()
        self.view = ZoomableView(self.scene)
        self.view.scene_mouse_press.connect(self.handle_scene_click)
        self.view.marker_action_click.connect(self.handle_marker_action)
        self.view.marker_move_finished.connect(self.finalize_marker_move)
        main_splitter.addWidget(self.view)

        # Right: Point Set List
        self.point_set_list_widget = QListWidget()
        self.point_set_list_widget.currentItemChanged.connect(
            self.on_point_set_selection_changed
        )
        self.point_set_list_widget.itemDoubleClicked.connect(
            self.on_point_set_double_clicked
        )
        main_splitter.addWidget(self.point_set_list_widget)

        main_splitter.setSizes([250, 950, 200])

        # Toolbar
        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))  # Larger icons
        self.addToolBar(toolbar)
        self._setup_toolbar(toolbar)

        self.statusBar()
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None  # Current image pixmap

        self._update_window_title()

    def _setup_toolbar(self, toolbar: QToolBar):
        """Adds actions to the toolbar.
        Tries local icons first, then falls back to system theme.
        """

        def get_icon(icon_filename: str, theme_name: str) -> QIcon:
            """Helper to load local icon first, then theme."""
            local_icon_path = resource_path(os.path.join("icons", icon_filename))

            if os.path.exists(local_icon_path):
                return QIcon(local_icon_path)
            else:
                return QIcon.fromTheme(theme_name)

        open_action = QAction(get_icon("go-down.svg", "go-down"), "", self)
        open_action.setShortcut(QKeySequence.Open)
        open_action.setToolTip("Import Images... (Ctrl+O)")
        open_action.triggered.connect(self.open_images)
        toolbar.addAction(open_action)

        open_project_action = QAction(
            get_icon("document-open.svg", "document-open"), "", self
        )
        open_project_action.setToolTip("Open Project...")
        open_project_action.triggered.connect(self.open_project)
        toolbar.addAction(open_project_action)

        save_action = QAction(
            get_icon("document-save.svg", "document-save"), "", self
        )
        save_action.setShortcut(QKeySequence.Save)
        save_action.setToolTip("Save Project (Ctrl+S)")
        save_action.triggered.connect(self.save_points)
        toolbar.addAction(save_action)

        save_as_action = QAction(
            get_icon("document-save-as.svg", "document-save-as"), "", self
        )
        save_as_action.setShortcut(QKeySequence.SaveAs)
        save_as_action.setToolTip("Save Project As... (Ctrl+Shift+S)")
        save_as_action.triggered.connect(self.save_points_as)
        toolbar.addAction(save_as_action)

        toolbar.addSeparator()

        # Export Operations
        export_scene_action = QAction(
            get_icon("document-print.svg", "document-print"), "", self
        )
        export_scene_action.setToolTip("Export Scene As GLTF...")
        export_scene_action.triggered.connect(self.export_scene_as)
        toolbar.addAction(export_scene_action)

        toolbar.addSeparator()

        # Tools
        self.add_point_tool_action = QAction(
            get_icon("list-add.svg", "list-add"), "", self
        )
        self.add_point_tool_action.setCheckable(True)
        self.add_point_tool_action.setChecked(True)
        self.add_point_tool_action.setToolTip("Add/Move Point Tool")
        self.add_point_tool_action.triggered.connect(self.activate_add_point_tool)
        toolbar.addAction(self.add_point_tool_action)

        self.delete_point_tool_action = QAction(
            get_icon("edit-cut.svg", "edit-cut"), "", self
        )
        self.delete_point_tool_action.setCheckable(True)
        self.delete_point_tool_action.setToolTip("Delete Point Tool")
        self.delete_point_tool_action.triggered.connect(self.activate_delete_point_tool)
        toolbar.addAction(self.delete_point_tool_action)

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        self.tool_action_group.addAction(self.add_point_tool_action)
        self.tool_action_group.addAction(self.delete_point_tool_action)

        toolbar.addSeparator()

        # Calibration
        calibrate_action = QAction(get_icon("measure.svg", "accessories-engineering"), "", self)
        calibrate_action.setToolTip("Run Calibration (SfM + Bundle Adjustment)")
        calibrate_action.triggered.connect(self.run_calibration)
        if not _pycolmap_available:
            calibrate_action.setEnabled(False)
            calibrate_action.setToolTip("Calibration disabled (PyCOLMAP not found)")
        toolbar.addAction(calibrate_action)

    def _update_window_title(self):
        """Updates the window title to include the current project file path."""
        base_title = "Pointgram"
        if self.current_save_path:
            filename = os.path.basename(self.current_save_path)
            self.setWindowTitle(f"{base_title} - [{filename}]")
        else:
            self.setWindowTitle(f"{base_title} - [Untitled]")

    # --- Tool Activation Slots ---
    @Slot()
    def activate_add_point_tool(self):
        """Activates the add/move point tool."""
        self.view.set_tool_mode(tool="add_move")
        self.statusBar().showMessage("Add/Move Point Tool Activated", 2000)

    @Slot()
    def activate_delete_point_tool(self):
        """Activates the delete point tool."""
        self.view.set_tool_mode(tool="delete")
        self.statusBar().showMessage("Delete Point Tool Activated", 2000)

    # --- Image Handling ---
    @Slot()
    def open_images(self):
        """Opens a file dialog to select and add images to the current project."""
        # Consider warning if a project is already loaded?
        file_dialog = QFileDialog(self)
        file_dialog.setNameFilter("Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        file_dialog.setFileMode(QFileDialog.ExistingFiles)

        if file_dialog.exec():
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                new_paths = [p for p in selected_files if p not in self.image_paths]
                if not new_paths:
                    self.statusBar().showMessage(
                        "Selected images already loaded.", 3000
                    )
                    return

                start_index = len(self.image_paths)
                self.image_paths.extend(new_paths)

                # TODO: Load thumbnails asynchronously?
                for path in new_paths:
                    self.load_thumbnail(path)

                self.update_image_list()

                # Select first newly added image if nothing was selected before
                if self.current_image_index == -1 or new_paths:
                    self.image_list_widget.setCurrentRow(start_index)

    def load_thumbnail(self, image_path):
        """Loads and caches a thumbnail for the given image path."""
        if image_path in self.thumbnails:
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.thumbnails[image_path] = None
            print(f"Warning: Could not load thumbnail for {image_path}")
            return

        scaled_pixmap = pixmap.scaled(
            self.thumbnail_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.thumbnails[image_path] = scaled_pixmap

    def update_image_list(self):
        """Refreshes the image list widget with thumbnails and status indicators."""
        selected_row = self.image_list_widget.currentRow()
        self.image_list_widget.clear()

        calibrated_indices = set()
        if self.calibration_results and "poses" in self.calibration_results:
            calibrated_indices = set(self.calibration_results["poses"].keys())

        active_set_indices = set()
        if (
            self.active_point_set_index != -1
            and self.active_point_set_index in self.point_data
        ):
            active_set_indices = set(
                self.point_data[self.active_point_set_index].keys()
            )

        for i, path in enumerate(self.image_paths):
            filename = os.path.basename(path)
            has_point_in_active_set = i in active_set_indices
            is_calibrated = i in calibrated_indices
            has_any_point = any(i in obs for obs in self.point_data.values())

            line1 = f"{i}: {filename}"

            active_symbol = (
                "⚪" if has_point_in_active_set else "⚫"
            )  # Point in active set?
            if is_calibrated:
                calib_symbol = "✅"  # Calibrated
            elif has_any_point:
                calib_symbol = "❔"  # Has points, but not calibrated
            else:
                calib_symbol = "➖"  # No points

            line2 = f"{active_symbol} {calib_symbol}"
            item_text = f"{line1}\n{line2}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)  # Store original index
            item.setToolTip(path)

            thumbnail = self.thumbnails.get(path)
            if thumbnail:
                item.setIcon(QIcon(thumbnail))

            self.image_list_widget.addItem(item)

        if 0 <= selected_row < self.image_list_widget.count():
            self.image_list_widget.setCurrentRow(selected_row)
        elif self.image_paths:
            self.image_list_widget.setCurrentRow(0)

    @Slot(QListWidgetItem, QListWidgetItem)
    def on_image_selection_changed(
        self, current_item: QListWidgetItem, previous_item: QListWidgetItem
    ):
        """Displays the selected image."""
        if current_item:
            index = current_item.data(Qt.UserRole)
            if 0 <= index < len(self.image_paths):
                if index != self.current_image_index:
                    self.display_image(index)

    def display_image(self, index: int):
        """Loads and displays the image, managing scene items and error arrows."""
        if not (0 <= index < len(self.image_paths)):
            print(f"Warning: Invalid image index {index} requested.")
            self.clear_scene_and_pixmap()
            return

        previous_transform = self.view.transform() if self.pixmap_item else None

        image_path = self.image_paths[index]
        pixmap = QPixmap(image_path)

        self.clear_scene_and_pixmap()  # Clears pixmap, markers, AND error arrows

        if pixmap.isNull():
            self.statusBar().showMessage(
                f"Error: Could not load image {image_path}", 5000
            )
            return

        self.image_dimensions[index] = (pixmap.width(), pixmap.height())
        self.current_image_index = index

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.pixmap_item.setZValue(-1)  # Ensure image is behind markers/arrows

        self.redraw_markers_and_errors_for_current_image()

        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        if previous_transform is None:
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        else:
            self.view.setTransform(previous_transform)  # Restore view state

        self.statusBar().showMessage(f"Displayed: {image_path}", 3000)

    def clear_scene_and_pixmap(self):
        """Removes current pixmap, all markers, and all error arrows from the scene."""
        if self.pixmap_item and self.pixmap_item.scene() == self.scene:
            self.scene.removeItem(self.pixmap_item)
        self.pixmap_item = None

        items_to_remove = [
            item for item in self.scene.items() if isinstance(item, CrosshairMarker)
        ]
        for item in items_to_remove:
            if item.scene() == self.scene:
                self.scene.removeItem(item)

        for arrow_item in self.error_arrow_items:
            if arrow_item.scene() == self.scene:
                self.scene.removeItem(arrow_item)
        self.error_arrow_items.clear()

        self.scene.setSceneRect(QRectF(self.view.rect()))

    # --- Point Handling ---
    @Slot(QGraphicsItem, QPointF)
    def handle_marker_action(self, marker_item: QGraphicsItem, scene_pos: QPointF):
        """Handles actions on markers based on the active tool."""
        if not isinstance(marker_item, CrosshairMarker):
            return

        if self.delete_point_tool_action.isChecked():
            self.delete_point_observation(marker_item)

    @Slot(QPointF)
    def handle_scene_click(self, scene_pos: QPointF):
        """Handles placing a point based on the active tool and context."""
        # Triggered on Mouse RELEASE by ZoomableView (after potential drag-to-zoom)

        if not self.add_point_tool_action.isChecked():
            return
        if self.current_image_index < 0 or self.pixmap_item is None:
            self.statusBar().showMessage("Load an image first!", 3000)
            return

        if not self.pixmap_item.boundingRect().contains(scene_pos):
            self.statusBar().showMessage("Placement outside image bounds.", 2000)
            return

        image_index = self.current_image_index
        point_coords = scene_pos

        if self.active_point_set_index == -1:
            # No active set, create a new one
            self.create_new_point_set(image_index, point_coords)
        else:
            active_set_points = self.point_data.get(self.active_point_set_index, {})
            if image_index in active_set_points:
                # If clicking on an image that *already* has a point in the *active* set,
                # assume the user wants a *new* point set.
                self.create_new_point_set(image_index, point_coords)
            else:
                # Add point to the currently active set
                self.add_point_to_set(
                    self.active_point_set_index, image_index, point_coords
                )

        self.update_point_set_list()
        self.update_image_list()  # Update checkmarks etc.

    @Slot(QGraphicsItem, QPointF)
    def finalize_marker_move(self, marker_item: QGraphicsItem, final_pos: QPointF):
        """Updates the point data after a marker drag is finished."""
        if not isinstance(marker_item, CrosshairMarker):
            return

        set_index = marker_item.data(Qt.UserRole)
        target_img_idx = -1

        if set_index in self.point_data:
            for img_idx, (coords, mk) in self.point_data[set_index].items():
                if mk == marker_item:
                    target_img_idx = img_idx
                    break

        if target_img_idx != -1:
            self.point_data[set_index][target_img_idx] = (final_pos, marker_item)
            self.statusBar().showMessage(f"Moved point in Set {set_index}", 1500)
        else:
            print(f"Error: Could not find moved marker in data structure.")

    def create_new_point_set(self, image_index: int, point_coords: QPointF):
        """Creates a new point set containing the first point."""
        new_set_id = self._next_point_set_id
        self._next_point_set_id += 1
        marker = self.create_marker_item(point_coords, new_set_id)
        self.point_data[new_set_id] = {image_index: (point_coords, marker)}
        self.active_point_set_index = new_set_id  # New set becomes active
        self.scene.addItem(marker)
        self.style_marker(marker, new_set_id)  # Style based on active state
        self.statusBar().showMessage(f"Created Point Set {new_set_id}", 2000)

    def add_point_to_set(self, set_index: int, image_index: int, point_coords: QPointF):
        """Adds a point observation to an existing point set."""
        if set_index not in self.point_data:
            print(f"Error: Trying to add point to non-existent set {set_index}")
            return
        if image_index in self.point_data[set_index]:
            print(
                f"Error: Image {image_index} already has a point in set {set_index}. Logic error?"
            )
            return

        marker = self.create_marker_item(point_coords, set_index)
        self.point_data[set_index][image_index] = (point_coords, marker)
        self.scene.addItem(marker)
        self.style_marker(marker, set_index)
        self.statusBar().showMessage(f"Added point to Set {set_index}", 2000)

    def delete_point_observation(self, marker: CrosshairMarker):
        """Deletes a single point observation (marker), potentially the whole set if empty."""
        set_index = marker.data(Qt.UserRole)
        target_img_idx = -1

        if set_index in self.point_data:
            for img_idx, (coords, mk) in self.point_data[set_index].items():
                if mk == marker:
                    target_img_idx = img_idx
                    break

        if target_img_idx == -1:
            print(f"Error: Could not find marker in data structure for deletion.")
            return

        reply = QMessageBox.question(
            self,
            "Delete Point",
            f"Delete point for image {target_img_idx} from Point Set {set_index}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.point_data[set_index][target_img_idx]
            if marker.scene() == self.scene:
                self.scene.removeItem(marker)

            self.statusBar().showMessage(
                f"Deleted point ({target_img_idx}) from Point Set {set_index}", 2000
            )

            if not self.point_data[set_index]:  # Check if the set is now empty
                del self.point_data[set_index]
                self.statusBar().showMessage(
                    f"Deleted empty Point Set {set_index}", 2000
                )
                if self.active_point_set_index == set_index:
                    self.active_point_set_index = -1

            self.update_point_set_list()
            self.update_image_list()
            if self.current_image_index == target_img_idx:
                self.redraw_markers_and_errors_for_current_image()

    def create_marker_item(self, position: QPointF, set_index: int) -> CrosshairMarker:
        """Factory method to create a CrosshairMarker instance."""
        set_name = self.point_set_names.get(set_index, str(set_index))
        marker = CrosshairMarker(position, set_index, set_name)
        return marker

    def style_marker(self, marker: CrosshairMarker, set_index: int):
        """Applies styling (color, label) to a marker based on active state and name."""
        if not isinstance(marker, CrosshairMarker):
            return

        is_active = set_index == self.active_point_set_index
        pen_color = (
            QColor(Qt.GlobalColor.yellow) if is_active else QColor(Qt.GlobalColor.red)
        )

        marker.set_style(color=pen_color, width=1.0, cosmetic=True)
        marker.setZValue(1)  # Ensure markers are above image

        # Update marker text based on stored name or index
        set_name = self.point_set_names.get(set_index, str(set_index))
        marker.set_text(set_name)

        if hasattr(marker, "text_label") and marker.text_label:
            marker.text_label.setVisible(True)

    def update_point_set_list(self):
        """Refreshes the point set list, showing image indices and reprojection errors."""
        self.point_set_list_widget.blockSignals(True)
        self.point_set_list_widget.clear()

        sorted_set_ids = sorted(self.point_data.keys())
        row_to_select = -1
        default_text_color = self.palette().color(self.point_set_list_widget.foregroundRole()).name()

        for i, set_id in enumerate(sorted_set_ids):
            point_set_observations = self.point_data[set_id]
            custom_name = self.point_set_names.get(set_id)
            display_name = custom_name if custom_name else f"Point Set {set_id}"

            # Use HTML for multi-line and color formatting
            html_lines = [
                # Display format: "ID: Name"
                f"<font color='{default_text_color}'><b>{set_id}: {display_name}</b></font>"
            ]

            # Add error information lines
            for img_idx in sorted(point_set_observations.keys()):
                error_info = self.reprojection_errors.get(set_id, {}).get(img_idx)
                error_str = ""
                error_color_hex = "gray"

                if error_info and "magnitude" in error_info:
                    error_mag = error_info["magnitude"]
                    if error_mag is not None:
                        error_color = interpolate_color(error_mag, ERROR_COLOR_STOPS)
                        error_color_hex = error_color.name()
                        error_str = f"<font color='{error_color_hex}'>({error_mag:.1f}px)</font>"
                    else:
                        error_str = "<font color='gray'>(Err?)</font>"
                else:
                    error_str = "<font color='gray'>(---)</font>"

                html_lines.append(
                    f"&nbsp;&nbsp;{error_str} - <font color='{default_text_color}'>{img_idx}</font>"
                )

            full_html = "<br>".join(html_lines)
            label_widget = QLabel(full_html)
            label_widget.setTextFormat(Qt.RichText)
            label_widget.setWordWrap(True)
            # Ensure background is transparent for selection highlight to show
            label_widget.setAutoFillBackground(False)
            label_widget.setStyleSheet("background-color: transparent;")


            item = QListWidgetItem()
            item.setData(Qt.UserRole, set_id)
            item.setSizeHint(label_widget.sizeHint())
            self.point_set_list_widget.addItem(item)
            self.point_set_list_widget.setItemWidget(item, label_widget)

            if set_id == self.active_point_set_index:
                row_to_select = i

        if row_to_select != -1:
            self.point_set_list_widget.setCurrentRow(row_to_select)
        else:
            self.point_set_list_widget.setCurrentRow(-1) # Deselect if active set removed

        self.point_set_list_widget.blockSignals(False)

    @Slot(QListWidgetItem)
    def on_point_set_double_clicked(self, item: QListWidgetItem):
        """Handles double-click on a point set item to initiate renaming."""
        if not item:
            return

        set_id = item.data(Qt.UserRole)
        if set_id is None:
            return

        current_name = self.point_set_names.get(set_id, "")
        default_display_name = f"Point Set {set_id}" # Used if current_name is empty

        # Use QInputDialog to get the new name
        new_name, ok = QInputDialog.getText(
            self,
            f"Rename Point Set {set_id}",
            "Enter new name:",
            text=current_name if current_name else "", # Show current name in input field
        )

        if ok:
            new_name = new_name.strip()
            if new_name:
                # Update the name in our dictionary
                self.point_set_names[set_id] = new_name
                self.statusBar().showMessage(f"Renamed Point Set {set_id} to '{new_name}'", 3000)
            else:
                # If the user entered an empty name, remove the custom name
                if set_id in self.point_set_names:
                    del self.point_set_names[set_id]
                    self.statusBar().showMessage(f"Removed custom name for Point Set {set_id}", 3000)
                else:
                    # No change needed if it didn't have a custom name and user entered empty
                    return

            # Refresh the list to show the new name
            self.update_point_set_list()
            # Update markers currently visible
            self.redraw_markers_and_errors_for_current_image()

    @Slot(QListWidgetItem, QListWidgetItem)
    def on_point_set_selection_changed(
        self, current_item: QListWidgetItem, previous_item: QListWidgetItem
    ):
        """Updates the active point set index and related UI elements."""
        new_active_index = current_item.data(Qt.UserRole) if current_item else -1

        if new_active_index != self.active_point_set_index:
            self.active_point_set_index = new_active_index
            status = (
                f"Active Point Set: {self.active_point_set_index}"
                if self.active_point_set_index != -1
                else "No active point set"
            )
            self.statusBar().showMessage(status, 2000)
            self.redraw_markers_and_errors_for_current_image()
            self.update_image_list()

    def redraw_markers_and_errors_for_current_image(self):
        """Adds/Updates markers and their reprojection error arrows for the current image."""
        if self.current_image_index < 0:
            return

        # Clear Existing Error Arrows
        for arrow_item in self.error_arrow_items:
            if arrow_item.scene() == self.scene:
                self.scene.removeItem(arrow_item)
        self.error_arrow_items.clear()

        # Add/Style Markers for current image
        markers_for_current_img = self.get_markers_for_image(self.current_image_index)
        for marker in markers_for_current_img:
            if marker.scene() != self.scene:
                self.scene.addItem(marker)
            set_idx = marker.data(Qt.UserRole)
            if set_idx is not None:
                self.style_marker(marker, set_idx)

                # Draw Error Arrow if data exists
                error_info = self.reprojection_errors.get(set_idx, {}).get(
                    self.current_image_index
                )
                if error_info and "dx" in error_info and "dy" in error_info:
                    dx = error_info["dx"]
                    dy = error_info["dy"]
                    if (
                        dx is not None and dy is not None
                    ):  # Ensure error values are valid
                        start_pos = marker.pos()
                        # Error vector points *from* projected *towards* observed.
                        # Draw from marker (observed) pos towards (marker_pos + error * scale)
                        end_pos = start_pos + QPointF(
                            dx * ERROR_ARROW_SCALE, dy * ERROR_ARROW_SCALE
                        )

                        arrow_line = QGraphicsLineItem(
                            start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y()
                        )
                        pen = QPen(ERROR_ARROW_COLOR, ERROR_ARROW_WIDTH)
                        pen.setCosmetic(True)
                        arrow_line.setPen(pen)
                        arrow_line.setZValue(
                            0
                        )  # Below markers (Z=1) but above image (Z=-1)

                        self.scene.addItem(arrow_line)
                        self.error_arrow_items.append(
                            arrow_line
                        )  # Track for later clearing
            else:
                print(f"Warning: Marker found without set_index data.")

    def get_markers_for_image(self, image_index: int) -> list[CrosshairMarker]:
        """Retrieves all marker items associated with a specific image index."""
        markers = []
        for set_idx, observations in self.point_data.items():
            if image_index in observations:
                coords, marker = observations[image_index]
                markers.append(marker)
        return markers

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

    def _populate_colmap_database(self, db_path: str) -> bool:
        """
        Creates and populates the COLMAP database using PyCOLMAP bindings,
        including manually writing matches and two-view geometries based
        on point_data. Requires image dimensions to be pre-loaded/cached.

        Args:
            db_path: Absolute path to the COLMAP database file.

        Returns:
            True if successful, False otherwise.
        """
        if not _pycolmap_available:
            print("Error: Cannot populate COLMAP database, PyCOLMAP not available.")
            return False

        print(f"Creating/Populating COLMAP database using PyCOLMAP: {db_path}")

        db_dir = os.path.dirname(db_path)
        os.makedirs(db_dir, exist_ok=True)

        if os.path.exists(db_path):
            try:
                os.remove(db_path)
                print(f"Removed existing database: {db_path}")
            except OSError as e:
                QMessageBox.critical(
                    self,
                    "Database Error",
                    f"Could not remove existing database file:\n{db_path}\n{e}",
                )
                return False

        db = None
        try:
            db = pycolmap.Database(db_path)
            print(f"Database object created: {db}")

            # Determine Keypoint Order and Prepare Data Structure for COLMAP
            keypoints_per_image = {}
            self.keypoint_maps = {
                idx: {} for idx in range(len(self.image_paths))
            }  # Reset and cache map
            sorted_set_ids = sorted(self.point_data.keys())

            for img_idx in range(len(self.image_paths)):
                keypoints_per_image[img_idx] = []
                current_keypoint_idx = 0
                for set_id in sorted_set_ids:
                    if img_idx in self.point_data[set_id]:
                        coords, _ = self.point_data[set_id][img_idx]
                        # COLMAP expects (x, y, scale, orientation)
                        keypoints_per_image[img_idx].append(
                            (coords.x(), coords.y(), 1.0, 0.0)
                        )
                        # Store the mapping: image -> set -> keypoint index in that image's list
                        self.keypoint_maps[img_idx][set_id] = current_keypoint_idx
                        current_keypoint_idx += 1
            print("Keypoint data and maps prepared and stored in self.keypoint_maps.")

            # Use Transaction for atomic database writes
            with pycolmap.DatabaseTransaction(db):
                SIMPLE_PINHOLE_MODEL_ID = pycolmap.CameraModelId.SIMPLE_PINHOLE
                camera_ids = {}  # Map app img_idx to COLMAP cam_id
                image_ids = {}  # Map app img_idx to COLMAP image_id

                # Write Cameras, Images, Keypoints, Descriptors
                for img_idx, img_path in enumerate(self.image_paths):
                    img_basename = os.path.basename(img_path)
                    width, height = self.image_dimensions.get(img_idx, (0, 0))
                    if width <= 0 or height <= 0:
                        raise ValueError(
                            f"Missing or invalid dimensions ({width}x{height}) for image {img_idx}: {img_basename}. Run dimension loading first."
                        )

                    # Create Camera (Simple Pinhole with estimated focal length)
                    # TODO: Allow user to specify camera intrinsics?
                    focal_length = 1.2 * max(width, height)  # Simple heuristic
                    cx = width / 2.0
                    cy = height / 2.0
                    params = np.array([focal_length, cx, cy], dtype=np.float64)
                    camera = pycolmap.Camera(
                        model=SIMPLE_PINHOLE_MODEL_ID,
                        width=width,
                        height=height,
                        params=params,
                    )
                    # Let pycolmap assign the ID
                    cam_id = db.write_camera(camera, use_camera_id=False)
                    camera_ids[img_idx] = cam_id
                    print(f"  Written camera {cam_id} for image {img_idx}")

                    image = pycolmap.Image(name=img_basename, camera_id=cam_id)
                    # Let pycolmap assign the ID
                    img_id = db.write_image(image, use_image_id=False)
                    image_ids[img_idx] = img_id
                    print(f"  Written image {img_id} for image {img_idx}")

                    # Write Keypoints and dummy Descriptors
                    keypoints = keypoints_per_image.get(img_idx, [])
                    num_keypoints = len(keypoints)
                    if num_keypoints > 0:
                        keypoints_array = np.array(keypoints, dtype=np.float32)
                        # Create dummy descriptors as COLMAP requires them
                        descriptors_array = np.zeros(
                            (num_keypoints, 128), dtype=np.uint8
                        )
                        print(
                            f"  Writing {num_keypoints} keypoints/descriptors for image {img_id}..."
                        )
                        db.write_keypoints(img_id, keypoints_array)
                        db.write_descriptors(img_id, descriptors_array)
                        print(f"    Written keypoints/descriptors.")

                print("Cameras, Images, Keypoints, Descriptors written.")

                # Manually Write Matches and Two-View Geometries based on shared point sets
                print("Writing manual matches and geometries...")
                num_pairs_matched = 0
                img_indices = list(range(len(self.image_paths)))
                for i in range(len(img_indices)):
                    for j in range(i + 1, len(img_indices)):
                        idx1 = img_indices[i]
                        idx2 = img_indices[j]

                        image_id1 = image_ids[idx1]
                        image_id2 = image_ids[idx2]

                        # Find common set_ids observed in both images using the cached map
                        common_sets = set(self.keypoint_maps[idx1].keys()) & set(
                            self.keypoint_maps[idx2].keys()
                        )

                        if common_sets:
                            matches_list = []
                            for set_id in common_sets:
                                # Get the keypoint index corresponding to this set_id in each image
                                kp_idx1 = self.keypoint_maps[idx1][set_id]
                                kp_idx2 = self.keypoint_maps[idx2][set_id]
                                matches_list.append((kp_idx1, kp_idx2))

                            if matches_list:
                                num_pairs_matched += 1
                                matches_array = np.array(matches_list, dtype=np.uint32)
                                db.write_matches(image_id1, image_id2, matches_array)

                                # Write a basic TwoViewGeometry entry
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
            # Transaction automatically commits here if no exceptions

            print("PyCOLMAP database population and manual matching finished.")
            return True

        except Exception as e:
            # Transaction automatically rolls back on exception
            QMessageBox.critical(
                self,
                "Database Error",
                f"PyCOLMAP error creating/populating database/matches:\n{e}",
            )
            print(f"ERROR: PyCOLMAP database error: {e}")
            import traceback

            traceback.print_exc()
            return False

    @Slot()
    def run_calibration(self):
        """
        Prepares data and runs the COLMAP pipeline using PyCOLMAP bindings.
        Uses manually specified points/matches instead of feature extraction/matching.
        """
        if not _pycolmap_available:
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
        QApplication.processEvents()  # Update UI

        # Basic sanity checks
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
        if num_point_sets < 3:  # COLMAP needs >= 3 points for robust triangulation/pose
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

        # Reset previous results and UI state related to calibration
        self.calibration_results = None
        self.reprojection_errors = {}
        self.update_image_list()
        self.update_point_set_list()
        self.redraw_markers_and_errors_for_current_image()
        QApplication.processEvents()

        # Setup temporary working directory for COLMAP files
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
            print(
                "Warning: No project path set. Using system temp directory for COLMAP."
            )

        colmap_work_dir = os.path.join(colmap_base_dir, "colmap_py_work")
        # Clean up previous run's directory if it exists
        if os.path.exists(colmap_work_dir):
            print(f"Removing existing PyCOLMAP working directory: {colmap_work_dir}")
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
            print(f"Created PyCOLMAP working directory: {colmap_work_dir}")
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
        image_copy_dir_abs = os.path.join(
            colmap_work_dir, "images"
        )  # COLMAP often expects images in a subdir
        sparse_output_path_abs = os.path.join(colmap_work_dir, "sparse")

        # Verify all required image dimensions are loaded
        self.statusBar().showMessage("Verifying image dimensions...")
        QApplication.processEvents()
        all_dims_loaded = True
        for idx in range(len(self.image_paths)):
            if idx not in self.image_dimensions:
                if not self._load_dimensions_for_image(idx):  # Try loading if missing
                    QMessageBox.critical(
                        self,
                        "Dimension Error",
                        f"Failed to load dimensions for image {idx}: {self.image_paths[idx]}. Cannot proceed.",
                    )
                    self.statusBar().showMessage(
                        "Calibration failed: Image dimension error.", 5000
                    )
                    shutil.rmtree(
                        colmap_work_dir, ignore_errors=True
                    )  # Clean up temp dir
                    return
        print("Image dimensions verified.")

        # 1. Create and Populate COLMAP Database
        self.statusBar().showMessage("PyCOLMAP: Populating database & matches...")
        QApplication.processEvents()
        if not self._populate_colmap_database(database_path_abs):
            self.statusBar().showMessage(
                "Calibration failed: Error populating database/matches.", 5000
            )
            shutil.rmtree(colmap_work_dir, ignore_errors=True)
            return

        # 2. Copy Images to COLMAP working directory
        self.statusBar().showMessage("PyCOLMAP: Preparing image directory...")
        QApplication.processEvents()
        os.makedirs(image_copy_dir_abs, exist_ok=True)
        try:
            for idx, orig_path in enumerate(self.image_paths):
                # Use basename to avoid path conflicts
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

        # 3. Skip Feature Extraction & Matching (using manual points/matches)
        print("Skipping feature extraction and matching (using manual points).")

        # 4. Run Incremental Mapping (SfM)
        self.statusBar().showMessage(
            "PyCOLMAP: Running incremental mapping... (This may take time)"
        )
        QApplication.processEvents()
        os.makedirs(sparse_output_path_abs, exist_ok=True)

        # Configure pipeline options - adjust thresholds for sparse manual data
        pipeline_options = pycolmap.IncrementalPipelineOptions()
        pipeline_options.min_num_matches = 3
        pipeline_options.mapper.init_min_num_inliers = 3
        pipeline_options.mapper.init_min_tri_angle = 1.0
        pipeline_options.mapper.abs_pose_min_num_inliers = 3
        pipeline_options.mapper.abs_pose_max_error = 24.0  # Higher tolerance
        pipeline_options.mapper.filter_min_tri_angle = (
            0.0  # Keep points even with poor geometry
        )

        reconstructions = None
        try:
            print("Starting pycolmap.incremental_mapping...")
            print(
                f"Using IncrementalPipelineOptions: min_num_matches={pipeline_options.min_num_matches}"
            )
            print(f"Using IncrementalMapperOptions:")
            print(
                f"  init_min_num_inliers={pipeline_options.mapper.init_min_num_inliers}"
            )
            print(f"  init_min_tri_angle={pipeline_options.mapper.init_min_tri_angle}")
            print(
                f"  abs_pose_min_num_inliers={pipeline_options.mapper.abs_pose_min_num_inliers}"
            )
            print(f"  abs_pose_max_error={pipeline_options.mapper.abs_pose_max_error}")
            print(
                f"  filter_min_tri_angle={pipeline_options.mapper.filter_min_tri_angle}"
            )

            reconstructions = pycolmap.incremental_mapping(
                database_path=database_path_abs,
                image_path=image_copy_dir_abs,  # Point to the copied images
                output_path=sparse_output_path_abs,
                options=pipeline_options,
            )
            print("pycolmap.incremental_mapping finished.")
        except Exception as e:
            QMessageBox.critical(
                self, "PyCOLMAP Error", f"Error during incremental mapping:\n{e}"
            )
            self.statusBar().showMessage(
                "Calibration failed: PyCOLMAP mapping error.", 5000
            )
            import traceback

            traceback.print_exc()
            return

        # 5. Parse PyCOLMAP Output
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

        print(f"PyCOLMAP produced {len(reconstructions)} reconstruction model(s).")

        # Find the largest reconstruction model
        largest_rec_id = -1
        max_reg_images = -1
        largest_rec = None
        for rec_id, rec in reconstructions.items():
            num_reg = rec.num_reg_images()
            print(
                f"  Model {rec_id}: {num_reg} registered images, {rec.num_points3D()} points."
            )
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

        print(
            f"Processing largest reconstruction (ID: {largest_rec_id}) with {max_reg_images} registered images."
        )
        rec = largest_rec

        # Extract Data into internal format
        self.calibration_results = {
            "intrinsics": {},
            "poses": {},
            "points_3d": [],
            "point_ids": [],
            "registered_indices": [],
        }
        registered_image_ids_map = {img.image_id: img for img in rec.images.values()}
        print(f"Total points in reconstruction: {rec.num_points3D()}")

        # Map original app image index to COLMAP image_id
        original_idx_to_colmap_id = {}
        colmap_id_to_original_idx = {}  # Reverse mapping
        try:
            # Open DB read-only this time
            db_read = pycolmap.Database(database_path_abs)
            all_db_images = db_read.read_all_images()
            # Map image filename to COLMAP image_id
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

        # Extract Intrinsics and Poses for registered images
        for img_idx, colmap_image_id in original_idx_to_colmap_id.items():
            if colmap_image_id in registered_image_ids_map:
                image = registered_image_ids_map[colmap_image_id]  # COLMAP Image object
                camera = rec.cameras[image.camera_id]  # COLMAP Camera object

                print(
                    f"Parsing camera for img_idx {img_idx}: colmap_img_id={colmap_image_id}, colmap_cam_id={image.camera_id}"
                )
                print(f"  Camera Model ID: {camera.model}, Params: {camera.params}")

                # Extract Intrinsics (only handle SIMPLE_PINHOLE)
                if camera.model == pycolmap.CameraModelId.SIMPLE_PINHOLE:
                    if len(camera.params) == 3:
                        f, cx, cy = camera.params
                        K_matrix = [[f, 0, cx], [0, f, cy], [0, 0, 1]]
                        self.calibration_results["intrinsics"][img_idx] = {
                            "K": K_matrix
                        }
                        print(f"  Stored intrinsics for img_idx {img_idx}")
                    else:
                        print(
                            f"Warning: Unexpected number of params for SIMPLE_PINHOLE camera {image.camera_id} (image {img_idx}). Expected 3, got {len(camera.params)}. Skipping intrinsics."
                        )
                        continue
                else:
                    print(
                        f"Warning: Skipping intrinsics for image {img_idx}. Unsupported camera model ID: {camera.model}"
                    )

                # Extract Poses (Cam-to-World for export)
                try:
                    pose_w2c = (
                        image.cam_from_world
                    )  # pycolmap.Rigid3d object (World-to-Camera)
                    R_w2c = pose_w2c.rotation.matrix()
                    t_w2c = pose_w2c.translation

                    # Invert to get Camera-to-World
                    R_c2w = R_w2c.T
                    t_c2w = -R_c2w @ t_w2c

                    self.calibration_results["poses"][img_idx] = {
                        "R": R_c2w.tolist(),
                        "t": t_c2w.tolist(),
                    }
                    self.calibration_results["registered_indices"].append(img_idx)
                    print(f"  Stored pose for img_idx {img_idx}")

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

        # Extract 3D Points and map back to original Set IDs where possible
        points_list = []  # XYZ coordinates
        point_ids_list = []  # Corresponding original Set IDs (or None)
        # Need the inverse map: {img_idx: {keypoint_idx: set_id}}
        keypoint_idx_to_set_id_map = {}
        if hasattr(self, "keypoint_maps") and self.keypoint_maps:
            for img_idx, set_map in self.keypoint_maps.items():
                keypoint_idx_to_set_id_map[img_idx] = {v: k for k, v in set_map.items()}
        else:
            print(
                "WARNING: Cannot map Point3D IDs to Set IDs - self.keypoint_maps missing or empty."
            )

        colmap_point3D_id_to_set_id = {}  # Cache mapping: colmap p3d_id -> app set_id
        num_mapped_points = 0
        for (
            p3d_id,
            point3d,
        ) in rec.points3D.items():  # Iterate through COLMAP's 3D points
            points_list.append(point3d.xyz.tolist())

            # Try to find the corresponding app set_id using the point's track
            found_set_id = None
            if keypoint_idx_to_set_id_map:
                # Iterate through observations (track elements) of this 3D point
                for track_el in point3d.track.elements:
                    img_id = track_el.image_id  # COLMAP image_id
                    kp_idx = track_el.point2D_idx  # Index in the keypoints list

                    original_img_idx = colmap_id_to_original_idx.get(
                        img_id
                    )  # Map back to app img_idx

                    if (
                        original_img_idx is not None
                        and original_img_idx in keypoint_idx_to_set_id_map
                    ):
                        # Look up set_id using the keypoint index in that image's map
                        set_id = keypoint_idx_to_set_id_map[original_img_idx].get(
                            kp_idx
                        )
                        if set_id is not None:
                            found_set_id = set_id
                            break  # Found the set_id

            if found_set_id is not None:
                colmap_point3D_id_to_set_id[p3d_id] = found_set_id
                point_ids_list.append(found_set_id)
                num_mapped_points += 1
            else:
                point_ids_list.append(None)

        self.calibration_results["points_3d"] = points_list
        self.calibration_results["point_ids"] = (
            point_ids_list  # Corresponds index-wise to points_3d
        )

        print(f"Mapped {num_mapped_points} COLMAP 3D points back to original Set IDs.")
        print(
            f"Extracted {len(self.calibration_results['intrinsics'])} intrinsics sets."
        )
        print(f"Extracted {len(self.calibration_results['poses'])} poses.")
        print(f"Extracted {len(self.calibration_results['points_3d'])} 3D points.")
        print(
            f"Registered image indices (original): {sorted(self.calibration_results['registered_indices'])}"
        )

        # Calculate Reprojection Errors
        self.statusBar().showMessage("Calculating reprojection errors...")
        QApplication.processEvents()
        self.reprojection_errors = (
            {}
        )  # Reset errors: {set_id: {img_idx: {'dx', 'dy', 'magnitude'}}}

        num_errors_calculated = 0
        # Iterate through registered images in the reconstruction
        for img_idx, colmap_image_id in original_idx_to_colmap_id.items():
            if colmap_image_id in registered_image_ids_map:
                image = registered_image_ids_map[colmap_image_id]  # COLMAP Image
                camera = rec.cameras[image.camera_id]  # COLMAP Camera
                pose_w2c = image.cam_from_world  # World-to-Camera Pose (Rigid3d)

                # Iterate through the 2D observations associated with this image
                for point2D in image.points2D:
                    observed_xy = point2D.xy  # Observed keypoint location (u, v)
                    p3d_id = point2D.point3D_id  # ID of the 3D point

                    # Check if this observation is linked to a valid 3D point
                    if p3d_id != -1 and p3d_id in rec.points3D:
                        # Get the corresponding Set ID we mapped earlier
                        set_id = colmap_point3D_id_to_set_id.get(p3d_id)

                        # Only calculate error if this point corresponds to one of our manual sets
                        if set_id is not None:
                            point3D = rec.points3D[p3d_id]  # COLMAP Point3D object
                            world_point = point3D.xyz  # 3D point in world coordinates

                            # 1. Transform world point to camera coordinates
                            point_in_camera_coords = pose_w2c * world_point

                            # 2. Project point from camera coordinates to image plane
                            if (
                                point_in_camera_coords[2] > 1e-6
                            ):  # Check if point is in front
                                if (
                                    camera.model
                                    == pycolmap.CameraModelId.SIMPLE_PINHOLE
                                ):
                                    try:
                                        f, cx, cy = camera.params
                                        X, Y, Z = point_in_camera_coords
                                        u = f * (X / Z) + cx
                                        v = f * (Y / Z) + cy
                                        projected_xy = np.array(
                                            [u, v], dtype=np.float64
                                        )

                                        # Calculate error: (observed - projected)
                                        error_vec = observed_xy - projected_xy
                                        dx, dy = error_vec[0], error_vec[1]
                                        magnitude = np.linalg.norm(error_vec)

                                        # Store error
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

        # Final UI Updates after successful calibration
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

    # --- Export Functions ---
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

        # Validate structure of calibration_results
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
            print(msg)
            self.statusBar().showMessage(
                "Export failed: Invalid calibration results structure.", 3000
            )
            return False, msg
        if not isinstance(results_to_export["intrinsics"], dict):
            msg = "Export failed: Calibration 'intrinsics' data is not a dictionary."
            self.statusBar().showMessage(msg, 3000)
            return False, msg

        # Check image dimensions are available for registered images
        missing_dims = False
        indices_to_check = results_to_export.get("registered_indices", [])
        if not indices_to_check and results_to_export.get("poses"):  # Fallback check
            indices_to_check = list(results_to_export["poses"].keys())

        for img_idx in indices_to_check:
            if img_idx not in self.image_dimensions:
                print(
                    f"Attempting lazy load of dimensions for image {img_idx} during export."
                )
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

        # Call the exporter function
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

    @Slot()
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
