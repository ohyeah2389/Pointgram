import os
from typing import Dict, Tuple, List

from PySide6.QtCore import Qt, Slot, QPointF, QRectF
from PySide6.QtGui import QPixmap, QPen, QColor, QIcon
from PySide6.QtWidgets import (
    QListWidgetItem,
    QMessageBox,
    QGraphicsItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QLabel,
    QInputDialog,
    QFileDialog,
)

from graphics_widgets import CrosshairMarker

# Reprojection error visualization options
ERROR_ARROW_SCALE = 5.0
ERROR_ARROW_COLOR = QColor(Qt.GlobalColor.cyan)
ERROR_ARROW_WIDTH = 0

# Error color gradient stops (magnitude, color)
ERROR_COLOR_STOPS = [
    (0.0, QColor(Qt.GlobalColor.blue)),
    (1.0, QColor(Qt.GlobalColor.green)),
    (3.0, QColor(Qt.GlobalColor.yellow)),
    (6.0, QColor(Qt.GlobalColor.red)),
    (10.0, QColor(Qt.GlobalColor.magenta)),
]


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
    return stops[-1][1]


class PointInteractionMixin:
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

                for path in new_paths:
                    self.load_thumbnail(path)

                self.update_image_list()

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

            active_symbol = "⚪" if has_point_in_active_set else "⚫"
            if is_calibrated:
                calib_symbol = "✅"
            elif has_any_point:
                calib_symbol = "❔"
            else:
                calib_symbol = "➖"

            line2 = f"{active_symbol} {calib_symbol}"
            item_text = f"{line1}\n{line2}"

            item = QListWidgetItem(item_text)
            item.setData(Qt.UserRole, i)
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

        self.clear_scene_and_pixmap()

        if pixmap.isNull():
            self.statusBar().showMessage(
                f"Error: Could not load image {image_path}", 5000
            )
            return

        self.image_dimensions[index] = (pixmap.width(), pixmap.height())
        self.current_image_index = index

        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        self.pixmap_item.setZValue(-1)

        self.redraw_markers_and_errors_for_current_image()

        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        if previous_transform is None:
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        else:
            self.view.setTransform(previous_transform)

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
            self.create_new_point_set(image_index, point_coords)
        else:
            active_set_points = self.point_data.get(self.active_point_set_index, {})
            if image_index in active_set_points:
                self.create_new_point_set(image_index, point_coords)
            else:
                self.add_point_to_set(
                    self.active_point_set_index, image_index, point_coords
                )

        self.update_point_set_list()
        self.update_image_list()

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
            print("Error: Could not find moved marker in data structure.")

    def create_new_point_set(self, image_index: int, point_coords: QPointF):
        """Creates a new point set containing the first point."""
        new_set_id = self._next_point_set_id
        self._next_point_set_id += 1
        marker = self.create_marker_item(point_coords, new_set_id)
        self.point_data[new_set_id] = {image_index: (point_coords, marker)}
        self.active_point_set_index = new_set_id
        self.scene.addItem(marker)
        self.style_marker(marker, new_set_id)
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
            print("Error: Could not find marker in data structure for deletion.")
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

            if not self.point_data[set_index]:
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
        return CrosshairMarker(position, set_index, set_name)

    def style_marker(self, marker: CrosshairMarker, set_index: int):
        """Applies styling (color, label) to a marker based on active state and name."""
        if not isinstance(marker, CrosshairMarker):
            return

        is_active = set_index == self.active_point_set_index
        pen_color = (
            QColor(Qt.GlobalColor.yellow) if is_active else QColor(Qt.GlobalColor.red)
        )

        marker.set_style(color=pen_color, width=1.0, cosmetic=True)
        marker.setZValue(1)

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
        default_text_color = self.palette().color(
            self.point_set_list_widget.foregroundRole()
        ).name()

        for i, set_id in enumerate(sorted_set_ids):
            point_set_observations = self.point_data[set_id]
            custom_name = self.point_set_names.get(set_id)
            display_name = custom_name if custom_name else f"Point Set {set_id}"

            html_lines = [
                f"<font color='{default_text_color}'><b>{set_id}: {display_name}</b></font>"
            ]

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
            self.point_set_list_widget.setCurrentRow(-1)

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

        new_name, ok = QInputDialog.getText(
            self,
            f"Rename Point Set {set_id}",
            "Enter new name:",
            text=current_name if current_name else "",
        )

        if ok:
            new_name = new_name.strip()
            if new_name:
                self.point_set_names[set_id] = new_name
                self.statusBar().showMessage(
                    f"Renamed Point Set {set_id} to '{new_name}'", 3000
                )
            else:
                if set_id in self.point_set_names:
                    del self.point_set_names[set_id]
                    self.statusBar().showMessage(
                        f"Removed custom name for Point Set {set_id}", 3000
                    )
                else:
                    return

            self.update_point_set_list()
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

        for arrow_item in self.error_arrow_items:
            if arrow_item.scene() == self.scene:
                self.scene.removeItem(arrow_item)
        self.error_arrow_items.clear()

        markers_for_current_img = self.get_markers_for_image(self.current_image_index)
        for marker in markers_for_current_img:
            if marker.scene() != self.scene:
                self.scene.addItem(marker)
            set_idx = marker.data(Qt.UserRole)
            if set_idx is not None:
                self.style_marker(marker, set_idx)

                error_info = self.reprojection_errors.get(set_idx, {}).get(
                    self.current_image_index
                )
                if error_info and "dx" in error_info and "dy" in error_info:
                    dx = error_info["dx"]
                    dy = error_info["dy"]
                    if dx is not None and dy is not None:
                        start_pos = marker.pos()
                        end_pos = start_pos + QPointF(
                            dx * ERROR_ARROW_SCALE, dy * ERROR_ARROW_SCALE
                        )

                        arrow_line = QGraphicsLineItem(
                            start_pos.x(), start_pos.y(), end_pos.x(), end_pos.y()
                        )
                        pen = QPen(ERROR_ARROW_COLOR, ERROR_ARROW_WIDTH)
                        pen.setCosmetic(True)
                        arrow_line.setPen(pen)
                        arrow_line.setZValue(0)

                        self.scene.addItem(arrow_line)
                        self.error_arrow_items.append(arrow_line)
            else:
                print("Warning: Marker found without set_index data.")

    def get_markers_for_image(self, image_index: int) -> list[CrosshairMarker]:
        """Retrieves all marker items associated with a specific image index."""
        markers = []
        for set_idx, observations in self.point_data.items():
            if image_index in observations:
                coords, marker = observations[image_index]
                markers.append(marker)
        return markers
