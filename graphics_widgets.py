from PySide6.QtWidgets import (
    QGraphicsView,
    QGraphicsPathItem,
    QGraphicsSimpleTextItem,
    QGraphicsEllipseItem,
    QGraphicsItem,
)
from PySide6.QtGui import (
    QPainterPath,
    QPen,
    QBrush,
    QPainter,
    QWheelEvent,
    QMouseEvent,
    QTransform,
)
from PySide6.QtCore import Qt, Signal, QPointF
from typing import Union

# --- Constants ---
MARKER_SIZE = 10
TEXT_OFFSET = 5
TARGET_PRECISION_ZOOM_FACTOR = 4.0
PRECISION_ZOOM_THRESHOLD = 0.01


# --- Custom Graphics Item for Crosshair Marker ---
class CrosshairMarker(QGraphicsPathItem):
    """A graphics item representing a point marker with a crosshair, circle, and text label."""

    def __init__(self, position: QPointF, set_index: int, size: float = MARKER_SIZE):
        super().__init__()
        self.setPos(position)
        self.set_index = set_index
        self.size = size
        self._update_path()

        self.setData(Qt.UserRole, set_index)
        self.setFlag(QGraphicsItem.ItemIsSelectable, False)
        # Crosshair scales with zoom
        self.setFlag(QGraphicsItem.ItemIgnoresTransformations, False)

        # Text Label Child Item
        self.text_label = QGraphicsSimpleTextItem(str(set_index), self)
        # Keep text size constant regardless of zoom
        self.text_label.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        font = self.text_label.font()
        font.setPointSize(8)
        self.text_label.setFont(font)
        self.text_label.setPos(TEXT_OFFSET, -TEXT_OFFSET)

        # Circle Child Item
        radius = self.size / 2.0
        self.circle = QGraphicsEllipseItem(-radius, -radius, self.size, self.size, self)
        # Keep circle line width constant regardless of zoom
        self.circle.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
        self.circle.setBrush(Qt.NoBrush)

        self.set_style()

    def _update_path(self):
        """Sets the QPainterPath for the crosshair."""
        path = QPainterPath()
        half_size = self.size / 2.0
        path.moveTo(-half_size, 0)
        path.lineTo(half_size, 0)
        path.moveTo(0, -half_size)
        path.lineTo(0, half_size)
        self.setPath(path)

    def set_style(self, color=Qt.red, width=1.0, cosmetic=True):
        """Applies color and pen style to the marker and its children."""
        pen = QPen(color, width)
        pen.setCosmetic(cosmetic)
        self.setPen(pen)
        self.text_label.setBrush(QBrush(color))
        circle_pen = QPen(color, width * 0.8)
        circle_pen.setCosmetic(True)
        self.circle.setPen(circle_pen)

    def type(self) -> int:
        """Custom type identifier for easily finding these items."""
        # UserType must be >= 65536
        return QGraphicsItem.UserType + 1


# --- Custom Graphics View ---
class ZoomableView(QGraphicsView):
    """A QGraphicsView subclass that supports zooming, middle-mouse panning,
    marker interaction (move/delete via tool), and placement."""

    # Signal emitted on MOUSE RELEASE after placing a new point
    scene_mouse_press = Signal(QPointF)
    # Signal emitted on LEFT-CLICK on a marker when delete tool is active
    marker_action_click = Signal(QGraphicsItem, QPointF)  # Renamed signal
    # Signal emitted when a marker drag operation finishes
    marker_move_finished = Signal(QGraphicsItem, QPointF)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)

        self.setDragMode(
            QGraphicsView.NoDrag
        )  # Panning handled by middle mouse override
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setMouseTracking(True)

        self._current_tool_mode = "add_move"

        self._is_middle_button_panning = False
        self._last_pan_pos = QPointF()
        self._dragged_marker: Union[QGraphicsItem, None] = None
        self._is_placing_new_marker = False
        self._original_transform_before_action: Union[QTransform, None] = None

        self.set_cursor_for_mode()

    def wheelEvent(self, event: QWheelEvent):
        """Zooms the view towards the mouse cursor."""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor
        old_pos = self.mapToScene(event.position().toPoint())
        zoom_factor = zoom_in_factor if event.angleDelta().y() > 0 else zoom_out_factor
        self.scale(zoom_factor, zoom_factor)
        new_pos = self.mapToScene(event.position().toPoint())
        delta = new_pos - old_pos
        self.translate(delta.x(), delta.y())

    def mousePressEvent(self, event: QMouseEvent):
        """Handles mouse presses for panning, marker interaction (drag start/delete),
        and starting the placement of a new point."""
        view_pos = event.pos()
        scene_pos = self.mapToScene(view_pos)

        # Middle Button Pan Override
        if event.button() == Qt.MiddleButton:
            self._is_middle_button_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return

        # Check for clicks on markers or their children
        item = self.itemAt(view_pos)
        marker_item = None
        if item:
            if item.type() == CrosshairMarker.UserType + 1:
                marker_item = item
            elif isinstance(item.parentItem(), CrosshairMarker):
                marker_item = item.parentItem()

        if marker_item:
            # Handle Left Click on Marker
            if self._current_tool_mode == "add_move":
                # Start Marker Drag with temporary precision zoom
                self._original_transform_before_action = self.transform()
                current_scale = self._original_transform_before_action.m11()
                if (
                    current_scale
                    < TARGET_PRECISION_ZOOM_FACTOR - PRECISION_ZOOM_THRESHOLD
                ):
                    zoom_factor = TARGET_PRECISION_ZOOM_FACTOR / current_scale
                    self.scale(zoom_factor, zoom_factor)
                self._dragged_marker = marker_item
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
            elif self._current_tool_mode == "delete":
                self.marker_action_click.emit(marker_item, scene_pos)
                event.accept()
                return

        else:
            # Handle Left Click on Background
            if self._current_tool_mode == "add_move":
                # Start Placing New Point with temporary precision zoom
                self._original_transform_before_action = self.transform()
                current_scale = self._original_transform_before_action.m11()
                if (
                    current_scale
                    < TARGET_PRECISION_ZOOM_FACTOR - PRECISION_ZOOM_THRESHOLD
                ):
                    zoom_factor = TARGET_PRECISION_ZOOM_FACTOR / current_scale
                    self.scale(zoom_factor, zoom_factor)
                self._is_placing_new_marker = True
                event.accept()
                return
            elif self._current_tool_mode == "delete":
                # Do nothing on background click for delete tool
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handles mouse moves for panning, marker dragging, and placement drag."""
        if self._is_middle_button_panning:
            delta = event.pos() - self._last_pan_pos
            self._last_pan_pos = event.pos()
            hs = self.horizontalScrollBar()
            vs = self.verticalScrollBar()
            hs.setValue(hs.value() - delta.x())
            vs.setValue(vs.value() - delta.y())
            event.accept()
            return

        if self._dragged_marker and self._current_tool_mode == "add_move":
            current_scene_pos = self.mapToScene(event.pos())
            self._dragged_marker.setPos(current_scene_pos)
            event.accept()
            return

        if self._is_placing_new_marker and self._current_tool_mode == "add_move":
            event.accept()  # Consume drag events during placement
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Handles mouse releases to finalize panning, marker drags, or new point placement."""
        if event.button() == Qt.MiddleButton and self._is_middle_button_panning:
            self._is_middle_button_panning = False
            self.set_cursor_for_mode()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._dragged_marker:
            final_pos = self._dragged_marker.pos()
            if self._current_tool_mode == "add_move":
                self.marker_move_finished.emit(self._dragged_marker, final_pos)
            self._dragged_marker = None
            # Restore original transform if we zoomed temporarily
            if self._original_transform_before_action is not None:
                self.setTransform(self._original_transform_before_action)
                self._original_transform_before_action = None
            self.set_cursor_for_mode()
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._is_placing_new_marker:
            if self._current_tool_mode == "add_move":
                final_pos = self.mapToScene(event.pos())
                self.scene_mouse_press.emit(final_pos)
            self._is_placing_new_marker = False
            # Restore original transform if we zoomed temporarily
            if self._original_transform_before_action is not None:
                self.setTransform(self._original_transform_before_action)
                self._original_transform_before_action = None
            self.set_cursor_for_mode()
            event.accept()
            return

        super().mouseReleaseEvent(event)

        # Ensure cursor is correct if nothing else handled the release
        if (
            not self._is_middle_button_panning
            and not self._dragged_marker
            and not self._is_placing_new_marker
        ):
            self.set_cursor_for_mode()

    def set_tool_mode(self, tool: str):
        """Sets the interaction mode ('add_move' or 'delete')."""
        self._current_tool_mode = tool
        self.setDragMode(QGraphicsView.NoDrag)  # Panning is always middle mouse
        self.set_cursor_for_mode()

    def set_cursor_for_mode(self):
        """Updates the mouse cursor based on the current tool mode."""
        if self._current_tool_mode == "add_move":
            cursor = Qt.CrossCursor
        elif self._current_tool_mode == "delete":
            cursor = Qt.PointingHandCursor
        else:
            cursor = Qt.ArrowCursor
        self.setCursor(cursor)
