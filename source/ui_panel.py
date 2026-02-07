import os
import sys
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QHBoxLayout,
    QSplitter,
    QListWidget,
    QGraphicsScene,
    QToolBar,
    QGraphicsPixmapItem,
)
from PySide6.QtGui import QAction, QKeySequence, QIcon, QActionGroup
from PySide6.QtCore import Qt, QSize

from graphics_widgets import ZoomableView


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, relative_path)


class UiPanelMixin:
    def _setup_ui(self):
        """Creates and arranges the UI widgets."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)

        self.image_list_widget = QListWidget()
        self.image_list_widget.currentItemChanged.connect(
            self.on_image_selection_changed
        )
        self.image_list_widget.setIconSize(self.thumbnail_size)
        main_splitter.addWidget(self.image_list_widget)

        self.scene = QGraphicsScene()
        self.view = ZoomableView(self.scene)
        self.view.scene_mouse_press.connect(self.handle_scene_click)
        self.view.marker_action_click.connect(self.handle_marker_action)
        self.view.marker_move_finished.connect(self.finalize_marker_move)
        main_splitter.addWidget(self.view)

        self.point_set_list_widget = QListWidget()
        self.point_set_list_widget.currentItemChanged.connect(
            self.on_point_set_selection_changed
        )
        self.point_set_list_widget.itemDoubleClicked.connect(
            self.on_point_set_double_clicked
        )
        main_splitter.addWidget(self.point_set_list_widget)
        main_splitter.setSizes([250, 950, 200])

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)
        self._setup_toolbar(toolbar)

        self.statusBar()
        self.pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._update_window_title()

    def _setup_toolbar(self, toolbar: QToolBar):
        """Adds actions to the toolbar.
        Tries local icons first, then falls back to system theme.
        """

        def add_action(
            icon_filename: str,
            theme_name: str,
            tooltip: str,
            slot,
            shortcut: Optional[QKeySequence] = None,
            checkable: bool = False,
            checked: bool = False,
        ) -> QAction:
            local_icon_path = resource_path(os.path.join("icons", icon_filename))
            icon = (
                QIcon(local_icon_path)
                if os.path.exists(local_icon_path)
                else QIcon.fromTheme(theme_name)
            )
            action = QAction(icon, "", self)
            if shortcut:
                action.setShortcut(shortcut)
            action.setCheckable(checkable)
            if checked:
                action.setChecked(True)
            action.setToolTip(tooltip)
            action.triggered.connect(slot)
            toolbar.addAction(action)
            return action

        for icon_file, theme, tip, slot, shortcut in [
            (
                "go-down.svg",
                "go-down",
                "Import Images... (Ctrl+O)",
                self.open_images,
                QKeySequence.Open,
            ),
            ("document-open.svg", "document-open", "Open Project...", self.open_project, None),
            (
                "document-save.svg",
                "document-save",
                "Save Project (Ctrl+S)",
                self.save_points,
                QKeySequence.Save,
            ),
            (
                "document-save-as.svg",
                "document-save-as",
                "Save Project As... (Ctrl+Shift+S)",
                self.save_points_as,
                QKeySequence.SaveAs,
            ),
        ]:
            add_action(icon_file, theme, tip, slot, shortcut)

        toolbar.addSeparator()
        add_action(
            "document-print.svg",
            "document-print",
            "Export Scene As GLTF...",
            self.export_scene_as,
        )
        toolbar.addSeparator()

        self.add_point_tool_action = add_action(
            "list-add.svg",
            "list-add",
            "Add/Move Point Tool",
            self.activate_add_point_tool,
            checkable=True,
            checked=True,
        )
        self.delete_point_tool_action = add_action(
            "edit-cut.svg",
            "edit-cut",
            "Delete Point Tool",
            self.activate_delete_point_tool,
            checkable=True,
        )

        self.tool_action_group = QActionGroup(self)
        self.tool_action_group.setExclusive(True)
        self.tool_action_group.addAction(self.add_point_tool_action)
        self.tool_action_group.addAction(self.delete_point_tool_action)

        toolbar.addSeparator()
        calibrate_action = add_action(
            "measure.svg",
            "accessories-engineering",
            "Run Calibration (SfM + Bundle Adjustment)",
            self.run_calibration,
        )
        if not self._pycolmap_available:
            calibrate_action.setEnabled(False)
            calibrate_action.setToolTip("Calibration disabled (PyCOLMAP not found)")
