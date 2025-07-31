import sys
from typing import Optional, Tuple, Literal, Any
import numpy as np
from dataclasses import dataclass

# Use PySide6 as specified in the original code
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
                               QStackedWidget, QFormLayout, QSpinBox,
                               QGridLayout, QSizePolicy)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QScreen

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib

# Set the backend for Matplotlib
matplotlib.use('qtagg')


@dataclass
class AppData:
    """
    Data class to hold the application's state.
    """
    grid_size: Tuple[int, int] = (10, 10)
    obstacle_map: Optional[np.ndarray] = None
    start: Optional[Tuple[int, int, int]] = (0, 0, 0)
    goal: Optional[Tuple[int, int, int]] = (9, 9, 0)


class CompassWidget(QWidget):
    """
    A widget with 8 buttons arranged like a compass to select an orientation (0-7).
    """
    orientation_changed = Signal(int)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self.current_orientation = 0
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setObjectName("compassTitle")

        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)

        self.button_map = {
            (1, 2): 0, (0, 2): 1, (0, 1): 2, (0, 0): 3,
            (1, 0): 4, (2, 0): 5, (2, 1): 6, (2, 2): 7,
        }

        arrows = {
            0: "→", 1: "↗", 2: "↑", 3: "↖",
            4: "←", 5: "↙", 6: "↓", 7: "↘",
        }

        # Map orientation index to angle in degrees
        self.index_to_angle = {i: i * 45 for i in range(8)}

        self.buttons = {}
        for (row, col), orientation in self.button_map.items():
            button = QPushButton(arrows[orientation])
            button.setObjectName("compassButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda _, o=orientation: self.set_orientation(o))
            grid_layout.addWidget(button, row, col)
            self.buttons[orientation] = button

        self.orientation_label = QLabel("0°")
        self.orientation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orientation_label.setObjectName("compassValue")
        grid_layout.addWidget(self.orientation_label, 1, 1)

        layout.addWidget(title_label)
        layout.addLayout(grid_layout)

        self.set_orientation(0)

    def set_orientation(self, orientation: int):
        if self.current_orientation in self.buttons:
            self.buttons[self.current_orientation].setChecked(False)

        self.current_orientation = orientation
        self.buttons[orientation].setChecked(True)

        # Display the angle, but emit the index
        angle = self.index_to_angle[orientation]
        self.orientation_label.setText(f"{angle}°")
        self.orientation_changed.emit(orientation)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Reinforcement Learning GUI")
        screen = QApplication.primaryScreen().availableGeometry()
        self.setGeometry(int(screen.width() * 0.15), int(screen.height() * 0.15),
                         int(screen.width() * 0.7), int(screen.height() * 0.7))

        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Warning: Could not load 'style.qss'. {e}")

        self.data = AppData()
        self.data.goal = (
            self.data.grid_size[0] - 1, self.data.grid_size[1] - 1, 0)

        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.start_page = StartPage(self)
        self.config_page = GridConfigurationPage(self)
        self.grid_generator = GridGenerator(self)
        self.end_page = EndPage(self)

        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.config_page)
        self.stacked_widget.addWidget(self.grid_generator)
        self.stacked_widget.addWidget(self.end_page)

        self.connect_signals_and_navigation()
        self.stacked_widget.setCurrentIndex(0)

    def connect_signals_and_navigation(self):
        self.config_page.grid_size_changed.connect(
            lambda size: self.update_data('grid_size', size))
        self.config_page.start_pos_changed.connect(
            lambda pos: self.update_data('start', (pos[0], pos[1], self.data.start[2])))
        self.config_page.goal_pos_changed.connect(
            lambda pos: self.update_data('goal', (pos[0], pos[1], self.data.goal[2])))

        self.grid_generator.start_orientation_changed.connect(
            lambda o: self.update_data('start', (self.data.start[0], self.data.start[1], o)))
        self.grid_generator.goal_orientation_changed.connect(
            lambda o: self.update_data('goal', (self.data.goal[0], self.data.goal[1], o)))
        self.grid_generator.grid_confirmed.connect(
            lambda map_data: self.update_data('obstacle_map', map_data))

        self.start_page.button_next.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1))
        self.config_page.button_next.clicked.connect(self.go_to_grid_generator)
        self.grid_generator.confirm_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(3))
        self.end_page.button_close.clicked.connect(self.close)

    def go_to_grid_generator(self):
        self.grid_generator.update_grid(
            self.data.grid_size, self.data.start, self.data.goal)
        self.stacked_widget.setCurrentIndex(2)

    def update_data(self, field_name: str, value: Any):
        if hasattr(self.data, field_name):
            print(f"Updating self.data.{field_name} to {value}")
            setattr(self.data, field_name, value)
            print("Current Data:", self.data)
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")


class StartPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        label = QLabel("Environment Setup")
        label.setObjectName("titleLabel")
        self.button_next = QPushButton("Start")
        self.button_next.setProperty("class", "navigation")
        layout.addWidget(label)
        layout.addWidget(self.button_next)


class GridConfigurationPage(QWidget):
    grid_size_changed = Signal(tuple)
    start_pos_changed = Signal(tuple)
    goal_pos_changed = Signal(tuple)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Grid Configuration")
        title.setObjectName("titleLabel")
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.spinbox_grid_cols = QSpinBox()
        self.spinbox_grid_cols.setRange(4, 100)
        self.spinbox_grid_cols.setValue(10)
        self.spinbox_grid_rows = QSpinBox()
        self.spinbox_grid_rows.setRange(4, 100)
        self.spinbox_grid_rows.setValue(10)
        self.spinbox_start_x = QSpinBox()
        self.spinbox_start_x.setMinimum(0)
        self.spinbox_start_y = QSpinBox()
        self.spinbox_start_y.setMinimum(0)
        self.spinbox_goal_x = QSpinBox()
        self.spinbox_goal_x.setMinimum(0)
        self.spinbox_goal_y = QSpinBox()
        self.spinbox_goal_y.setMinimum(0)
        self.button_next = QPushButton("Continue to Obstacle Drawing")
        self.button_next.setProperty("class", "navigation")

        form_layout.addRow("X Size:", self.spinbox_grid_cols)
        form_layout.addRow("Y Size:", self.spinbox_grid_rows)
        form_layout.addRow("Start X:", self.spinbox_start_x)
        form_layout.addRow("Start Y:", self.spinbox_start_y)
        form_layout.addRow("Goal X:", self.spinbox_goal_x)
        form_layout.addRow("Goal Y:", self.spinbox_goal_y)

        layout.addWidget(title)
        layout.addLayout(form_layout)
        layout.addWidget(self.button_next,
                         alignment=Qt.AlignmentFlag.AlignCenter)

        self.spinbox_grid_cols.valueChanged.connect(
            self._on_grid_dimensions_changed)
        self.spinbox_grid_rows.valueChanged.connect(
            self._on_grid_dimensions_changed)
        self.spinbox_start_x.valueChanged.connect(self._on_start_changed)
        self.spinbox_start_y.valueChanged.connect(self._on_start_changed)
        self.spinbox_goal_x.valueChanged.connect(self._on_goal_changed)
        self.spinbox_goal_y.valueChanged.connect(self._on_goal_changed)
        self._on_grid_dimensions_changed()

    def _on_grid_dimensions_changed(self):
        cols = self.spinbox_grid_cols.value()
        rows = self.spinbox_grid_rows.value()
        self.spinbox_start_x.setMaximum(cols - 1)
        self.spinbox_start_y.setMaximum(rows - 1)
        self.spinbox_goal_x.setMaximum(cols - 1)
        self.spinbox_goal_y.setMaximum(rows - 1)
        self.spinbox_goal_x.setValue(cols - 1)
        self.spinbox_goal_y.setValue(rows - 1)
        self.grid_size_changed.emit((cols, rows))
        self._on_start_changed()
        self._on_goal_changed()

    def _on_start_changed(self):
        self.start_pos_changed.emit(
            (self.spinbox_start_x.value(), self.spinbox_start_y.value()))

    def _on_goal_changed(self):
        self.goal_pos_changed.emit(
            (self.spinbox_goal_x.value(), self.spinbox_goal_y.value()))


class GridGenerator(QWidget):
    grid_confirmed = Signal(np.ndarray)
    start_orientation_changed = Signal(int)
    goal_orientation_changed = Signal(int)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()

        self.start_compass = CompassWidget("Start Orientation")
        self.goal_compass = CompassWidget("Goal Orientation")

        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        content_layout.addWidget(self.start_compass)
        content_layout.addWidget(self.canvas, 1)
        content_layout.addWidget(self.goal_compass)

        self.confirm_button = QPushButton("Confirm Grid and Finish")
        self.confirm_button.setObjectName("confirmButton")

        main_layout.addLayout(content_layout)
        main_layout.addWidget(self.confirm_button,
                              alignment=Qt.AlignmentFlag.AlignCenter)

        colors = ['#ffffff', '#e74c3c', '#2ecc71', '#3498db']
        self.cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = BoundaryNorm(bounds, self.cmap.N)

        # --- FIX: Initialize state attributes ---
        self.nx = 0
        self.ny = 0
        self.start_pos = None
        self.goal_pos = None
        self.display_map = None
        self.is_drawing = False
        self.is_dragged = False
        self.last_pos = None
        self.drag_mode = 'draw'

        self.connect_events()

    def update_grid(self, grid_size: Tuple[int, int], start_pos: Tuple[int, int, int], goal_pos: Tuple[int, int, int]):
        self.nx, self.ny = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        self.display_map = np.zeros((self.ny, self.nx), dtype=int)
        self.display_map[self.start_pos[1], self.start_pos[0]] = 2
        self.display_map[self.goal_pos[1], self.goal_pos[0]] = 3
        self.start_compass.set_orientation(start_pos[2])
        self.goal_compass.set_orientation(goal_pos[2])
        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()

        self.im = self.ax.imshow(
            self.display_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.ax.set_xlim(-0.5, self.nx - 0.5)
        self.ax.set_ylim(self.ny - 0.5, -0.5)

        self.ax.set_title(
            "Click/drag to draw obstacles. Use compasses to set orientation.")
        self._draw_orientation_arrows()
        self.canvas.draw()

    def _draw_orientation_arrows(self):
        orientations = {
            0: (1, 0), 1: (0.707, -0.707), 2: (0, -1), 3: (-0.707, -0.707),
            4: (-1, 0), 5: (-0.707, 0.707), 6: (0, 1), 7: (0.707, 0.707)
        }

        sx, sy, so = self.start_pos
        dx, dy = orientations[so]
        # Arrow length and head size reduced to fit inside the cell
        self.ax.arrow(sx, sy, dx*0.25, dy*0.25, head_width=0.2,
                      head_length=0.2, fc='k', ec='k')

        gx, gy, go = self.goal_pos
        dx, dy = orientations[go]
        # Arrow length and head size reduced to fit inside the cell
        self.ax.arrow(gx, gy, dx*0.25, dy*0.25, head_width=0.2,
                      head_length=0.2, fc='k', ec='k')

    def connect_events(self):
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.confirm_button.clicked.connect(self.on_confirm)
        self.start_compass.orientation_changed.connect(
            self.on_start_orientation_change)
        self.goal_compass.orientation_changed.connect(
            self.on_goal_orientation_change)

    def on_start_orientation_change(self, orientation: int):
        self.start_pos = (self.start_pos[0], self.start_pos[1], orientation)
        self.start_orientation_changed.emit(orientation)
        self.setup_plot()
        self.canvas.draw_idle()

    def on_goal_orientation_change(self, orientation: int):
        self.goal_pos = (self.goal_pos[0], self.goal_pos[1], orientation)
        self.goal_orientation_changed.emit(orientation)
        self.setup_plot()
        self.canvas.draw_idle()

    def _is_protected_cell(self, x: int, y: int) -> bool:
        return (x == self.start_pos[0] and y == self.start_pos[1]) or \
               (x == self.goal_pos[0] and y == self.goal_pos[1])

    def on_press(self, event):
        if not event.inaxes:
            return
        self.is_drawing = True
        self.is_dragged = False
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            self.drag_mode = 'erase' if self.display_map[iy,
                                                         ix] == 1 else 'draw'

    def on_release(self, event):
        if not self.is_drawing:
            return
        if not self.is_dragged and event.inaxes:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < self.nx and 0 <= iy < self.ny and not self._is_protected_cell(ix, iy):
                self.display_map[iy, ix] = 1 - self.display_map[iy, ix]
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
        self.is_drawing = False

    def on_motion(self, event):
        if not self.is_drawing or not event.inaxes:
            return
        self.is_dragged = True
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != getattr(self, 'last_pos', None):
            if not self._is_protected_cell(ix, iy):
                self.display_map[iy, ix] = 1 if self.drag_mode == 'draw' else 0
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def on_confirm(self):
        final_obstacle_map = np.where(self.display_map == 1, 1, 0)
        self.grid_confirmed.emit(final_obstacle_map)


class EndPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel("Configuration Finished")
        label.setObjectName("titleLabel")
        self.button_close = QPushButton("Close Window")
        self.button_close.setObjectName("closeButton")
        layout.addWidget(label)
        layout.addWidget(self.button_close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
