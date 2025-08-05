import sys
from typing import Optional, Tuple, Any
import numpy as np
from dataclasses import dataclass, fields
import math

# Use PySide6 as specified in the original code
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QFormLayout,
    QSpinBox,
    QGridLayout,
    QSizePolicy,
    QButtonGroup,
    QSlider,
    QFileDialog,
    QMessageBox,
)
from PySide6.QtCore import Signal, Qt, QThread, QObject
from PySide6.QtGui import QScreen

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib

# Import classes and functions from the Reinforcement Learning script
from rl_differential import GridWorld, QLearningAgent, train_adaptative

matplotlib.use("qtagg")


@dataclass
class AppData:
    """
    Data class to hold the application's state.
    """
    grid_size: Tuple[int, int] = (10, 10)
    obstacle_map: Optional[np.ndarray] = None
    start: Optional[Tuple[int, int, int]] = (0, 0, 0)
    goal: Optional[Tuple[int, int, int]] = (9, 9, 0)
    goal_orientation_irrelevant: bool = False
    agent_type: str = "Omnidirectional"
    # Variables for the parameter sliders
    param1: int = 50
    param2: int = 50
    param3: int = 50
    param4: int = 50
    param5: int = 50


class TrainingWorker(QObject):
    """
    Runs the training task in a separate thread to avoid freezing the GUI.
    """
    # Signal emitted at the end of training, sending the trained agent, environment, and metrics
    training_finished = Signal(object, object, object)

    def __init__(self, app_data: AppData):
        super().__init__()
        self.data = app_data

    def run(self):
        """
        Sets up the environment and agent, then starts the training.
        """
        print("Starting training in the worker thread...")

        # Convert orientation index (0-7) to radians
        start_angle = (self.data.start[2] * 45 * np.pi) / 180.0
        goal_angle = (self.data.goal[2] * 45 * np.pi) / 180.0

        env = GridWorld(
            width=self.data.grid_size[0],
            height=self.data.grid_size[1],
            grid_map=self.data.obstacle_map,
            start=(self.data.start[1], self.data.start[0],
                   start_angle),  # (row, col)
            goal=(self.data.goal[1], self.data.goal[0],
                  goal_angle),  # (row, col),
            final_orientation_irrelevant=self.data.goal_orientation_irrelevant
        )
        agent = QLearningAgent(env, alpha=0.1, gamma=0.99,
                               min_epsilon=0.05, max_epsilon=0.9)
        max_steps = env.height * env.width * 2
        _, _, metrics = train_adaptative(
            env, agent, success_window=100, max_steps=max_steps, inc_step=0.02)

        print("Training finished.")

        # Emit the signal with the results
        self.training_finished.emit(agent, env, metrics)


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
        arrows = {0: "→", 1: "↗", 2: "↑", 3: "↖",
                  4: "←", 5: "↙", 6: "↓", 7: "↘"}
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
        angle = self.index_to_angle[orientation]
        self.orientation_label.setText(f"{angle}°")
        self.orientation_changed.emit(orientation)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Reinforcement Learning GUI")
        screen_geometry = self.screen().availableGeometry()
        screen_width = int(screen_geometry.width() * 0.8)
        screen_height = int(screen_geometry.height() * 0.8)
        self.setGeometry(
            screen_geometry.center().x() - screen_width // 2,
            screen_geometry.center().y() - screen_height // 2,
            screen_width,
            screen_height,
        )

        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Warning: Could not load 'style.qss'. {e}")

        self.data = AppData()
        self.trained_agent = None
        self.trained_env = None
        self.training_metrics = None

        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.start_page = StartPage(self)
        self.grid_source_page = GridSourcePage(self)
        self.config_page = GridConfigurationPage(self)
        self.grid_generator = GridGenerator(self)
        self.training_config_page = TrainingConfigurationPage(self)
        self.training_page = TrainingScreen(self)
        self.results_page = InteractiveResultsPage(self)
        self.end_page = EndPage(self)

        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.grid_source_page)
        self.stacked_widget.addWidget(self.config_page)
        self.stacked_widget.addWidget(self.grid_generator)
        self.stacked_widget.addWidget(self.training_config_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.results_page)
        self.stacked_widget.addWidget(self.end_page)

        self.connect_signals_and_navigation()
        self.stacked_widget.setCurrentIndex(0)

    def connect_signals_and_navigation(self):
        # Navigation
        self.start_page.button_next.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1))
        self.grid_source_page.navigation_requested.connect(
            self.go_to_config_page)
        self.config_page.button_next.clicked.connect(self.go_to_grid_generator)
        self.grid_generator.confirm_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(4))
        self.training_config_page.button_finish.clicked.connect(
            self.start_training)
        self.results_page.button_finish.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(7))
        self.end_page.button_close.clicked.connect(self.close)

        # Data updates
        self.config_page.grid_size_changed.connect(
            lambda size: self.update_data("grid_size", size))
        self.config_page.start_pos_changed.connect(
            lambda pos: self.update_data("start", (pos[0], pos[1], self.data.start[2])))
        self.config_page.goal_pos_changed.connect(
            lambda pos: self.update_data("goal", (pos[0], pos[1], self.data.goal[2])))
        self.grid_generator.start_orientation_changed.connect(
            lambda o: self.update_data("start", (self.data.start[0], self.data.start[1], o)))
        self.grid_generator.goal_orientation_changed.connect(
            lambda o: self.update_data("goal", (self.data.goal[0], self.data.goal[1], o)))
        self.grid_generator.goal_orientation_relevance_changed.connect(
            lambda irrelevant: self.update_data("goal_orientation_irrelevant", irrelevant))
        self.grid_generator.grid_confirmed.connect(
            lambda map_data: self.update_data("obstacle_map", map_data))
        self.training_config_page.agent_type_changed.connect(
            lambda agent: self.update_data("agent_type", agent))
        self.training_config_page.param_changed.connect(self.update_data)

    def go_to_config_page(self, is_imported: bool):
        if not is_imported:
            self.update_data("obstacle_map", None)
        self.config_page.update_view(is_imported)
        self.stacked_widget.setCurrentIndex(2)

    def go_to_grid_generator(self):
        if self.data.obstacle_map is not None:
            rows, cols = self.data.obstacle_map.shape
            self.update_data("grid_size", (cols, rows))
        self.grid_generator.update_grid(
            self.data.grid_size, self.data.start, self.data.goal)
        self.stacked_widget.setCurrentIndex(3)

    def start_training(self):
        self.stacked_widget.setCurrentIndex(5)
        self.training_page.start_worker()

    def update_data(self, field_name: str, value: Any):
        if hasattr(self.data, field_name):
            setattr(self.data, field_name, value)
            print(f"Updating self.data.{field_name} to {value}")
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")

    def on_training_complete(self, agent, env, metrics):
        """
        Slot to receive the training result from the worker thread.
        """
        print("MainWindow notified of training completion. Navigating to results page.")
        self.trained_agent = agent
        self.trained_env = env
        self.training_metrics = metrics
        self.results_page.setup_page()
        self.stacked_widget.setCurrentIndex(6)


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


class GridSourcePage(QWidget):
    navigation_requested = Signal(bool)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)
        title = QLabel("Grid Source")
        title.setObjectName("titleLabel")
        button_draw = QPushButton("Draw Grid from Scratch")
        button_draw.setProperty("class", "navigation")
        button_draw.clicked.connect(
            lambda: self.navigation_requested.emit(False))
        button_import = QPushButton("Import Grid from File")
        button_import.setProperty("class", "navigation")
        button_import.clicked.connect(self._handle_import)
        self.import_status_label = QLabel("No file imported.")
        self.import_status_label.setObjectName("statusLabel")
        self.import_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(button_draw)
        layout.addWidget(button_import)
        layout.addWidget(self.import_status_label)

    def _handle_import(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Grid", "", "Numpy files (*.npy);;CSV files (*.csv)")
        if not file_path:
            return
        try:
            if file_path.endswith('.npy'):
                grid_map = np.load(file_path)
            elif file_path.endswith('.csv'):
                grid_map = np.loadtxt(file_path, delimiter=',')
            else:
                raise ValueError("Unsupported file type")
            if grid_map.ndim != 2 or not np.all(np.isin(grid_map, [0, 1])):
                raise ValueError("Map must be a 2D array of 0s and 1s.")
            rows, cols = grid_map.shape
            self.main_window.update_data("obstacle_map", grid_map)
            self.main_window.update_data("grid_size", (cols, rows))
            file_name = file_path.split('/')[-1]
            self.import_status_label.setText(f"Imported: {file_name}")
            self.import_status_label.setProperty("class", "success")
            self.navigation_requested.emit(True)
        except Exception as e:
            self.import_status_label.setText(f"Error: {e}")
            self.import_status_label.setProperty("class", "error")
            print(f"Failed to import grid: {e}")

        # Re-polish to apply dynamic property changes from QSS
        self.import_status_label.style().unpolish(self.import_status_label)
        self.import_status_label.style().polish(self.import_status_label)


class GridConfigurationPage(QWidget):
    grid_size_changed = Signal(tuple)
    start_pos_changed = Signal(tuple)
    goal_pos_changed = Signal(tuple)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("Grid Configuration")
        title.setObjectName("titleLabel")
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        self.spinbox_grid_cols = QSpinBox()
        self.spinbox_grid_cols.setRange(4, 100)
        self.spinbox_grid_rows = QSpinBox()
        self.spinbox_grid_rows.setRange(4, 100)
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
        self.update_view(False)

    def update_view(self, is_imported: bool):
        self.spinbox_grid_cols.setEnabled(not is_imported)
        self.spinbox_grid_rows.setEnabled(not is_imported)
        if is_imported:
            cols, rows = self.main_window.data.grid_size
            self.spinbox_grid_cols.setValue(cols)
            self.spinbox_grid_rows.setValue(rows)
        else:
            self.spinbox_grid_cols.setValue(10)
            self.spinbox_grid_rows.setValue(10)
        self._on_grid_dimensions_changed()

    def _on_grid_dimensions_changed(self, value=None):
        cols = self.spinbox_grid_cols.value()
        rows = self.spinbox_grid_rows.value()
        self.spinbox_start_x.setMaximum(cols - 1)
        self.spinbox_start_y.setMaximum(rows - 1)
        self.spinbox_goal_x.setMaximum(cols - 1)
        self.spinbox_goal_y.setMaximum(rows - 1)
        if not self.spinbox_grid_cols.isEnabled():
            self.spinbox_goal_x.setValue(
                min(self.spinbox_goal_x.value(), cols - 1))
            self.spinbox_goal_y.setValue(
                min(self.spinbox_goal_y.value(), rows - 1))
        else:
            self.spinbox_goal_x.setValue(cols - 1)
            self.spinbox_goal_y.setValue(rows - 1)
        self.grid_size_changed.emit((cols, rows))
        self._on_start_changed()
        self._on_goal_changed()

    def _on_start_changed(self, value=None):
        self.start_pos_changed.emit(
            (self.spinbox_start_x.value(), self.spinbox_start_y.value()))

    def _on_goal_changed(self, value=None):
        self.goal_pos_changed.emit(
            (self.spinbox_goal_x.value(), self.spinbox_goal_y.value()))


class GridGenerator(QWidget):
    grid_confirmed = Signal(np.ndarray)
    start_orientation_changed = Signal(int)
    goal_orientation_changed = Signal(int)
    goal_orientation_relevance_changed = Signal(bool)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        main_layout = QVBoxLayout(self)
        content_layout = QHBoxLayout()
        start_controls_layout = QVBoxLayout()
        self.start_compass = CompassWidget("Start Orientation")
        start_controls_layout.addStretch()
        start_controls_layout.addWidget(self.start_compass)
        start_controls_layout.addStretch()
        goal_controls_layout = QVBoxLayout()
        self.goal_compass = CompassWidget("Goal Orientation")
        self.button_irrelevant = QPushButton("Irrelevant Orientation")
        self.button_irrelevant.setCheckable(True)
        self.button_irrelevant.setProperty("class", "toggleButton")
        goal_controls_layout.addStretch()
        goal_controls_layout.addWidget(self.goal_compass)
        goal_controls_layout.addWidget(
            self.button_irrelevant, 0, Qt.AlignmentFlag.AlignCenter)
        goal_controls_layout.addStretch()
        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addLayout(start_controls_layout)
        content_layout.addWidget(self.canvas, 1)
        content_layout.addLayout(goal_controls_layout)
        self.confirm_button = QPushButton("Confirm Grid and Continue")
        self.confirm_button.setObjectName("confirmButton")
        main_layout.addLayout(content_layout)
        main_layout.addWidget(self.confirm_button,
                              alignment=Qt.AlignmentFlag.AlignCenter)
        colors = ["#ffffff", "#e74c3c", "#2ecc71", "#3498db"]
        self.cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = BoundaryNorm(bounds, self.cmap.N)
        self.nx, self.ny = 0, 0
        self.start_pos, self.goal_pos = None, None
        self.display_map = None
        self.is_drawing, self.is_dragged = False, False
        self.last_pos, self.drag_mode = None, "draw"
        self.connect_events()

    def update_grid(self, grid_size: Tuple[int, int], start_pos: Tuple[int, int, int], goal_pos: Tuple[int, int, int]):
        self.nx, self.ny = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos
        if self.main_window.data.obstacle_map is not None:
            self.display_map = self.main_window.data.obstacle_map.copy().astype(int)
        else:
            self.display_map = np.zeros((self.ny, self.nx), dtype=int)
        self.display_map[self.start_pos[1], self.start_pos[0]] = 2
        self.display_map[self.goal_pos[1], self.goal_pos[0]] = 3
        self.start_compass.set_orientation(start_pos[2])
        self.goal_compass.set_orientation(goal_pos[2])
        self.setup_plot()

    def setup_plot(self):
        self.ax.clear()
        self.im = self.ax.imshow(self.display_map, cmap=self.cmap,
                                 norm=self.norm, interpolation="nearest", origin='lower')

        # --- CORRECTION: Draw grid lines correctly ---
        self.ax.set_xticks(np.arange(-0.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black",
                     linestyle="-", linewidth=0.5)
        self.ax.tick_params(which="minor", size=0)
        # Hide major tick labels
        self.ax.tick_params(axis='x', which='major',
                            bottom=False, top=False, labelbottom=False)
        self.ax.tick_params(axis='y', which='major',
                            left=False, right=False, labelleft=False)

        self.ax.set_xlim(-0.5, self.nx - 0.5)
        self.ax.set_ylim(-0.5, self.ny - 0.5)
        self.ax.set_title(
            "Click/drag to draw obstacles. Use compasses for orientation.")
        self._draw_orientation_arrows()
        self.canvas.draw()

    def _draw_orientation_arrows(self):
        visual_orientations = {
            0: (1, 0), 1: (0.707, 0.707), 2: (0, 1), 3: (-0.707, 0.707),
            4: (-1, 0), 5: (-0.707, -0.707), 6: (0, -1), 7: (0.707, -0.707)
        }
        sx, sy, so = self.start_pos
        dx, dy = visual_orientations[so]
        self.ax.arrow(sx, sy, dx * 0.25, dy * 0.25,
                      head_width=0.2, head_length=0.2, fc="k", ec="k")
        if not self.main_window.data.goal_orientation_irrelevant:
            gx, gy, go = self.goal_pos
            dx, dy = visual_orientations[go]
            self.ax.arrow(gx, gy, dx * 0.25, dy * 0.25,
                          head_width=0.2, head_length=0.2, fc="k", ec="k")

    def connect_events(self):
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)
        self.confirm_button.clicked.connect(self.on_confirm)
        self.start_compass.orientation_changed.connect(
            self.on_start_orientation_change)
        self.goal_compass.orientation_changed.connect(
            self.on_goal_orientation_change)
        self.button_irrelevant.toggled.connect(self._on_relevance_changed)

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

    def _on_relevance_changed(self, is_irrelevant: bool):
        self.goal_compass.setEnabled(not is_irrelevant)
        self.goal_orientation_relevance_changed.emit(is_irrelevant)
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
            self.drag_mode = "erase" if self.display_map[iy,
                                                         ix] == 1 else "draw"

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
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.last_pos:
            if not self._is_protected_cell(ix, iy):
                self.display_map[iy, ix] = 1 if self.drag_mode == "draw" else 0
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def on_confirm(self):
        final_obstacle_map = np.where(self.display_map == 1, 1, 0)
        self.grid_confirmed.emit(final_obstacle_map)


class TrainingConfigurationPage(QWidget):
    agent_type_changed = Signal(str)
    param_changed = Signal(str, int)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.agent_types = ["Omnidirectional", "Differential", "Car Like"]
        self.param_names = [f.name for f in fields(
            AppData) if f.name.startswith("param")]
        self.value_labels = {}
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(40)
        title = QLabel("Select Agent Type")
        title.setObjectName("titleLabel")
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.buttons = {}
        for agent_name in self.agent_types:
            button = QPushButton(agent_name)
            button.setProperty("class", "agentButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, name=agent_name: self.select_agent(name) if checked else None)
            buttons_layout.addWidget(button)
            self.buttons[agent_name] = button
            self.button_group.addButton(button)
        main_layout.addLayout(buttons_layout)
        sliders_title = QLabel("Training Parameters")
        sliders_title.setObjectName("titleLabel")
        main_layout.addWidget(
            sliders_title, alignment=Qt.AlignmentFlag.AlignCenter)
        sliders_container = QWidget()
        sliders_layout = QFormLayout(sliders_container)
        sliders_layout.setSpacing(50)
        sliders_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        sliders_container.setObjectName("slidersContainer")
        for i, param_name in enumerate(self.param_names):
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setRange(1, 100)
            slider.setValue(50)
            self.value_labels[param_name] = QLabel("50")
            self.value_labels[param_name].setObjectName("sliderValueLabel")
            self.value_labels[param_name].setMinimumWidth(30)
            slider.valueChanged.connect(
                lambda value, name=param_name: self._on_slider_changed(name, value))
            slider_row_layout = QHBoxLayout()
            slider_row_layout.addWidget(slider)
            slider_row_layout.addWidget(self.value_labels[param_name])
            sliders_layout.addRow(f"Parameter {i+1}:", slider_row_layout)
            self.param_changed.emit(param_name, 50)
        main_layout.addWidget(sliders_container)
        main_layout.addStretch()
        self.button_finish = QPushButton("Train Agent")
        self.button_finish.setProperty("class", "navigation")
        main_layout.addWidget(self.button_finish,
                              alignment=Qt.AlignmentFlag.AlignCenter)
        self.buttons[self.agent_types[0]].setChecked(True)
        self.select_agent(self.agent_types[0])

    def select_agent(self, agent_name: str):
        self.agent_type_changed.emit(agent_name)

    def _on_slider_changed(self, name: str, value: int):
        self.value_labels[name].setText(str(value))
        self.param_changed.emit(name, value)


class TrainingScreen(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        title = QLabel("Training Agent")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label = QLabel("Preparing to start training...")
        self.status_label.setObjectName("trainingStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(self.status_label)

    def start_worker(self):
        self.status_label.setText(
            "Training in progress... Please wait.\nThe interface will remain responsive.")
        self.thread = QThread()
        self.worker = TrainingWorker(self.main_window.data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.training_finished.connect(
            self.main_window.on_training_complete)
        self.worker.training_finished.connect(self.thread.quit)
        self.worker.training_finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()


class InteractiveResultsPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.selected_start_pos = None

        # Main Layout
        main_layout = QVBoxLayout(self)

        # Title
        title = QLabel("Learned Policy Visualization")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)

        # Top container for controls and plot
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        # --- Left Side: Controls (Reorganized for better centering) ---
        controls_container = QWidget()
        controls_container.setObjectName("controlsContainer")
        controls_container.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(controls_container)

        self.instruction_label = QLabel(
            "Click on the grid to select a start point.")
        self.instruction_label.setObjectName("instructionLabel")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_compass = CompassWidget("Select Start Orientation")

        self.visualize_button = QPushButton("Visualize Path")
        self.visualize_button.setProperty("class", "navigation")

        # Add widgets to the centered layout
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.instruction_label)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.start_compass)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.visualize_button)
        controls_layout.addStretch(1)

        # --- Right Side: Matplotlib Canvas ---
        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_layout.addWidget(self.canvas, 1)

        # Bottom Finish Button
        self.button_finish = QPushButton("Finish Visualization")
        self.button_finish.setProperty("class", "navigation")
        main_layout.addWidget(self.button_finish,
                              alignment=Qt.AlignmentFlag.AlignCenter)

        # Connect signals
        self.visualize_button.clicked.connect(
            self.plot_greedy_path_from_selection)
        self.canvas.mpl_connect('button_press_event', self.on_canvas_click)

    def setup_page(self):
        env = self.main_window.trained_env
        if not env:
            return
        self.selected_start_pos = (
            env.start[0], env.start[1])  # Default to env start
        self.start_compass.set_orientation(env.start[2])
        self.instruction_label.setText(
            f"Start point selected at (Y={env.start[0]}, X={env.start[1]}).\nClick grid to change or visualize path.")
        self.draw_base_grid()
        self.draw_path_from_point(self.selected_start_pos, env.start[2])

    def draw_base_grid(self):
        self.ax.clear()
        env = self.main_window.trained_env
        if not env:
            self.ax.text(0.5, 0.5, "Error: Training data not found.",
                         ha='center', va='center')
            self.canvas.draw()
            return

        H, W = env.height, env.width
        bg = np.ones((H, W, 3))
        bg[env.grid == 1] = (0.2, 0.2, 0.2)  # Darker obstacles
        self.ax.imshow(bg, origin="lower", interpolation="nearest")

        # Draw grid lines
        self.ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        self.ax.grid(which="minor", color="k",
                     linestyle="-", linewidth=0.5, alpha=0.2)
        self.ax.tick_params(which="minor", size=0)
        self.ax.tick_params(axis='x', which='major',
                            bottom=False, top=False, labelbottom=False)
        self.ax.tick_params(axis='y', which='major',
                            left=False, right=False, labelleft=False)

        self.ax.set_xlim([-0.5, W - 0.5])
        self.ax.set_ylim([-0.5, H - 0.5])
        self.ax.set_title("Click a cell to select a start, then visualize")
        self.ax.scatter(env.goal[1], env.goal[0], marker="*",
                        c="#e74c3c", s=250, zorder=5, label="Goal", edgecolors='black')
        self.canvas.draw()

    def on_canvas_click(self, event):
        if not event.inaxes:
            return

        env = self.main_window.trained_env
        if not env:
            return

        # Convert click coordinates to cell indices
        c, r = int(round(event.xdata)), int(round(event.ydata))

        # Check if click is within bounds
        if not (0 <= r < env.height and 0 <= c < env.width):
            return

        # Check if the cell is an obstacle
        if env.grid[r, c] == 1:
            QMessageBox.warning(self, "Invalid Start",
                                "The selected cell is an obstacle.")
            return

        self.selected_start_pos = (r, c)
        self.instruction_label.setText(
            f"Start point selected at (Y={r}, X={c}).\nAdjust orientation and visualize.")

        # Redraw grid and show a temporary marker for the selected start
        self.draw_base_grid()
        self.ax.scatter(c, r, marker="o", c="#2ecc71", s=150,
                        zorder=5, label="Selected Start", edgecolors='black')
        self.canvas.draw()

    def plot_greedy_path_from_selection(self):
        if self.selected_start_pos is None:
            QMessageBox.warning(
                self, "No Start Point", "Please click on the grid to select a starting point first.")
            return

        start_k = self.start_compass.current_orientation
        self.draw_path_from_point(self.selected_start_pos, start_k)

    def draw_path_from_point(self, start_pos, start_orientation):
        env = self.main_window.trained_env
        agent = self.main_window.trained_agent
        if not env or not agent:
            QMessageBox.critical(self, "Error", "Training data not found.")
            return

        start_r, start_c = start_pos
        start_k = start_orientation

        self.draw_base_grid()

        eps_bak = agent.epsilon
        agent.epsilon = 0.0

        env.reset_new_position((start_r, start_c, start_k))
        s = env.agent_pos
        path = [s]
        limit = env.width * env.height * 2
        for i in range(limit):
            a = agent.choose_action(s)
            s, _, done = env.step(a)
            path.append(s)
            if done:
                break
            if i == limit - 1:
                QMessageBox.warning(self, "Path Limit Reached",
                                    "The path limit was reached without finding a solution.")
                break

        agent.epsilon = eps_bak

        self.ax.scatter(start_c, start_r, marker="o", c="#2ecc71",
                        s=150, zorder=5, label="Selected Start", edgecolors='black')

        for (pr, pc, _) in path:
            if env.grid[pr, pc] == 0 and (pr, pc) != (env.goal[0], env.goal[1]):
                self.ax.add_patch(plt.Rectangle(
                    (pc - 0.5, pr - 0.5), 1, 1, fill=True, alpha=0.3, color="#3498db", zorder=3))

        xs, ys, us, vs = [], [], [], []
        arrow_scale = 0.35
        for (pr, pc, kk) in path:
            ang = env.idx_to_angle(kk)
            xs.append(pc)
            ys.append(pr)
            us.append(math.cos(ang) * arrow_scale)
            vs.append(math.sin(ang) * arrow_scale)
        if xs:
            self.ax.quiver(xs, ys, us, vs, angles='xy', scale_units='xy',
                           scale=1, width=0.015, color='yellow', zorder=6)

        self.ax.set_title(
            f"Greedy path from (Y={start_r}, X={start_c}, θ={start_k}°) | Steps: {len(path)-1}")
        self.canvas.draw()


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
