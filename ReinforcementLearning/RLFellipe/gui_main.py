import sys
from typing import Optional, Tuple, Literal, Any
from dataclasses import dataclass

# Use PySide6 as specified in the original code
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QPushButton, QLabel,
                               QStackedWidget, QFormLayout, QComboBox,
                               QSpinBox)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QScreen

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
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


class MainWindow(QMainWindow):
    """
    The main window of the application. It manages the pages (widgets)
    and the shared application data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Reinforcement Learning GUI")

        # Adjust the size of the main window to be 70% of screen size
        screen = QApplication.primaryScreen().availableGeometry()
        window_width = int(screen.width() * 0.7)
        window_height = int(screen.height() * 0.7)
        self.setGeometry(
            int((screen.width() - window_width) / 2),
            int((screen.height() - window_height) / 2),
            window_width, window_height
        )

        # --- Load and apply external stylesheet ---
        try:
            with open("style.qss", "r") as f:
                style_sheet = f.read()
                self.setStyleSheet(style_sheet)
        except FileNotFoundError:
            print("Warning: 'style.qss' file not found. Using default styles.")
        except Exception as e:
            print(f"Error loading 'style.qss': {e}")

        # Define data object
        self.data = AppData()
        # Set a default goal based on the default grid size
        self.data.goal = (
            self.data.grid_size[0] - 1, self.data.grid_size[1] - 1, 0)

        # Create a stacked widget to hold the pages
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Create instances of the pages
        self.start_page = StartPage(self)
        self.config_page = GridConfigurationPage(self)
        self.grid_generator = GridGenerator(self)  # Pass main window reference
        self.end_page = EndPage(self)

        # Add pages to the stacked widget in the correct order
        self.stacked_widget.addWidget(self.start_page)        # Index 0
        self.stacked_widget.addWidget(self.config_page)       # Index 1
        self.stacked_widget.addWidget(self.grid_generator)    # Index 2
        self.stacked_widget.addWidget(self.end_page)          # Index 3

        # Define starting page
        self.stacked_widget.setCurrentIndex(0)

        # --- Centralized Signal and Navigation Connection ---
        self.connect_signals_and_navigation()

    def connect_signals_and_navigation(self):
        """Connects all signals and navigation logic in one place."""
        # Data update signals
        self.config_page.grid_size_changed.connect(
            lambda size: self.update_data('grid_size', size)
        )
        self.config_page.start_pose_changed.connect(
            lambda pose: self.update_data('start', pose)
        )
        self.config_page.goal_pose_changed.connect(
            lambda pose: self.update_data('goal', pose)
        )
        self.grid_generator.grid_confirmed.connect(
            lambda generated_map: self.update_data(
                'obstacle_map', generated_map)
        )

        # Page navigation
        self.start_page.button_next.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.config_page.button_next.clicked.connect(self.go_to_grid_generator)
        self.grid_generator.confirm_button.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(3)
        )
        self.end_page.button_close.clicked.connect(self.close)

    def go_to_grid_generator(self):
        """
        Action to perform before switching to the GridGenerator page.
        It ensures the grid is updated with the latest configuration.
        """
        self.grid_generator.update_grid(
            self.data.grid_size, self.data.start, self.data.goal)
        self.stacked_widget.setCurrentIndex(2)

    def update_data(self, field_name: Literal['obstacle_map', 'start', 'goal', 'grid_size'], value: Any):
        """
        Safely updates a field in the AppData dataclass.
        """
        if hasattr(self.data, field_name):
            print(f"Updating self.data.{field_name} to {value}")
            setattr(self.data, field_name, value)
            print("Current Data:", self.data)
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")


class StartPage(QWidget):
    """
    The initial page of the application.
    """

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("Environment Setup")
        self.button_next = QPushButton("Start")

        # --- Set object names and properties for styling via QSS ---
        label.setObjectName("titleLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_next.setProperty("class", "navigation")

        layout.addWidget(label)
        layout.addWidget(self.button_next)


class GridConfigurationPage(QWidget):
    """
    A page to configure grid size and agent's starting and goal positions.
    """
    grid_size_changed = Signal(tuple)
    start_pose_changed = Signal(tuple)
    goal_pose_changed = Signal(tuple)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        # Main layout for the page
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        # Title
        title = QLabel("Grid Configuration")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Form layout for the configuration options
        form_layout = QFormLayout()
        form_layout.setSpacing(15)
        form_layout.setContentsMargins(0, 20, 0, 20)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        # --- Widgets ---
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

        # Add widgets to form layout
        form_layout.addRow(QLabel("Grid Columns:"), self.spinbox_grid_cols)
        form_layout.addRow(QLabel("Grid Rows:"), self.spinbox_grid_rows)
        form_layout.addRow(QLabel("Start X:"), self.spinbox_start_x)
        form_layout.addRow(QLabel("Start Y:"), self.spinbox_start_y)
        form_layout.addRow(QLabel("Goal X:"), self.spinbox_goal_x)
        form_layout.addRow(QLabel("Goal Y:"), self.spinbox_goal_y)

        # Add all parts to the main layout
        layout.addWidget(title)
        layout.addLayout(form_layout)
        layout.addWidget(self.button_next,
                         alignment=Qt.AlignmentFlag.AlignCenter)

        # --- Connections ---
        self.spinbox_grid_cols.valueChanged.connect(
            self._on_grid_dimensions_changed)
        self.spinbox_grid_rows.valueChanged.connect(
            self._on_grid_dimensions_changed)
        self.spinbox_start_x.valueChanged.connect(self._on_start_changed)
        self.spinbox_start_y.valueChanged.connect(self._on_start_changed)
        self.spinbox_goal_x.valueChanged.connect(self._on_goal_changed)
        self.spinbox_goal_y.valueChanged.connect(self._on_goal_changed)

        # --- Initial Population ---
        self._on_grid_dimensions_changed()

    def _on_grid_dimensions_changed(self):
        """Handles changes in either grid dimension spin box."""
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
        """Handles changes in any of the start position spin boxes."""
        x = self.spinbox_start_x.value()
        y = self.spinbox_start_y.value()
        orientation = 0
        self.start_pose_changed.emit((x, y, orientation))

    def _on_goal_changed(self):
        """Handles changes in any of the goal position spin boxes."""
        x = self.spinbox_goal_x.value()
        y = self.spinbox_goal_y.value()
        orientation = 0
        self.goal_pose_changed.emit((x, y, orientation))


class GridGenerator(QWidget):
    """
    A widget that allows the user to draw obstacles on a grid,
    while protecting the start and goal positions.
    """
    grid_confirmed = Signal(np.ndarray)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        # --- Matplotlib Configuration ---
        self.fig = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas = FigureCanvas(self.fig)
        self.ax = self.fig.add_subplot(111)

        # --- Custom Colormap for visualization ---
        # 0: Empty (White), 1: Obstacle (Red), 2: Start (Green), 3: Goal (Blue)
        colors = ['#ffffff', '#e74c3c', '#2ecc71', '#3498db']
        self.cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = BoundaryNorm(bounds, self.cmap.N)

        # --- PySide Widgets Configuration ---
        self.confirm_button = QPushButton("Confirm Grid and Finish")
        self.confirm_button.setObjectName("confirmButton")

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.confirm_button)

        # --- Event Connection ---
        self.connect_events()

        # --- Initial State ---
        self.nx = 0
        self.ny = 0
        self.start_pos = None
        self.goal_pos = None
        self.display_map = None
        self.is_drawing = False
        self.is_dragged = False
        self.last_pos = None
        self.drag_mode = 'draw'  # 'draw' or 'erase'

    def update_grid(self, grid_size: Tuple[int, int], start_pos: Tuple[int, int, int], goal_pos: Tuple[int, int, int]):
        """Resets the grid with a new configuration."""
        print(
            f"GridGenerator: Updating grid to size {grid_size} with start {start_pos} and goal {goal_pos}...")
        self.nx, self.ny = grid_size
        self.start_pos = start_pos
        self.goal_pos = goal_pos

        self.display_map = np.zeros((self.ny, self.nx), dtype=int)

        self.display_map[self.start_pos[1], self.start_pos[0]] = 2
        self.display_map[self.goal_pos[1], self.goal_pos[0]] = 3

        self.last_pos = None
        self.setup_plot()

    def setup_plot(self):
        """Configures/Refreshes the appearance of the grid, axes, and title."""
        self.ax.clear()
        self.im = self.ax.imshow(
            self.display_map, cmap=self.cmap, norm=self.norm, interpolation='nearest')
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks(np.arange(0, self.nx, 1))
        self.ax.set_yticks(np.arange(0, self.ny, 1))
        self.ax.set_title(
            "Click or drag to draw obstacles. Start (Green) and Goal (Blue) are protected.")
        self.canvas.draw()

    def connect_events(self):
        """Connects the Matplotlib mouse events and the button click."""
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.confirm_button.clicked.connect(self.on_confirm)

    def _is_protected_cell(self, x: int, y: int) -> bool:
        """Checks if a cell is the start or goal position."""
        is_start = (x == self.start_pos[0] and y == self.start_pos[1])
        is_goal = (x == self.goal_pos[0] and y == self.goal_pos[1])
        return is_start or is_goal

    def on_press(self, event):
        """Handler for mouse button press."""
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self.is_dragged = False
        self.last_pos = None

        # Determine drag mode based on the initial cell clicked
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            if self.display_map[iy, ix] == 1:
                self.drag_mode = 'erase'
            else:
                self.drag_mode = 'draw'

    def on_release(self, event):
        """Handler for mouse button release."""
        if not self.is_drawing:
            return
        # If it was a single click (no drag), toggle the cell
        if not self.is_dragged and event.inaxes == self.ax:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < self.nx and 0 <= iy < self.ny and not self._is_protected_cell(ix, iy):
                # Toggle obstacle (0 to 1, or 1 to 0)
                self.display_map[iy, ix] = 1 - self.display_map[iy, ix]
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
        self.is_drawing = False

    def on_motion(self, event):
        """Handler for mouse movement."""
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.is_dragged = True
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.last_pos:
            if not self._is_protected_cell(ix, iy):
                # Set cell value based on the mode determined on press
                if self.drag_mode == 'draw':
                    self.display_map[iy, ix] = 1
                else:  # 'erase' mode
                    self.display_map[iy, ix] = 0

                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def on_confirm(self):
        """Handler for the 'Confirm' button click."""
        print("Grid confirmed! Emitting signal...")
        # Create the final obstacle map (0s and 1s) for the algorithm.
        # Start and Goal are considered non-obstacles (value 0).
        final_obstacle_map = np.where(self.display_map == 1, 1, 0)
        self.grid_confirmed.emit(final_obstacle_map)


class EndPage(QWidget):
    """
    The final page of the application.
    """

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("Configuration Finished")
        self.button_close = QPushButton("Close Window")

        label.setObjectName("titleLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.button_close.setObjectName("closeButton")

        layout.addWidget(label)
        layout.addWidget(self.button_close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
