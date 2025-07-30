import sys
from typing import Optional, List, Tuple, Literal, Any
from dataclasses import dataclass

# Use PySide6 as specified in the original code
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QPushButton, QLabel,
                               QStackedWidget)
from PySide6.QtCore import Signal, Qt
from PySide6.QtGui import QScreen

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
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
    start: Optional[Tuple[int, int, int]] = None
    goal: Optional[Tuple[int, int, int]] = None


class MainWindow(QMainWindow):
    """
    The main window of the application. It manages the pages (widgets)
    and the shared application data.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Reinforcement Learning GUI")

        # Adjust the size of the main window to be 70% of screen size
        screen = QScreen.availableGeometry(QApplication.primaryScreen())
        window_width = screen.width() * 0.7
        window_height = screen.height() * 0.7
        self.setGeometry(
            (screen.width() - window_width) / 2,
            (screen.height() - window_height) / 2,
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

        # Create a stacked widget to hold the pages
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Create instances of the pages
        self.start_page = StartPage(self)
        self.grid_generator = GridGenerator(self, self.data.grid_size)
        self.end_page = EndPage(self)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.grid_generator)
        self.stacked_widget.addWidget(self.end_page)

        # Define starting page
        self.stacked_widget.setCurrentIndex(0)

        # Connecting Signals
        self.grid_generator.grid_confirmed.connect(
            lambda generated_map: self.update_data(
                'obstacle_map', generated_map)
        )

    def update_data(self, field_name: Literal['obstacle_map', 'start', 'goal', 'path'], value: Any):
        """
        Safely updates a field in the AppData dataclass.
        """
        if hasattr(self.data, field_name):
            print(f"Updating self.data.{field_name}...")
            setattr(self.data, field_name, value)
            print("Data:", self.data)
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")


class StartPage(QWidget):
    """
    The initial page of the application.
    """

    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("Environment Setup")
        button_next = QPushButton("Start")

        # --- Set object names and properties for styling via QSS ---
        label.setObjectName("titleLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_next.setProperty("class", "navigation")

        layout.addWidget(label)
        layout.addWidget(button_next)

        # Connect the button to switch to the grid generator page (index 1)
        button_next.clicked.connect(
            lambda: self.main_window.stacked_widget.setCurrentIndex(1))


class GridGenerator(QWidget):
    """
    A widget that allows the user to draw obstacles on a grid.
    """
    grid_confirmed = Signal(np.ndarray)

    def __init__(self, main_window: MainWindow, grid_size: Tuple[int, int] = ()):
        super().__init__(main_window)
        self.main_window = main_window
        self.nx, self.ny = grid_size
        self.obstacle_map = np.zeros((self.ny, self.nx), dtype=int)
        self.is_drawing = False
        self.is_dragged = False
        self.last_pos = None

        # --- Matplotlib Configuration ---
        fig = Figure(figsize=(8, 8), tight_layout=True)
        self.ax = fig.add_subplot(111)
        self.im = self.ax.imshow(
            self.obstacle_map, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        self.canvas = FigureCanvas(fig)

        # --- PySide Widgets Configuration ---
        self.confirm_button = QPushButton("Confirm Grid")
        # Set object name for specific styling
        self.confirm_button.setObjectName("confirmButton")

        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.confirm_button)

        # --- Event Connection ---
        self.setup_plot()
        self.connect_events()

    def setup_plot(self):
        """Configures the appearance of the grid, axes, and title."""
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks(np.arange(0, self.nx, 1))
        self.ax.set_yticks(np.arange(0, self.ny, 1))
        self.ax.set_title("Click or drag to draw obstacles")
        self.canvas.draw()

    def connect_events(self):
        """Connects the Matplotlib mouse events and the button click."""
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.confirm_button.clicked.connect(self.on_confirm)

    def on_press(self, event):
        """Handler for mouse button press."""
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self.is_dragged = False
        self.last_pos = None

    def on_release(self, event):
        """Handler for mouse button release."""
        if not self.is_drawing:
            return
        if not self.is_dragged and event.inaxes == self.ax:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
                self.im.set_data(self.obstacle_map)
                self.canvas.draw_idle()
        self.is_drawing = False

    def on_motion(self, event):
        """Handler for mouse movement."""
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.is_dragged = True
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.last_pos:
            # --- FIX: Restore toggle functionality for drawing and erasing ---
            self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
            self.im.set_data(self.obstacle_map)
            self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def on_confirm(self):
        """Handler for the 'Confirm' button click."""
        print("Grid confirmed! Emitting signal...")
        self.grid_confirmed.emit(self.obstacle_map)
        self.main_window.stacked_widget.setCurrentIndex(2)


class EndPage(QWidget):
    """
    The final page of the application.
    """

    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("Configuration Finished")
        button_close = QPushButton("Close Window")

        # --- Set object names for styling ---
        label.setObjectName("titleLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_close.setObjectName("closeButton")

        layout.addWidget(label)
        layout.addWidget(button_close)

        # Connect the button to close the application
        button_close.clicked.connect(self.main_window.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
