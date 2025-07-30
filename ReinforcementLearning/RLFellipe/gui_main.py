# Imports
import sys
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget,
                               QVBoxLayout, QPushButton, QLabel,
                               QStackedWidget)

from dataclasses import dataclass
from typing import Optional, List, Tuple, Literal, Any

# Main Window


@dataclass
class AppData:
    obstacle_map: Optional[np.ndarray] = None
    start: Optional[Tuple[int, int]] = None
    goal: Optional[Tuple[int, int, int]] = None
    path: Optional[Tuple[int, int, int]] = None


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Reinforcement Learning GUI")

        # Adjust the size of the main window
        screen = QApplication.primaryScreen()
        available_geometry = screen.availableGeometry()
        window_widht = available_geometry.width()*0.7
        window_height = available_geometry.height()*0.7
        central_point = available_geometry.center()

        self.setGeometry(central_point.x() - window_widht / 2,
                         central_point.y() - window_height / 2,
                         window_widht, window_height)

        # Define data object
        self.data = AppData()

        # Create a stacked widget to hold the pages
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Create instances of the pages
        self.page1 = Page1(self)
        self.page2 = Page2(self)
        self.grid_generator = GridGenerator(
            self, nx=self.data['grid_size'][0], ny=self.data['grid_size'][1])

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)
        self.stacked_widget.addWidget(self.grid_generator)

        # Define starting page
        self.stacked_widget.setCurrentIndex(0)

        # Connecting Signals
        self.grid_generator.grid_confirmed.connect(
            lambda generated_map: self.update_data(
                'obstacle_map', generated_map)
        )

    def update_data(self, field_name: Literal['obstacle_map', 'start', 'goal', 'path'], value: Any):
        # Verifica se o campo existe no nosso dataclass para segurança
        if hasattr(self.data, field_name):
            print(f"Updating self.data.{field_name}...")
            # Usa setattr para definir dinamicamente o atributo
            setattr(self.data, field_name, value)
            print("Data:", self.data)
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")


class Page1(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        label = QLabel("Page 1")
        button_next = QPushButton("Go to Page 3")

        layout.addWidget(label)
        layout.addWidget(button_next)

        # Connect the button to switch to the next page (index 1)
        button_next.clicked.connect(
            lambda: self.main_window.stacked_widget.setCurrentIndex(2))


class Page2(QWidget):
    def __init__(self, main_window: MainWindow):
        super().__init__(main_window)
        self.main_window = main_window

        layout = QVBoxLayout(self)
        label = QLabel("Page 2")
        button_close = QPushButton("Close")

        layout.addWidget(label)
        layout.addWidget(button_close)

        # Connect the button to switch to the next page (index 1)
        button_close.clicked.connect(
            lambda: self.main_window.close())


class GridGenerator(QWidget):
    """
    """
    grid_confirmed = Signal(List[str, np.ndarray])

    def __init__(self, main_window: MainWindow, nx: int = 10, ny: int = 10,):
        super().__init__(main_window)
        self.nx = nx
        self.ny = ny

        # O mapa de obstáculos agora é um atributo da classe
        self.obstacle_map = np.zeros((self.ny, self.nx), dtype=int)

        # Atributos para controlar o estado do desenho
        self.is_drawing = False
        self.is_dragged = False
        self.last_pos = None

        # --- Configuração do Matplotlib ---
        # 1. Cria a figura e o eixo do Matplotlib
        fig = Figure(figsize=(8, 8))
        self.ax = fig.add_subplot(111)
        self.im = self.ax.imshow(
            self.obstacle_map, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')

        # 2. Cria o canvas do PySide para exibir a figura
        self.canvas = FigureCanvas(fig)

        # --- Configuração dos Widgets PySide ---
        # Botão para confirmar a seleção do grid
        self.confirm_button = QPushButton("Confirmar Grid")

        # Layout principal para o widget
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)
        layout.addWidget(self.confirm_button)

        # --- Conexão dos Eventos ---
        self.setup_plot()
        self.connect_events()

    def setup_plot(self):
        """Configura a aparência do grid, eixos e título."""
        self.ax.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle='-', linewidth=1)
        self.ax.tick_params(which="minor", size=0)
        self.ax.set_xticks(np.arange(0, self.nx, 1))
        self.ax.set_yticks(np.arange(0, self.ny, 1))
        self.ax.set_title("Clique ou Arraste para Desenhar Obstáculos")
        self.canvas.draw()

    def connect_events(self):
        """Conecta os eventos do mouse do Matplotlib e o clique do botão."""
        # Eventos do Matplotlib
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        # Evento do botão do PySide6
        self.confirm_button.clicked.connect(self.on_confirm)

    def on_press(self, event):
        """Handler para pressionar o botão do mouse."""
        if event.inaxes != self.ax:
            return
        self.is_drawing = True
        self.is_dragged = False
        self.last_pos = None

    def on_release(self, event):
        """Handler para soltar o botão do mouse."""
        if not self.is_drawing:
            return

        # Lógica para clique simples (sem arrastar)
        if not self.is_dragged and event.inaxes == self.ax:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
                self.im.set_data(self.obstacle_map)
                self.canvas.draw_idle()

        self.is_drawing = False

    def on_motion(self, event):
        """Handler para movimento do mouse."""
        if not self.is_drawing or event.inaxes != self.ax:
            return
        self.is_dragged = True

        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.last_pos:
            self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
            self.im.set_data(self.obstacle_map)
            self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def on_confirm(self):
        """Handler para o clique do botão 'Confirmar'."""
        print("Grid confirmado! Emitindo sinal...")
        # Emite o sinal com o mapa de obstáculos como payload
        self.grid_confirmed.emit(self.obstacle_map)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
