from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QSlider, QStackedWidget, QSizePolicy
)
import sys
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

# Certifica-se de que o backend do Matplotlib é o QtAgg
matplotlib.use('QtAgg')


# --- Adaptação do seu grid_generator ---
# Em vez de uma função que bloqueia com plt.show(), criamos uma classe de Widget
# que pode ser embutida na nossa aplicação PySide6.

class InteractiveGridCanvas(FigureCanvas):
    """Uma tela Matplotlib interativa para gerar a grade de obstáculos."""

    def __init__(self, nx=20, ny=20, parent=None, width=5, height=5, dpi=100):
        self.nx = nx
        self.ny = ny
        self.obstacle_map = np.zeros((self.ny, self.nx), dtype=int)

        # Cria a figura Matplotlib
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super().__init__(fig)  # Chama o construtor da classe pai

        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Expanding)
        self.updateGeometry()

        # Lógica de estado da interação
        self.state = {'drawing': False, 'dragged': False, 'last_pos': None}
        self.im = None
        self.plot()

        # Conecta os eventos do mouse
        self.mpl_connect('button_press_event', self.on_press)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_motion)

    def plot(self):
        """Desenha ou atualiza a grade na tela."""
        if self.im is None:
            self.im = self.axes.imshow(
                self.obstacle_map, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            # Configuração visual do grid
            self.axes.set_xticks(np.arange(-.5, self.nx, 1), minor=True)
            self.axes.set_yticks(np.arange(-.5, self.ny, 1), minor=True)
            self.axes.grid(which="minor", color="black",
                           linestyle='-', linewidth=1)
            self.axes.tick_params(which="minor", size=0)
            self.axes.set_xticks(np.arange(0, self.nx, 1))
            self.axes.set_yticks(np.arange(0, self.ny, 1))
            self.axes.set_title("Clique ou Arraste para Desenhar Obstáculos")
        else:
            self.im.set_data(self.obstacle_map)

        self.draw()  # Redesenha o canvas

    # --- Handlers de evento (lógica do seu código original) ---
    def on_press(self, event):
        if event.inaxes != self.axes:
            return
        self.state['drawing'] = True
        self.state['dragged'] = False
        self.state['last_pos'] = None

    def on_release(self, event):
        if event.inaxes != self.axes and self.state['drawing']:
            self.state['drawing'] = False
            return

        if not self.state['dragged']:
            ix, iy = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= ix < self.nx and 0 <= iy < self.ny:
                self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
                self.plot()
        self.state['drawing'] = False

    def on_motion(self, event):
        if not self.state['drawing'] or event.inaxes != self.axes:
            return
        self.state['dragged'] = True
        ix, iy = int(round(event.xdata)), int(round(event.ydata))
        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.state['last_pos']:
            self.obstacle_map[iy, ix] = 1 - self.obstacle_map[iy, ix]
            self.plot()
            self.state['last_pos'] = (ix, iy)


# --- Tela Principal da Aplicação ---
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Coletor de Dados com PySide6")
        self.setGeometry(100, 100, 500, 600)  # Posição e tamanho da janela

        # Dicionário para armazenar todos os dados coletados
        self.collected_data = {
            "nome_simulacao": "",
            "fator_risco": 50.0,
            "grid_map": None
        }

        # Widget central que conterá as telas
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Cria as telas e adiciona ao "baralho"
        self.screen1 = self.create_screen1()
        self.screen2 = self.create_screen2()
        self.screen3 = self.create_screen3()

        self.stacked_widget.addWidget(self.screen1)
        self.stacked_widget.addWidget(self.screen2)
        self.stacked_widget.addWidget(self.screen3)

    def go_to_screen(self, index):
        """Muda a tela visível."""
        # Se estivermos indo para a tela 3, atualiza o resumo
        if index == 2:
            self.update_summary_screen()
        self.stacked_widget.setCurrentIndex(index)

    def create_screen1(self):
        """Cria a primeira tela para coletar parâmetros."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        # Campo de Texto
        layout.addWidget(QLabel("Nome da Simulação:"))
        self.name_input = QLineEdit()
        layout.addWidget(self.name_input)

        # Slider
        layout.addSpacing(20)
        layout.addWidget(QLabel("Fator de Risco:"))
        self.slider_label = QLabel(f"{self.collected_data['fator_risco']:.1f}")
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(int(self.collected_data['fator_risco']))
        slider.valueChanged.connect(
            lambda val: self.slider_label.setText(f"{val:.1f}"))
        self.risk_slider = slider

        slider_layout = QHBoxLayout()
        slider_layout.addWidget(slider)
        slider_layout.addWidget(self.slider_label)
        layout.addLayout(slider_layout)

        # Botão para avançar
        layout.addSpacing(40)
        next_button = QPushButton("Próximo: Gerar Grid")
        next_button.clicked.connect(self.process_screen1)
        layout.addWidget(next_button)

        return widget

    @Slot()
    def process_screen1(self):
        """Salva os dados da tela 1 e avança."""
        self.collected_data["nome_simulacao"] = self.name_input.text()
        self.collected_data["fator_risco"] = float(self.risk_slider.value())
        print("Dados da Tela 1 salvos:", self.collected_data)
        self.go_to_screen(1)

    def create_screen2(self):
        """Cria a tela com o gerador de grid interativo."""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        layout.addWidget(
            QLabel("<h3>Passo 2: Desenhe o Mapa de Obstáculos</h3>"))
        self.grid_canvas = InteractiveGridCanvas(nx=20, ny=20)
        layout.addWidget(self.grid_canvas)

        next_button = QPushButton("Próximo: Ver Resumo")
        next_button.clicked.connect(self.process_screen2)
        layout.addWidget(next_button)

        return widget

    @Slot()
    def process_screen2(self):
        """Salva a grade e avança para a tela de resumo."""
        self.collected_data["grid_map"] = self.grid_canvas.obstacle_map
        print("Mapa de obstáculos salvo.")
        self.go_to_screen(2)

    def create_screen3(self):
        """Cria a tela final de resumo e execução."""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        layout.addWidget(QLabel("<h3>Passo 3: Resumo e Execução</h3>"))

        self.summary_name = QLabel()
        self.summary_risk = QLabel()
        self.summary_grid = QLabel()

        layout.addWidget(self.summary_name)
        layout.addWidget(self.summary_risk)
        layout.addWidget(self.summary_grid)
        layout.addSpacing(40)

        run_button = QPushButton("EXECUTAR CÓDIGO FINAL")
        run_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;")
        run_button.clicked.connect(self.run_final_code)
        layout.addWidget(run_button)

        return widget

    def update_summary_screen(self):
        """Atualiza os textos na tela de resumo."""
        nome = self.collected_data['nome_simulacao']
        risco = self.collected_data['fator_risco']
        grid_shape = self.collected_data['grid_map'].shape if self.collected_data['grid_map'] is not None else "N/A"

        self.summary_name.setText(f"<b>Nome:</b> {nome}")
        self.summary_risk.setText(f"<b>Fator de Risco:</b> {risco}")
        self.summary_grid.setText(f"<b>Dimensões do Grid:</b> {grid_shape}")

    @Slot()
    def run_final_code(self):
        """Função final que recebe todos os dados."""
        print("\n" + "="*40)
        print("EXECUTANDO O CÓDIGO FINAL")
        print("Dados recebidos:")
        print(f"  - Nome: {self.collected_data['nome_simulacao']}")
        print(f"  - Fator de Risco: {self.collected_data['fator_risco']}")
        print(
            f"  - Mapa de Obstáculos (numpy array):\n{self.collected_data['grid_map']}")
        print("="*40 + "\n")

        # Aqui você chamaria sua outra função/script
        # por exemplo: outro_script.main(self.collected_data)

        # Fecha a aplicação
        self.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
