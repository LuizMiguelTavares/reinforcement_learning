import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                               QWidget, QPushButton, QVBoxLayout, QLabel)

# --- Página 1 ---


class Pagina1(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout(self)
        label = QLabel("Esta é a Página 1")
        botao_proximo = QPushButton("Ir para a Página 2")

        layout.addWidget(label)
        layout.addWidget(botao_proximo)

        # Conecta o botão para ir para a próxima página (índice 1)
        botao_proximo.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1))

# --- Página 2 ---


class Pagina2(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout(self)
        label = QLabel("Esta é a Página 2")
        botao_anterior = QPushButton("Voltar para a Página 1")
        botao_proximo = QPushButton("Ir para a Página 3")

        layout.addWidget(label)
        layout.addWidget(botao_anterior)
        layout.addWidget(botao_proximo)

        # Conecta os botões para navegar
        botao_anterior.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(0))
        botao_proximo.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(2))

# --- Página 3 ---


class Pagina3(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        layout = QVBoxLayout(self)
        label = QLabel("Esta é a Página 3 - Fim!")
        botao_voltar = QPushButton("Voltar para a Página 2")

        layout.addWidget(label)
        layout.addWidget(botao_voltar)

        # Conecta o botão para voltar
        botao_voltar.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1))


# --- Janela Principal ---
class JanelaPrincipal(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Navegação entre Páginas com QStackedWidget")

        # Cria o QStackedWidget
        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        # Cria e adiciona as páginas ao QStackedWidget
        self.pagina1 = Pagina1(self.stacked_widget)
        self.pagina2 = Pagina2(self.stacked_widget)
        self.pagina3 = Pagina3(self.stacked_widget)

        self.stacked_widget.addWidget(self.pagina1)
        self.stacked_widget.addWidget(self.pagina2)
        self.stacked_widget.addWidget(self.pagina3)

        # Define a página inicial
        self.stacked_widget.setCurrentIndex(0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    janela = JanelaPrincipal()
    janela.resize(400, 200)
    janela.show()
    sys.exit(app.exec())
