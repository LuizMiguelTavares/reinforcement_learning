# Imports
import sys
from PySide6.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                               QWidget, QPushButton, QVBoxLayout, QLabel)

# Main Window


class MainWindow(QMainWindow):
    def __init__(self):
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

        # Create a stacked widget to hold the pages
        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        # Create instances of the pages
        self.page1 = Page1(self)
        self.page2 = Page2(self)

        # Add pages to the stacked widget
        self.stacked_widget.addWidget(self.page1)
        self.stacked_widget.addWidget(self.page2)

        # Define starting page
        self.stacked_widget.setCurrentIndex(0)


class Page1(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        label = QLabel("Page 1")
        button_next = QPushButton("Go to Page 2")

        layout.addWidget(label)
        layout.addWidget(button_next)

        # Connect the button to switch to the next page (index 1)
        button_next.clicked.connect(
            lambda: self.main_window.stacked_widget.setCurrentIndex(1))


class Page2(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__()
        self.main_window = main_window

        layout = QVBoxLayout(self)
        label = QLabel("Page 2")
        button_close = QPushButton("Close")

        layout.addWidget(label)
        layout.addWidget(button_close)

        # Connect the button to switch to the next page (index 1)
        button_close.clicked.connect(
            lambda: self.main_window.close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()

    sys.exit(app.exec())
