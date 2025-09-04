import sys
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QFont
from ui.home_view import HomeView
from ui.theme import apply_dark_theme, PRIMARY

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Calculadora de Álgebra Lineal')
        self.resize(1100, 700)
        self.setCentralWidget(HomeView())

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Calculadora de Álgebra Lineal')
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
