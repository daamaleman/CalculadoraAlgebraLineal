import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget
from PySide6.QtGui import QFont, QAction
from PySide6.QtCore import Qt
from ui.home_view import HomeView
from ui.theme import apply_dark_theme, PRIMARY
from ui.mini_keyboard import MiniKeyboardPanel, TEXT_TARGET_REGISTRY

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Calculadora de Álgebra Lineal')
        self.resize(1100, 700)
        self.setCentralWidget(HomeView())
        # Overlay cuadrado al frente de la app (dentro de la ventana)
        # Overlay cuadrado con layout para evitar recortes y superposiciones
        self._kb_overlay = QWidget(self)
        self._kb_overlay.setObjectName('kbOverlay')
        self._kb_overlay.setWindowFlags(Qt.FramelessWindowHint)
        self._kb_overlay.hide()
        self._kb_overlay.resize(360, 360)
        from PySide6.QtWidgets import QPushButton, QLabel, QVBoxLayout, QHBoxLayout
        ovl_root = QVBoxLayout(self._kb_overlay)
        ovl_root.setContentsMargins(8, 8, 8, 8)
        # Header
        header = QHBoxLayout()
        title = QLabel('Teclado rápido', self._kb_overlay)
        title.setStyleSheet('color: #ffd36b; font-weight: bold;')
        header.addWidget(title)
        header.addStretch()
        close_btn = QPushButton('×', self._kb_overlay)
        close_btn.setFixedSize(28, 24)
        close_btn.clicked.connect(lambda: self._kb_overlay.hide())
        header.addWidget(close_btn)
        ovl_root.addLayout(header)
        # Panel keyboard fills remaining space
        self._kb_panel = MiniKeyboardPanel(self._kb_overlay)
        ovl_root.addWidget(self._kb_panel, 1)
        # Estilo overlay sólido para mejor contraste
        self._kb_overlay.setStyleSheet('#kbOverlay { background-color: #1f1f22; border: 1px solid #333; border-radius: 8px; }')
        # Menú para mostrar/ocultar (F2)
        tools = self.menuBar().addMenu('Herramientas')
        act_toggle = QAction('Mostrar/Ocultar Teclado rápido', self)
        act_toggle.setShortcut('F2')
        act_toggle.triggered.connect(self._toggle_keyboard)
        tools.addAction(act_toggle)
        # Registrar cambios de foco para que el teclado sepa dónde insertar
        QApplication.instance().focusChanged.connect(TEXT_TARGET_REGISTRY.update)

    def _toggle_keyboard(self):
        if self._kb_overlay.isVisible():
            self._kb_overlay.hide()
        else:
            # Posicionar en esquina inferior derecha de la ventana principal
            geo = self.geometry()
            w, h = self._kb_overlay.width(), self._kb_overlay.height()
            x = geo.width() - w - 12
            y = geo.height() - h - 12
            self._kb_overlay.move(max(0, x), max(0, y))
            self._kb_overlay.show()
            self._kb_overlay.raise_()

def main():
    app = QApplication(sys.argv)
    app.setApplicationName('Calculadora de Álgebra Lineal')
    apply_dark_theme(app)
    w = MainWindow()
    w.show()
    try:
        exit_code = app.exec()
    except KeyboardInterrupt:
        # Cierre limpio al presionar Ctrl+C en la terminal
        exit_code = 0
    sys.exit(exit_code)

if __name__ == '__main__':
    main()
