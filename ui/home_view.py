from PySide6.QtWidgets import QWidget, QLabel, QTabWidget, QVBoxLayout
from .matrix_widgets import AugmentedSystemWidget

class HomeView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        tabs = QTabWidget()
        # This version ships the key requirement: Gauss–Jordan with board-like steps.
        tabs.addTab(AugmentedSystemWidget(), 'Sistemas de ecuaciones (Gauss–Jordan)')
        v.addWidget(tabs)
