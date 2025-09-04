from PySide6.QtGui import QPalette, QColor, QFont
from PySide6.QtCore import Qt

PRIMARY = '#0099A8'  # institutional accent color
BG = '#111418'
FG = '#E9EEF2'
ERROR = '#D9534F'

def apply_dark_theme(app):
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG))
    palette.setColor(QPalette.WindowText, QColor(FG))
    palette.setColor(QPalette.Base, QColor('#181C22'))
    palette.setColor(QPalette.AlternateBase, QColor('#23272E'))
    palette.setColor(QPalette.Text, QColor(FG))
    palette.setColor(QPalette.Button, QColor(PRIMARY))
    palette.setColor(QPalette.ButtonText, QColor('#FFFFFF'))
    palette.setColor(QPalette.Highlight, QColor('#FFB300'))  # gold accent
    palette.setColor(QPalette.HighlightedText, QColor('#23272E'))
    app.setPalette(palette)

    # Fuente elegante y profesional (Montserrat, fallback a Segoe UI)
    app.setFont(QFont('Montserrat', 11))

    # Estilos globales para widgets
    app.setStyleSheet('''
        QWidget {
            font-family: "Montserrat", "Segoe UI", Arial, sans-serif;
            color: #E9EEF2;
            background-color: #111418;
        }
        QTabWidget::pane {
            border: 2px solid #0099A8;
            border-radius: 8px;
            background: #181C22;
        }
        QTabBar::tab {
            background: #23272E;
            color: #E9EEF2;
            padding: 10px 20px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-size: 15px;
        }
        QTabBar::tab:selected {
            background: #0099A8;
            color: #FFF;
        }
        QLabel {
            font-size: 15px;
            color: #FFB300;
        }
        QPushButton {
            background-color: #0099A8;
            color: #FFF;
            border-radius: 6px;
            padding: 8px 18px;
            font-weight: bold;
            font-size: 14px;
        }
        QPushButton:hover {
            background-color: #FFB300;
            color: #23272E;
        }
        QLineEdit, QSpinBox {
            background: #23272E;
            color: #E9EEF2;
            border: 1.5px solid #0099A8;
            border-radius: 5px;
            padding: 5px 10px;
            font-size: 14px;
        }
        QTextEdit {
            background: #181C22;
            color: #E9EEF2;
            border-radius: 8px;
            font-size: 15px;
        }
        QMessageBox {
            background: #23272E;
        }
    ''')
