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
    palette.setColor(QPalette.Base, QColor('#0C0F12'))
    palette.setColor(QPalette.AlternateBase, QColor('#12161B'))
    palette.setColor(QPalette.Text, QColor(FG))
    palette.setColor(QPalette.Button, QColor('#151A20'))
    palette.setColor(QPalette.ButtonText, QColor(FG))
    palette.setColor(QPalette.Highlight, QColor(PRIMARY))
    palette.setColor(QPalette.HighlightedText, QColor('#FFFFFF'))
    app.setPalette(palette)
