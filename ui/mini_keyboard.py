from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QApplication, QLineEdit, QTextEdit, QGridLayout
from PySide6.QtGui import QFont
class _TextTargetRegistry:
    """Registra el último widget de texto enfocado para inserciones desde el teclado.

    Se actualiza vía QApplication.focusChanged desde main.py.
    """
    def __init__(self):
        self._widget = None

    def update(self, old, new):
        from PySide6.QtWidgets import QLineEdit, QTextEdit
        if isinstance(new, (QLineEdit, QTextEdit)):
            self._widget = new

    def current(self):
        return self._widget


# Instancia global simple
TEXT_TARGET_REGISTRY = _TextTargetRegistry()



class MiniKeyboardPanel(QWidget):
    """Panel embebible con atajos para exponentes y funciones comunes.

    Inserta en el widget con foco (QLineEdit o QTextEdit) el texto del botón.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        grid = QGridLayout()
        grid.setSpacing(6)

        def add_btn(row: int, col: int, label: str, text_to_insert: str | None = None):
            btn = QPushButton(label)
            # Expandir para ocupar celda completa
            from PySide6.QtWidgets import QSizePolicy
            btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            # Fuente grande y con superíndices para buena legibilidad en Windows
            btn.setFont(QFont('Segoe UI Symbol', 14))
            btn.clicked.connect(lambda: self._insert_text(text_to_insert or label))
            grid.addWidget(btn, row, col)

        # Superscripts (forma visual como en la imagen)
        supers = ['¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹','⁰']
        for idx, s in enumerate(supers):
            add_btn(0, idx, s, s)

        # Operadores/funciones útiles
        add_btn(1, 0, '^', '^')
        add_btn(1, 1, '(', '(')
        add_btn(1, 2, ')', ')')
        add_btn(1, 3, 'exp(', 'exp(')
        add_btn(1, 4, 'log(', 'log(')
        add_btn(1, 5, 'sqrt(', 'sqrt(')
        add_btn(1, 6, 'sin(', 'sin(')
        add_btn(1, 7, 'cos(', 'cos(')
        add_btn(1, 8, 'tan(', 'tan(')
        add_btn(1, 9, 'e', 'e')
        add_btn(1, 10, 'π', 'pi')

        add_btn(2, 0, 'x²', 'x²')
        add_btn(2, 1, 'x³', 'x³')
        add_btn(2, 2, '10^x', '10^x')
        add_btn(2, 3, 'e^x', 'e^x')
        add_btn(2, 4, '( )', '()')
        add_btn(2, 5, 'x^( )', 'x^')
        add_btn(2, 6, '·', '*')
        add_btn(2, 7, '+', '+')
        add_btn(2, 8, '−', '-')
        add_btn(2, 9, '/', '/')

        # Cuarta fila para completar el panel
        add_btn(3, 0, 'x', 'x')
        add_btn(3, 1, 'y', 'y')
        add_btn(3, 2, 'z', 'z')
        add_btn(3, 3, ' , ', ', ')
        add_btn(3, 4, ' ; ', '; ')
        add_btn(3, 5, '←', '')
        add_btn(3, 6, '0', '0')
        add_btn(3, 7, '1', '1')
        add_btn(3, 8, '2', '2')
        add_btn(3, 9, '3', '3')

        # Estilo para asegurar contraste del texto
        self.setStyleSheet(
            "QPushButton { color: #ffffff; background-color: #0aa0a0; border: 0; border-radius: 6px; padding: 4px 8px; }\n"
            "QPushButton:hover { background-color: #0bb0b0; }"
        )

        root.addLayout(grid)

    def _insert_text(self, text: str):
        # Priorizar el último widget de texto registrado
        w = TEXT_TARGET_REGISTRY.current()
        if w is None:
            w = QApplication.focusWidget()
        if isinstance(w, QLineEdit):
            pos = w.cursorPosition()
            s = w.text()
            w.setText(s[:pos] + text + s[pos:])
            w.setCursorPosition(pos + len(text))
        elif isinstance(w, QTextEdit):
            cursor = w.textCursor()
            cursor.insertText(text)
            w.setTextCursor(cursor)
        # Si el foco no está en un campo de texto, no hace nada
