from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox, QScrollArea
)
from PySide6.QtGui import QFont, QDoubleValidator, QRegularExpressionValidator
from PySide6.QtCore import QRegularExpression

from core import errores


class ErrorsWidget(QWidget):
    """Interfaz gráfica que expone las funciones didácticas de `core.errores`.

    Ofrece:
    - Descomposición en base 10 (enteros y fracciones) y base 2 (bits).
    - Explicaciones de tipos de error y ejemplos de punto flotante.
    - Demo opcional con NumPy.
    - Ejercicio principal: calcular E_a, E_r y propagación para f(x)=sin(x)+x^2.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self):
        v = QVBoxLayout(self)

        # Contenedor de resultados (secciones colapsables)
        self.results_container = QVBoxLayout()
        results_widget = QWidget()
        results_widget.setLayout(self.results_container)
        self.results_area = QScrollArea()
        self.results_area.setWidgetResizable(True)
        self.results_area.setWidget(results_widget)

        # Salida común (registro simple)
        self.output = QTextEdit()
        self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 11))

        # Base 10
        gb10 = QGroupBox('Descomposición Base 10')
        h10 = QHBoxLayout()
        self.input_b10 = QLineEdit()
        self.input_b10.setPlaceholderText('Ej: 84506 o 12.345')
        # Aceptar decimales: validador float
        dv = QDoubleValidator()
        dv.setNotation(QDoubleValidator.StandardNotation)
        self.input_b10.setValidator(dv)
        # validar visualmente al escribir
        self.input_b10.textChanged.connect(lambda _: self._apply_validation_style(self.input_b10))
        btn_b10 = QPushButton('Descomponer (base 10)')
        btn_b10.clicked.connect(self._on_decompose_base10)
        h10.addWidget(QLabel('Número:'))
        h10.addWidget(self.input_b10)
        h10.addWidget(btn_b10)
        gb10.setLayout(h10)

        # Base 2
        gb2 = QGroupBox('Descomposición Base 2')
        h2 = QHBoxLayout()
        self.input_b2 = QLineEdit()
        self.input_b2.setPlaceholderText('Ej: 1111001')
        # Validador para bits (solo 0 y 1)
        re = QRegularExpression('^[01]+$')
        rev = QRegularExpressionValidator(re)
        self.input_b2.setValidator(rev)
        # validar visualmente al escribir
        self.input_b2.textChanged.connect(lambda _: self._apply_validation_style(self.input_b2))
        btn_b2 = QPushButton('Descomponer (base 2)')
        btn_b2.clicked.connect(self._on_decompose_base2)
        h2.addWidget(QLabel('Bits:'))
        h2.addWidget(self.input_b2)
        h2.addWidget(btn_b2)
        gb2.setLayout(h2)

        # Explicaciones y ejemplo flotante
        gb_errors = QGroupBox('Conceptos de error y ejemplos')
        h_err = QHBoxLayout()
        btn_explain = QPushButton('Mostrar explicaciones de error')
        btn_explain.clicked.connect(self._on_explain_errors)
        btn_float = QPushButton('Ejemplo punto flotante (0.1+0.2)')
        btn_float.clicked.connect(self._on_float_examples)
        btn_numpy = QPushButton('Demo NumPy')
        btn_numpy.clicked.connect(self._on_numpy_demo)
        h_err.addWidget(btn_explain)
        h_err.addWidget(btn_float)
        h_err.addWidget(btn_numpy)
        gb_errors.setLayout(h_err)

        # Ejercicio principal
        gb_main = QGroupBox('Ejercicio: errores y propagación para f(x)=sin(x)+x^2')
        hmain = QHBoxLayout()
        self.input_xv = QLineEdit()
        self.input_xv.setPlaceholderText('x verdadero (ej: 1.234)')
        self.input_xa = QLineEdit()
        self.input_xa.setPlaceholderText('x aproximado (ej: 1.230)')
        # Validadores numéricos para xv/xa
        dv2 = QDoubleValidator()
        dv2.setNotation(QDoubleValidator.StandardNotation)
        self.input_xv.setValidator(dv2)
        self.input_xa.setValidator(dv2)
        # validar visualmente al escribir
        self.input_xv.textChanged.connect(lambda _: self._apply_validation_style(self.input_xv))
        self.input_xa.textChanged.connect(lambda _: self._apply_validation_style(self.input_xa))
        btn_calc = QPushButton('Calcular errores y propagación')
        btn_calc.clicked.connect(self._on_calc_errors)
        hmain.addWidget(QLabel('x_v:'))
        hmain.addWidget(self.input_xv)
        hmain.addWidget(QLabel('x_a:'))
        hmain.addWidget(self.input_xa)
        hmain.addWidget(btn_calc)
        gb_main.setLayout(hmain)

        # Botones adicionales: limpiar
        hb = QHBoxLayout()
        btn_clear = QPushButton('Limpiar salida')
        btn_clear.clicked.connect(self.output.clear)
        # botón para limpiar las secciones de resultados
        btn_clear_sections = QPushButton('Limpiar resultados')
        btn_clear_sections.clicked.connect(self._clear_results)
        hb.addStretch()
        hb.addWidget(btn_clear)
        hb.addWidget(btn_clear_sections)

        # Añadir todo al layout
        v.addWidget(gb10)
        v.addWidget(gb2)
        v.addWidget(gb_errors)
        v.addWidget(gb_main)
        v.addLayout(hb)
        v.addWidget(QLabel('Resultados (secciones desplegables):'))
        v.addWidget(self.results_area, stretch=1)
        v.addWidget(QLabel('Registro:'))
        v.addWidget(self.output, stretch=1)

    def _on_decompose_base10(self):
        s = self.input_b10.text().strip()
        try:
            # pasar la cadena tal cual a la función para soportar fracciones
            res = errores.decompose_base10(s)
        except Exception:
            res = 'Entrada inválida para base 10. Ingrese un entero, por ejemplo 84506.'
        self._show_section('Descomposición base 10', res)
        self.output.append('Descomposición base10 ejecutada.')

    def _on_decompose_base2(self):
        s = self.input_b2.text().strip()
        res = errores.decompose_base2(s)
        self._show_section('Descomposición base 2', res)
        self.output.append('Descomposición base2 ejecutada.')

    def _on_explain_errors(self):
        res = errores.explain_errors()
        self._show_section('Explicación: tipos de error', res)
        self.output.append('Explicaciones mostradas.')

    def _on_float_examples(self):
        res = errores.float_examples()
        self._show_section('Ejemplo punto flotante', res)
        self.output.append('Ejemplo punto flotante mostrado.')

    def _on_numpy_demo(self):
        res = errores.numpy_demo()
        self._show_section('Demo NumPy', res)
        self.output.append('Demo NumPy ejecutada.')

    def _on_calc_errors(self):
        s_v = self.input_xv.text().strip()
        s_a = self.input_xa.text().strip()
        try:
            xv = float(s_v)
            xa = float(s_a)
            table = errores.format_table_error(xv, xa)
            self._show_section('Errores y propagación', table)
            self.output.append('Cálculo de errores ejecutado.')
        except Exception:
            self.output.append('Entrada inválida para x_v o x_a. Use valores numéricos (ej: 1.23).')

    def _show_section(self, title: str, text: str):
        """Crea una sección colapsable (QGroupBox checkable) con el resultado y la añade
        al contenedor de resultados para una visual más ordenada."""
        gb = QGroupBox(title)
        gb.setCheckable(True)
        gb.setChecked(True)
        inner = QVBoxLayout()
        te = QTextEdit()
        te.setReadOnly(True)
        te.setFont(QFont('Consolas', 10))
        te.setPlainText(text)
        inner.addWidget(te)
        gb.setLayout(inner)
        self.results_container.addWidget(gb)

    def _clear_results(self):
        """Remueve todas las secciones generadas en results_container y libera los widgets."""
        layout = self.results_container
        # Mientras haya elementos, sacarlos y liberar el widget asociado
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

    def _apply_validation_style(self, widget: QLineEdit):
        """Aplica estilo visual según el validador del QLineEdit.
        Resalta en rojo si inválido y no vacío; vuelve a normal si válido o vacío.
        """
        validator = widget.validator()
        text = widget.text()
        if validator is None:
            widget.setStyleSheet("")
            return
        # QValidator.validate devuelve (state, pos)
        state = validator.validate(text, 0)[0]
        from PySide6.QtGui import QValidator
        if text and state != QValidator.Acceptable:
            widget.setStyleSheet('background-color: #fff0f0; border: 1px solid #ff4d4d;')
        else:
            widget.setStyleSheet('')
