from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox
)
from PySide6.QtGui import QFont, QDoubleValidator

from core.numericos import parse_function, biseccion, regla_falsa, format_iterations_table


class NumericalMethodsWidget(QWidget):
    """Vista para ejecutar Bisección y Regla Falsa.

    Permite ingresar f(x), intervalo [a,b], tolerancia y ejecutar cada método
    mostrando tabla de iteraciones y resumen final.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._build()

    def _build(self):
        v = QVBoxLayout(self)

        # Grupo de entrada
        gb = QGroupBox('Parámetros')
        h = QHBoxLayout()
        self.f_edit = QLineEdit(); self.f_edit.setPlaceholderText("f(x) = x^2 - 4; usa sin(x), cos(x), exp(x), etc.")
        self.a_edit = QLineEdit(); self.a_edit.setPlaceholderText('a');
        self.b_edit = QLineEdit(); self.b_edit.setPlaceholderText('b');
        self.tol_edit = QLineEdit(); self.tol_edit.setPlaceholderText('tolerancia (ej: 0.0001)');
        # Validadores
        dv = QDoubleValidator(); dv.setNotation(QDoubleValidator.StandardNotation)
        self.a_edit.setValidator(dv); self.b_edit.setValidator(dv); self.tol_edit.setValidator(dv)

        h.addWidget(QLabel('f(x):'))
        h.addWidget(self.f_edit, stretch=2)
        h.addWidget(QLabel('a:'))
        h.addWidget(self.a_edit)
        h.addWidget(QLabel('b:'))
        h.addWidget(self.b_edit)
        h.addWidget(QLabel('tol:'))
        h.addWidget(self.tol_edit)
        gb.setLayout(h)

        # Botones
        hb = QHBoxLayout()
        btn_bis = QPushButton('Ejecutar Bisección')
        btn_rf = QPushButton('Ejecutar Regla Falsa')
        btn_bis.clicked.connect(self._run_biseccion)
        btn_rf.clicked.connect(self._run_regla_falsa)
        hb.addWidget(btn_bis)
        hb.addWidget(btn_rf)

        # Salida
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 11))

        v.addWidget(gb)
        v.addLayout(hb)
        v.addWidget(self.output)

    def _read_inputs(self):
        expr = self.f_edit.text().strip()
        a_txt = self.a_edit.text().strip()
        b_txt = self.b_edit.text().strip()
        tol_txt = self.tol_edit.text().strip()
        if not expr or not a_txt or not b_txt or not tol_txt:
            raise ValueError('Complete f(x), a, b y tolerancia.')
        a = float(a_txt); b = float(b_txt); tol = float(tol_txt)
        if a >= b:
            raise ValueError('Se requiere a < b.')
        f = parse_function(expr)
        return f, a, b, tol, expr

    def _run_biseccion(self):
        try:
            f, a, b, tol, expr = self._read_inputs()
            res = biseccion(f, a, b, tol)
            table = format_iterations_table(res, 'Bisección')
            header = f"\nFunción: f(x) = {expr}\nIntervalo inicial: [{a}, {b}]\nTolerancia: {tol}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")

    def _run_regla_falsa(self):
        try:
            f, a, b, tol, expr = self._read_inputs()
            res = regla_falsa(f, a, b, tol)
            table = format_iterations_table(res, 'Regla Falsa')
            header = f"\nFunción: f(x) = {expr}\nIntervalo inicial: [{a}, {b}]\nTolerancia: {tol}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")
