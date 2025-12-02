from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton, QTextEdit, QGroupBox
)
from PySide6.QtGui import QFont

from core.numericos import parse_function, parse_number_expr, biseccion, regla_falsa, newton_raphson, metodo_secante, format_iterations_table


class NumericalMethodsWidget(QWidget):
    """Vista para ejecutar Bisección, Regla Falsa, Newton–Raphson y Secante.

    Permite ingresar f(x), intervalo [a,b], tolerancia, o x0/x1 según método,
    y ejecutar cada uno mostrando tabla de iteraciones y resumen final.
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
        self.x0_edit = QLineEdit(); self.x0_edit.setPlaceholderText('x0 (Newton/Secante)');
        self.x1_edit = QLineEdit(); self.x1_edit.setPlaceholderText('x1 (Secante)');
        self.tol_edit = QLineEdit(); self.tol_edit.setPlaceholderText('tolerancia (ej: 1e-4)');
        self.max_iter_edit = QLineEdit(); self.max_iter_edit.setPlaceholderText('max_iter (ej: 100)');
        # Sin validadores estrictos para permitir exponentes (ej: 2^3).
        # Se validará al leer usando parse_number_expr.

        h.addWidget(QLabel('f(x):'))
        h.addWidget(self.f_edit, stretch=2)
        h.addWidget(QLabel('a:'))
        h.addWidget(self.a_edit)
        h.addWidget(QLabel('b:'))
        h.addWidget(self.b_edit)
        h.addWidget(QLabel('x0:'))
        h.addWidget(self.x0_edit)
        h.addWidget(QLabel('x1:'))
        h.addWidget(self.x1_edit)
        h.addWidget(QLabel('tol:'))
        h.addWidget(self.tol_edit)
        h.addWidget(QLabel('max_iter:'))
        h.addWidget(self.max_iter_edit)
        gb.setLayout(h)

        # Botones
        hb = QHBoxLayout()
        btn_bis = QPushButton('Ejecutar Bisección')
        btn_rf = QPushButton('Ejecutar Regla Falsa')
        btn_newton = QPushButton('Ejecutar Newton–Raphson')
        btn_secante = QPushButton('Ejecutar Secante')
        btn_bis.clicked.connect(self._run_biseccion)
        btn_rf.clicked.connect(self._run_regla_falsa)
        btn_newton.clicked.connect(self._run_newton)
        btn_secante.clicked.connect(self._run_secante)
        hb.addWidget(btn_bis)
        hb.addWidget(btn_rf)
        hb.addWidget(btn_newton)
        hb.addWidget(btn_secante)

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
        max_iter_txt = self.max_iter_edit.text().strip()
        if not expr or not a_txt or not b_txt or not tol_txt:
            raise ValueError('Complete f(x), a, b y tolerancia.')
        a = parse_number_expr(a_txt)
        b = parse_number_expr(b_txt)
        tol = parse_number_expr(tol_txt)
        if a >= b:
            raise ValueError('Se requiere a < b.')
        max_iter = int(parse_number_expr(max_iter_txt)) if max_iter_txt else 100
        f = parse_function(expr)
        return f, a, b, tol, max_iter, expr

    def _run_biseccion(self):
        try:
            f, a, b, tol, max_iter, expr = self._read_inputs()
            res = biseccion(f, a, b, tol, max_iter=max_iter)
            table = format_iterations_table(res, 'Bisección')
            header = f"\nFunción: f(x) = {expr}\nIntervalo inicial: [{a}, {b}]\nTolerancia: {tol}\nMax iter: {max_iter}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")

    def _run_regla_falsa(self):
        try:
            f, a, b, tol, max_iter, expr = self._read_inputs()
            res = regla_falsa(f, a, b, tol, max_iter=max_iter)
            table = format_iterations_table(res, 'Regla Falsa')
            header = f"\nFunción: f(x) = {expr}\nIntervalo inicial: [{a}, {b}]\nTolerancia: {tol}\nMax iter: {max_iter}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")

    def _run_newton(self):
        try:
            expr = self.f_edit.text().strip()
            tol_txt = self.tol_edit.text().strip()
            max_iter_txt = self.max_iter_edit.text().strip()
            x0_txt = self.x0_edit.text().strip()
            if not expr or not tol_txt or not x0_txt:
                raise ValueError('Complete f(x), x0 y tolerancia (opcional max_iter).')
            f = parse_function(expr)
            tol = parse_number_expr(tol_txt)
            max_iter = int(parse_number_expr(max_iter_txt)) if max_iter_txt else 100
            x0 = parse_number_expr(x0_txt)

            # df: None para usar derivada numérica por defecto
            res = newton_raphson(f, df=None, x0=x0, tol=tol, max_iter=max_iter)
            table = format_iterations_table(res, 'Newton–Raphson')
            header = f"\nFunción: f(x) = {expr}\nx0: {x0}\nTolerancia: {tol}\nMax iter: {max_iter}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")

    def _run_secante(self):
        try:
            expr = self.f_edit.text().strip()
            tol_txt = self.tol_edit.text().strip()
            max_iter_txt = self.max_iter_edit.text().strip()
            x0_txt = self.x0_edit.text().strip()
            x1_txt = self.x1_edit.text().strip()
            if not expr or not tol_txt or not x0_txt or not x1_txt:
                raise ValueError('Complete f(x), x0, x1 y tolerancia (opcional max_iter).')
            f = parse_function(expr)
            tol = parse_number_expr(tol_txt)
            max_iter = int(parse_number_expr(max_iter_txt)) if max_iter_txt else 100
            x0 = parse_number_expr(x0_txt)
            x1 = parse_number_expr(x1_txt)

            res = metodo_secante(f, x0=x0, x1=x1, tol=tol, max_iter=max_iter)
            table = format_iterations_table(res, 'Secante')
            header = f"\nFunción: f(x) = {expr}\nx0: {x0}  x1: {x1}\nTolerancia: {tol}\nMax iter: {max_iter}\n"
            self.output.setPlainText(header + table)
        except Exception as e:
            self.output.setPlainText(f"Error: {e}")
