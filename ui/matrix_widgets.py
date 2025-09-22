
# ...existing code...

from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QTextEdit, QSpinBox

class VectorArithmeticWidget(QWidget):
    """Suma, resta y multiplicación de vectores"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size_line = QHBoxLayout()
        size_line.addWidget(QLabel('Dimensión n:'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(2, 8); self.n_spin.setValue(3)
        size_line.addWidget(self.n_spin)
        v.addLayout(size_line)
        self.v1 = VectorInput(3, 'v₁')
        self.v2 = VectorInput(3, 'v₂')
        v.addWidget(self.v1)
        v.addWidget(self.v2)
        # Escalar para multiplicación
        esc_line = QHBoxLayout()
        esc_line.addWidget(QLabel('Escalar:'))
        self.esc_edit = QLineEdit(); self.esc_edit.setValidator(number_validator()); self.esc_edit.setFixedWidth(60)
        esc_line.addWidget(self.esc_edit)
        v.addLayout(esc_line)
        # Botones
        btn_sum = QPushButton('Sumar v₁ + v₂')
        btn_res = QPushButton('Restar v₁ - v₂')
        btn_mul = QPushButton('Multiplicar escalar · v₁')
        hbtn = QHBoxLayout(); hbtn.addWidget(btn_sum); hbtn.addWidget(btn_res); hbtn.addWidget(btn_mul)
        v.addLayout(hbtn)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 12))
        v.addWidget(self.output)
        self.n_spin.valueChanged.connect(self._on_resize)
        btn_sum.clicked.connect(self._on_sum)
        btn_res.clicked.connect(self._on_res)
        btn_mul.clicked.connect(self._on_mul)

    def _on_resize(self):
        n = self.n_spin.value()
        self.v1.set_size(n)
        self.v2.set_size(n)

    def _on_sum(self):
        try:
            v1 = self.v1.get_vector()
            v2 = self.v2.get_vector()
            res = v1 + v2
            self.output.setPlainText(f"v₁ + v₂ = {res}")
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _on_res(self):
        try:
            v1 = self.v1.get_vector()
            v2 = self.v2.get_vector()
            res = v1 + (-v2)
            self.output.setPlainText(f"v₁ - v₂ = {res}")
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _on_mul(self):
        try:
            v1 = self.v1.get_vector()
            esc = self.esc_edit.text().strip()
            if esc == '': esc = '1'
            try:
                esc = float(esc.replace(',', '.'))
            except Exception:
                raise ValueError('Escalar inválido')
            res = esc * v1
            self.output.setPlainText(f"{esc} · v₁ = {res}")
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))
from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QTextEdit, QSpinBox
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt

from core.matrix import Matrix, parse_number
from core.formatter import block_from_matrix, with_op, pretty_matrix
from core.linsys import LinearSystemSolver
from core.vector import Vector
from core.linsys_vector import es_combinacion_lineal, resolver_ecuacion_vectorial
from core.matrix_eq import gauss_eliminacion
from .validators import number_validator
class VectorInput(QWidget):
    """Widget para ingresar un vector de tamaño n"""
    def __init__(self, n=3, label='Vector', parent=None):
        super().__init__(parent)
        self.n = n
        layout = QHBoxLayout(self)
        layout.addWidget(QLabel(label+':'))
        self.edits = []
        for i in range(n):
            e = QLineEdit(); e.setValidator(number_validator()); e.setFixedWidth(60)
            e.setPlaceholderText(f"0")
            layout.addWidget(e)
            self.edits.append(e)

    def get_vector(self):
        vals = []
        for i, e in enumerate(self.edits):
            t = e.text().strip()
            if t == '': t = '0'
            try:
                vals.append(float(t.replace(',', '.')))
            except Exception:
                raise ValueError(f"Entrada inválida en posición {i+1}: {t}")
        return Vector(vals)

    def set_size(self, n):
        # Redimensionar el widget
        for e in self.edits:
            e.setParent(None)
        self.edits = []
        self.n = n
        for i in range(n):
            e = QLineEdit(); e.setValidator(number_validator()); e.setFixedWidth(60)
            e.setPlaceholderText(f"0")
            self.layout().addWidget(e)
            self.edits.append(e)

class VectorPropertiesWidget(QWidget):
    """Propiedades algebraicas de R^n"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size_line = QHBoxLayout()
        size_line.addWidget(QLabel('Dimensión n:'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(2, 8); self.n_spin.setValue(3)
        size_line.addWidget(self.n_spin)
        v.addLayout(size_line)
        # Vectores
        self.v1 = VectorInput(3, 'v₁')
        self.v2 = VectorInput(3, 'v₂')
        self.v3 = VectorInput(3, 'v₃')
        v.addWidget(self.v1)
        v.addWidget(self.v2)
        v.addWidget(self.v3)
        # Escalar
        esc_line = QHBoxLayout()
        esc_line.addWidget(QLabel('Escalar:'))
        self.esc_edit = QLineEdit(); self.esc_edit.setValidator(number_validator()); self.esc_edit.setFixedWidth(60)
        esc_line.addWidget(self.esc_edit)
        v.addLayout(esc_line)
        # Botón y salida
        btn = QPushButton('Verificar propiedades')
        v.addWidget(btn)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 12))
        v.addWidget(self.output)
        btn.clicked.connect(self._on_check)
        self.n_spin.valueChanged.connect(self._on_resize)

    def _on_resize(self):
        n = self.n_spin.value()
        self.v1.set_size(n)
        self.v2.set_size(n)
        self.v3.set_size(n)

    def _on_check(self):
        try:
            n = self.n_spin.value()
            v1 = self.v1.get_vector()
            v2 = self.v2.get_vector()
            v3 = self.v3.get_vector()
            esc = self.esc_edit.text().strip()
            if esc == '': esc = '1'
            try:
                esc = float(esc.replace(',', '.'))
            except Exception:
                raise ValueError('Escalar inválido')
            # Suma y escalar
            suma = v1 + v2
            mult = esc * v1
            # Propiedades
            comm = Vector.check_commutative(v1, v2)
            assoc = Vector.check_associative(v1, v2, v3)
            cero = Vector.check_zero(v1)
            opuesto = Vector.check_opposite(v1)
            out = f"v₁ + v₂ = {suma}\n"
            out += f"{esc}·v₁ = {mult}\n\n"
            out += f"Conmutativa: {'✔️' if comm else '❌'}\n"
            out += f"Asociativa: {'✔️' if assoc else '❌'}\n"
            out += f"Vector cero: {'✔️' if cero else '❌'}\n"
            out += f"Vector opuesto: {'✔️' if opuesto else '❌'}\n"
            self.output.setPlainText(out)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

class LinearCombinationWidget(QWidget):
    """Combinación lineal y ecuación vectorial"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size_line = QHBoxLayout()
        size_line.addWidget(QLabel('Dimensión n:'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(2, 8); self.n_spin.setValue(3)
        size_line.addWidget(self.n_spin)
        size_line.addWidget(QLabel('Cantidad de vectores:'))
        self.k_spin = QSpinBox(); self.k_spin.setRange(2, 5); self.k_spin.setValue(3)
        size_line.addWidget(self.k_spin)
        v.addLayout(size_line)
        # Vectores
        self.vec_inputs = [VectorInput(3, f'v{i+1}') for i in range(3)]
        for w in self.vec_inputs:
            v.addWidget(w)
        self.target = VectorInput(3, 'Objetivo')
        v.addWidget(self.target)
        # Botones
        btn1 = QPushButton('¿Es combinación lineal?')
        btn2 = QPushButton('Resolver ecuación vectorial')
        hbtn = QHBoxLayout(); hbtn.addWidget(btn1); hbtn.addWidget(btn2)
        v.addLayout(hbtn)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 12))
        v.addWidget(self.output)
        self.n_spin.valueChanged.connect(self._on_resize)
        self.k_spin.valueChanged.connect(self._on_resize)
        btn1.clicked.connect(self._on_check_comb)
        btn2.clicked.connect(self._on_solve_eq)

    def _on_resize(self):
        n = self.n_spin.value(); k = self.k_spin.value()
        # Redimensionar inputs
        for w in self.vec_inputs:
            w.setParent(None)
        self.vec_inputs = [VectorInput(n, f'v{i+1}') for i in range(k)]
        for i, w in enumerate(self.vec_inputs):
            self.layout().insertWidget(1+i, w)
        self.target.set_size(n)

    def _on_check_comb(self):
        try:
            from core.formatter import pretty_matrix
            from core.matrix import Matrix
            vectores = [w.get_vector() for w in self.vec_inputs]
            objetivo = self.target.get_vector()
            es_comb, pasos = es_combinacion_lineal(vectores, objetivo)
            # Mostrar vectores como columnas
            mat_v = Matrix([v.values for v in vectores]).transpose()
            mat_obj = Matrix([objetivo.values]).transpose()
            vectores_str = '\n'.join(pretty_matrix(mat_v))
            objetivo_str = '\n'.join(pretty_matrix(mat_obj))
            out = 'Vectores dados (columnas):\n' + vectores_str + '\n\nVector objetivo:\n' + objetivo_str + '\n\n'
            # Determinar tipo de solución
            tipo = None
            if pasos and any('Sistema incompatible' in str(p) or 'incompatible' in str(p).lower() for p in pasos):
                tipo = 'incompatible'
            elif pasos and any('Infinitas soluciones' in str(p) for p in pasos):
                tipo = 'infinitas'
            else:
                tipo = 'unica' if es_comb else 'incompatible'
            if tipo == 'incompatible':
                out += '¿Es combinación lineal?: ❌ NO\n\n'
            else:
                out += '¿Es combinación lineal?: ✔️ SÍ\n\n'
            out += '\n'.join(str(p) for p in pasos)
            if tipo == 'infinitas':
                out += '\n\nEjemplo de combinación lineal: puedes expresar el vector objetivo como una combinación infinita de los vectores dados, variando los coeficientes libres.\n'
                out += 'Por ejemplo: c1*v1 + c2*v2 + ... + cn*vn = objetivo, donde algunos ci pueden tomar cualquier valor real.\n'
            self.output.setPlainText(out)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _on_solve_eq(self):
        try:
            vectores = [w.get_vector() for w in self.vec_inputs]
            objetivo = self.target.get_vector()
            _, pasos = resolver_ecuacion_vectorial(vectores, objetivo)
            out = 'Ecuación vectorial:\n\n' + '\n'.join(str(p) for p in pasos)
            self.output.setPlainText(out)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

class MatrixEquationWidget(QWidget):
    """Ecuación matricial AX = B"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size_line = QHBoxLayout()
        size_line.addWidget(QLabel('Filas de A:'))
        self.m_spin = QSpinBox(); self.m_spin.setRange(2, 6); self.m_spin.setValue(3)
        size_line.addWidget(self.m_spin)
        size_line.addWidget(QLabel('Columnas de A:'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(2, 6); self.n_spin.setValue(3)
        size_line.addWidget(self.n_spin)
        size_line.addWidget(QLabel('Columnas de B:'))
        self.k_spin = QSpinBox(); self.k_spin.setRange(1, 4); self.k_spin.setValue(1)
        size_line.addWidget(self.k_spin)
        v.addLayout(size_line)
        self.A = MatrixInput(3, 3)
        self.B = MatrixInput(3, 1)
        mats = QHBoxLayout()
        mats.addWidget(QLabel('Matriz A'))
        mats.addWidget(self.A)
        mats.addSpacing(10)
        mats.addWidget(QLabel('Matriz/vector B'))
        mats.addWidget(self.B)
        v.addLayout(mats)
        btn = QPushButton('Resolver AX = B')
        v.addWidget(btn)
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 12))
        v.addWidget(self.output)
        self.m_spin.valueChanged.connect(self._on_resize)
        self.n_spin.valueChanged.connect(self._on_resize)
        self.k_spin.valueChanged.connect(self._on_resize)
        btn.clicked.connect(self._on_solve)

    def _on_resize(self):
        m = self.m_spin.value(); n = self.n_spin.value(); k = self.k_spin.value()
        self.A.setParent(None); self.B.setParent(None)
        self.A = MatrixInput(m, n)
        self.B = MatrixInput(m, k)
        mats = self.layout().itemAt(1).layout()
        while mats.count():
            w = mats.takeAt(0).widget()
            if w:
                w.setParent(None)
        mats.addWidget(QLabel('Matriz A'))
        mats.addWidget(self.A)
        mats.addSpacing(10)
        mats.addWidget(QLabel('Matriz/vector B'))
        mats.addWidget(self.B)

    def _on_solve(self):
        try:
            A = [[float(x) for x in row] for row in self.A.to_matrix().rows()]
            B = [[float(x) for x in row] for row in self.B.to_matrix().rows()]
            X, pasos = gauss_eliminacion(A, B)
            out = '\n'.join(str(p) for p in pasos)
            if X is not None:
                out += f"\n\nSolución X = {X}"
            self.output.setPlainText(out)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

class MatrixInput(QWidget):
    """Grid editor for a matrix with live validation."""
    def __init__(self, rows=3, cols=3, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.grid = QGridLayout()
        self.setLayout(self.grid)
        self._build()

    def _build(self):
        self.editors = []
        for i in range(self.rows):
            row_edits = []
            for j in range(self.cols):
                e = QLineEdit()
                e.setAlignment(Qt.AlignCenter)
                e.setValidator(number_validator())
                e.setPlaceholderText('0')
                e.setFixedWidth(70)
                row_edits.append(e)
                self.grid.addWidget(e, i, j)
            self.editors.append(row_edits)

    def to_matrix(self) -> Matrix:
        rows = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                text = self.editors[i][j].text().strip()
                if text == '':
                    text = '0'
                try:
                    row.append(parse_number(text))
                except Exception:
                    raise ValueError(f'Entrada inválida en ({i+1},{j+1}): {text}')
            rows.append(row)
        return Matrix(rows)

class AugmentedSystemWidget(QWidget):
    """Input an augmented matrix [A|b] and show Gauss-Jordan steps as in class."""
    def __init__(self, m=3, n=3, parent=None):
        super().__init__(parent)
        self.m = m; self.n = n
        v = QVBoxLayout(self)
        size_line = QHBoxLayout()
        size_line.addWidget(QLabel('Filas (ecuaciones):'))
        self.rows_spin = QSpinBox(); self.rows_spin.setRange(1, 8); self.rows_spin.setValue(m)
        size_line.addWidget(self.rows_spin)
        size_line.addWidget(QLabel('Columnas (variables):'))
        self.cols_spin = QSpinBox(); self.cols_spin.setRange(1, 8); self.cols_spin.setValue(n)
        size_line.addWidget(self.cols_spin)
        self.resize_btn = QPushButton('Redimensionar')
        size_line.addWidget(self.resize_btn)
        v.addLayout(size_line)

        # Matrices A and b
        self.A = MatrixInput(m, n)
        self.b = MatrixInput(m, 1)
        mats = QHBoxLayout()
        mats.addWidget(QLabel('Matriz A'))
        mats.addWidget(self.A)
        mats.addSpacing(10)
        mats.addWidget(QLabel('Vector b'))
        mats.addWidget(self.b)
        v.addLayout(mats)

        # Actions
        actions = QHBoxLayout()
        self.solve_btn = QPushButton('Resolver (Gauss–Jordan)')
        actions.addWidget(self.solve_btn)
        v.addLayout(actions)

        # Output
        self.output = QTextEdit(); self.output.setReadOnly(True)
        self.output.setFont(QFont('Consolas', 12))
        v.addWidget(QLabel('Pasos (como en el pizarrón):'))
        v.addWidget(self.output)

        self.resize_btn.clicked.connect(self._on_resize)
        self.solve_btn.clicked.connect(self._on_solve)

    def _on_resize(self):
        self.m = self.rows_spin.value(); self.n = self.cols_spin.value()
        # rebuild only the editors
        self.A.setParent(None); self.b.setParent(None)
        self.A = MatrixInput(self.m, self.n)
        self.b = MatrixInput(self.m, 1)
        # second layout in the root (index 1) is the mats HBox
        mats = self.layout().itemAt(1).layout()
        while mats.count():
            w = mats.takeAt(0).widget()
            if w:
                w.setParent(None)
        mats.addWidget(QLabel('Matriz A'))
        mats.addWidget(self.A)
        mats.addSpacing(10)
        mats.addWidget(QLabel('Vector b'))
        mats.addWidget(self.b)

    def _on_solve(self):
        try:
            A = self.A.to_matrix(); b = self.b.to_matrix()
            aug = A.augmented_with(b)
            solver = LinearSystemSolver(aug)
            res = solver.solve(self.n)
            # Build pretty steps
            buffs = []
            for step in res['steps']:
                block = block_from_matrix(step.matrix)
                buffs.append(with_op(block, step.op))
            text = ('\n\n~\n\n').join(buffs)
            # classification
            typ = res['type']
            # Mostrar pivotes y variables libres
            pivots = res.get('pivot_cols', [])
            free_vars = res.get('free_vars', [])
            if pivots:
                piv_str = ', '.join([str(j+1) for j in pivots])
                text += f"\n\nColumnas pivote: {piv_str}"
            else:
                text += "\n\nNo hay columnas pivote."
            if free_vars:
                free_str = ', '.join([f"x{j+1}" for j in free_vars])
                text += f"\nVariables libres: {free_str}"
            else:
                text += "\nNo hay variables libres."
            # Consistencia
            if typ == 'unique':
                text += "\n\nEl sistema es consistente (solución única)."
                sol = res['solution']
                sol_str = ', '.join([f"x{i+1} = {sol[i]}" for i in range(len(sol))])
                text += f"\nSolución única: {sol_str}"
            elif typ == 'infinite':
                text += "\n\nEl sistema es consistente (infinitas soluciones)."
                text += "\nSistema con infinitas soluciones (libertad en variables)."
            elif typ == 'inconsistent':
                text += "\n\nEl sistema es inconsistente (sin solución)."
                text += "\nSistema inconsistente (sin solución)."
            self.output.setPlainText(text)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))
