
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
from core.formatter import block_from_matrix, with_op, pretty_matrix, frac_to_str, join_augmented
from core.linsys import LinearSystemSolver
from core.vector import Vector
from core.linsys_vector import es_combinacion_lineal, resolver_ecuacion_vectorial
from core.matrix_eq import gauss_eliminacion
from core.solve_systems import solve_linear_system, solve_homogeneous, analyze_linear_dependence
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
        from core.matrix import parse_number
        vals = []
        for i, e in enumerate(self.edits):
            t = e.text().strip()
            if t == '': t = '0'
            try:
                vals.append(parse_number(t))
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
            from core.formatter import pretty_matrix
            from core.matrix import Matrix
            A = self.A.to_matrix().rows()
            B = self.B.to_matrix().rows()
            X, pasos = gauss_eliminacion(A, B)
            out = '\n'.join(str(p) for p in pasos)
            if X is not None:
                # Ordenar la matriz X por filas y columnas (ya está ordenada por construcción)
                # Mostrar la solución X como matriz bonita y en fracciones
                mat_X = Matrix([[x for x in row] for row in X])
                out += f"\n\nSolución X (en fracciones):\n" + '\n'.join(pretty_matrix(mat_X))
            self.output.setPlainText(out)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

class SystemsSolverWidget(QWidget):
    """Resolver AX=B y AX=0 con pasos y conclusión"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size = QHBoxLayout()
        size.addWidget(QLabel('Filas (m):'))
        self.m_spin = QSpinBox(); self.m_spin.setRange(1, 6); self.m_spin.setValue(3)
        size.addWidget(self.m_spin)
        size.addWidget(QLabel('Columnas (n):'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 6); self.n_spin.setValue(3)
        size.addWidget(self.n_spin)
        v.addLayout(size)
        self.A = MatrixInput(3, 3)
        self.B = MatrixInput(3, 1)
        mats = QHBoxLayout(); mats.addWidget(QLabel('Matriz A')); mats.addWidget(self.A); mats.addSpacing(10); mats.addWidget(QLabel('Vector/matriz B')); mats.addWidget(self.B)
        v.addLayout(mats)
        btns = QHBoxLayout()
        btn_h = QPushButton('Resolver AX = 0 (Homogéneo)')
        btn_nh = QPushButton('Resolver AX = B (No homogéneo)')
        btns.addWidget(btn_h); btns.addWidget(btn_nh)
        v.addLayout(btns)
        self.output = QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont('Consolas', 11))
        v.addWidget(self.output)
        actions = QHBoxLayout()
        btn_copy = QPushButton('Copiar resultado')
        btn_clear = QPushButton('Limpiar')
        actions.addWidget(btn_copy); actions.addWidget(btn_clear); actions.addStretch()
        v.addLayout(actions)
        self.m_spin.valueChanged.connect(self._resize)
        self.n_spin.valueChanged.connect(self._resize)
        btn_h.clicked.connect(self._solve_h)
        btn_nh.clicked.connect(self._solve_nh)
        btn_copy.clicked.connect(self._copy_output)
        btn_clear.clicked.connect(self._clear_output)

    def _resize(self):
        m = self.m_spin.value(); n = self.n_spin.value()
        self.A.setParent(None); self.B.setParent(None)
        self.A = MatrixInput(m, n)
        self.B = MatrixInput(m, 1)
        mats = self.layout().itemAt(1).layout()
        while mats.count():
            w = mats.takeAt(0).widget()
            if w: w.setParent(None)
        mats.addWidget(QLabel('Matriz A')); mats.addWidget(self.A); mats.addSpacing(10); mats.addWidget(QLabel('Vector/matriz B')); mats.addWidget(self.B)

    def _solve_h(self):
        try:
            A = self.A.to_matrix().rows()
            res = solve_homogeneous(A)
            self._render_result(res)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _solve_nh(self):
        try:
            A = self.A.to_matrix().rows()
            B = [row[0] for row in self.B.to_matrix().rows()]
            res = solve_linear_system(A, B)
            self._render_result(res)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _render_result(self, res):
        from core.formatter import pretty_matrix
        from core.matrix import Matrix
        out = []
        out.append('\n'.join(res['steps']))
        # RREF
        R = res.get('rref')
        if R:
            out.append('\nRREF de la matriz aumentada:')
            out.append('\n'.join(pretty_matrix(Matrix(R))))
        # Mensajes
        out.append('\nEstado: ' + ('Consistente' if res.get('consistent') else 'Inconsistente'))
        out.append('Conclusión: ' + res.get('message', ''))
        # Solución
        if res.get('unique') and res.get('solution') is not None:
            sol = res['solution']
            out.append('\nSolución única x:')
            out.append('\n'.join(pretty_matrix(Matrix([[x] for x in sol]))))
        elif res.get('infinite'):
            x_part = res.get('solution') or []
            basis = res.get('param_basis') or []
            free = res.get('free_vars') or []
            out.append('\nSolución general (paramétrica):')
            out.append('x = x_p + Σ t_j v_j')
            out.append('x_p =\n' + '\n'.join(pretty_matrix(Matrix([[x] for x in x_part]))))
            for j, v in enumerate(basis):
                out.append(f"v{j+1} =\n" + '\n'.join(pretty_matrix(Matrix([[x] for x in v]))))
            out.append(f"Variables libres: {free}")
        self.output.setPlainText('\n'.join(out))

    def _copy_output(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.output.toPlainText())

    def _clear_output(self):
        self.output.clear()

class LinearDependenceWidget(QWidget):
    """Analizar dependencia/independencia lineal"""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        size = QHBoxLayout()
        size.addWidget(QLabel('Dimensión n:'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 8); self.n_spin.setValue(3)
        size.addWidget(self.n_spin)
        size.addWidget(QLabel('Cantidad de vectores k:'))
        self.k_spin = QSpinBox(); self.k_spin.setRange(1, 6); self.k_spin.setValue(3)
        size.addWidget(self.k_spin)
        v.addLayout(size)
        self.vec_inputs = [VectorInput(3, 'v1'), VectorInput(3, 'v2'), VectorInput(3, 'v3')]
        for w in self.vec_inputs: v.addWidget(w)
        btn = QPushButton('Analizar dependencia')
        v.addWidget(btn)
        self.output = QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont('Consolas', 11))
        v.addWidget(self.output)
        actions = QHBoxLayout()
        btn_copy = QPushButton('Copiar resultado')
        btn_clear = QPushButton('Limpiar')
        actions.addWidget(btn_copy); actions.addWidget(btn_clear); actions.addStretch()
        v.addLayout(actions)
        self.n_spin.valueChanged.connect(self._resize)
        self.k_spin.valueChanged.connect(self._resize)
        btn.clicked.connect(self._analyze)
        btn_copy.clicked.connect(self._copy_output)
        btn_clear.clicked.connect(self._clear_output)

    def _resize(self):
        n = self.n_spin.value(); k = self.k_spin.value()
        for w in self.vec_inputs: w.setParent(None)
        self.vec_inputs = [VectorInput(n, f'v{i+1}') for i in range(k)]
        for i, w in enumerate(self.vec_inputs): self.layout().insertWidget(1+i, w)

    def _analyze(self):
        try:
            vectors = [w.get_vector().values for w in self.vec_inputs]
            res = analyze_linear_dependence(vectors)
            self._render(res)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _render(self, res):
        from core.formatter import pretty_matrix
        from core.matrix import Matrix
        out = []
        # Planteo de la ecuación vectorial: c1 v1 + ... + ck vk = 0
        try:
            # Reconstruimos los vectores desde la UI para mostrarlos como columnas
            # Nota: usamos los valores tal cual para la visualización
            vectors_vals = [w.get_vector().values for w in self.vec_inputs]
            if vectors_vals:
                mat_cols = Matrix(vectors_vals).transpose()  # n x k (columnas = vi)
                zero_vec = Matrix([[0] for _ in range(self.n_spin.value())])
                k = len(vectors_vals)
                eq_str = ' + '.join([f"c{i+1}·v{i+1}" for i in range(k)]) + ' = 0'
                out.append('Ecuación vectorial (planteo):')
                out.append(eq_str)
                out.append('\nVectores (columnas):')
                out.append('\n'.join(pretty_matrix(mat_cols)))
                out.append('\nVector 0:')
                out.append('\n'.join(pretty_matrix(zero_vec)))
        except Exception:
            pass
        out.append('\n'.join(res['steps']))
        if res.get('rref'):
            out.append('\nRREF de la matriz aumentada:')
            out.append('\n'.join(pretty_matrix(Matrix(res['rref']))))
        out.append('\nConclusión: ' + res.get('message',''))
        if res.get('infinite'):
            out.append('\nBase del espacio de soluciones no triviales:')
            for j, v in enumerate(res.get('param_basis') or []):
                out.append(f"v{j+1} =\n" + '\n'.join(pretty_matrix(Matrix([[x] for x in v]))))
        self.output.setPlainText('\n'.join(out))

    def _copy_output(self):
        from PySide6.QtWidgets import QApplication
        QApplication.clipboard().setText(self.output.toPlainText())

    def _clear_output(self):
        self.output.clear()

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

class MatrixOpsWidget(QWidget):
    """Operaciones con matrices: suma, resta, escalar, producto y traspuesta."""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)

        # Tamaños para A y B
        size = QHBoxLayout()
        size.addWidget(QLabel('Filas de A (m):'))
        self.mA = QSpinBox(); self.mA.setRange(1, 8); self.mA.setValue(2)
        size.addWidget(self.mA)
        size.addWidget(QLabel('Columnas de A (n):'))
        self.nA = QSpinBox(); self.nA.setRange(1, 8); self.nA.setValue(2)
        size.addWidget(self.nA)
        size.addSpacing(20)
        size.addWidget(QLabel('Filas de B (p):'))
        self.mB = QSpinBox(); self.mB.setRange(1, 8); self.mB.setValue(2)
        size.addWidget(self.mB)
        size.addWidget(QLabel('Columnas de B (q):'))
        self.nB = QSpinBox(); self.nB.setRange(1, 8); self.nB.setValue(2)
        size.addWidget(self.nB)
        self.resizeBtn = QPushButton('Redimensionar')
        size.addWidget(self.resizeBtn)
        v.addLayout(size)

        # Entradas de matrices
        self.A = MatrixInput(self.mA.value(), self.nA.value())
        self.B = MatrixInput(self.mB.value(), self.nB.value())
        mats = QHBoxLayout()
        left = QVBoxLayout(); left.addWidget(QLabel('Matriz A')); left.addWidget(self.A)
        right = QVBoxLayout(); right.addWidget(QLabel('Matriz B')); right.addWidget(self.B)
        mats.addLayout(left); mats.addSpacing(15); mats.addLayout(right)
        v.addLayout(mats)

        # Escalar
        scaleline = QHBoxLayout()
        scaleline.addWidget(QLabel('Escalar k:'))
        self.k_edit = QLineEdit(); self.k_edit.setValidator(number_validator()); self.k_edit.setFixedWidth(100); self.k_edit.setPlaceholderText('1/2, -3, 0.25')
        scaleline.addWidget(self.k_edit)
        scaleline.addStretch()
        v.addLayout(scaleline)

        # Botones de operaciones
        btns1 = QHBoxLayout()
        self.btn_sum = QPushButton('A + B')
        self.btn_sub = QPushButton('A - B')
        self.btn_scal = QPushButton('k · A')
        self.btn_mul = QPushButton('A · B')
        self.btn_transpose = QPushButton('Traspuesta de A')
        btns1.addWidget(self.btn_sum); btns1.addWidget(self.btn_sub); btns1.addWidget(self.btn_scal); btns1.addWidget(self.btn_mul); btns1.addWidget(self.btn_transpose)
        v.addLayout(btns1)

        # Botón adicional para comparar (A+B)^T con A^T + B^T
        btns2 = QHBoxLayout()
        self.btn_transpose_sum_cmp = QPushButton('(A + B)ᵀ  vs  Aᵀ + Bᵀ')
        btns2.addWidget(self.btn_transpose_sum_cmp)
        # Nuevo botón para comparar (A-B)^T con A^T - B^T
        self.btn_transpose_diff_cmp = QPushButton('(A - B)ᵀ  vs  Aᵀ - Bᵀ')
        btns2.addWidget(self.btn_transpose_diff_cmp)
        btns2.addStretch()
        v.addLayout(btns2)

        # Salida
        self.output = QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont('Consolas', 11))
        v.addWidget(self.output)

        # Conexiones
        self.resizeBtn.clicked.connect(self._on_resize)
        self.btn_sum.clicked.connect(self._do_sum)
        self.btn_sub.clicked.connect(self._do_sub)
        self.btn_scal.clicked.connect(self._do_scal)
        self.btn_mul.clicked.connect(self._do_mul)
        self.btn_transpose.clicked.connect(self._do_transpose)
        self.btn_transpose_sum_cmp.clicked.connect(self._do_transpose_sum_compare)
        self.btn_transpose_diff_cmp.clicked.connect(self._do_transpose_diff_compare)

    def _on_resize(self):
        mA, nA, mB, nB = self.mA.value(), self.nA.value(), self.mB.value(), self.nB.value()
        # rebuild inputs
        self.A.setParent(None); self.B.setParent(None)
        self.A = MatrixInput(mA, nA)
        self.B = MatrixInput(mB, nB)
        mats = self.layout().itemAt(1).layout()
        # left VBox is at index 0, right VBox at index 2 (with spacing at 1)
        # Clear and rebuild to avoid layout confusion
        while mats.count():
            it = mats.takeAt(0)
            w = it.widget()
            if w: w.setParent(None)
        left = QVBoxLayout(); left.addWidget(QLabel('Matriz A')); left.addWidget(self.A)
        right = QVBoxLayout(); right.addWidget(QLabel('Matriz B')); right.addWidget(self.B)
        mats.addLayout(left); mats.addSpacing(15); mats.addLayout(right)

    def _get_A(self) -> Matrix:
        return self.A.to_matrix()

    def _get_B(self) -> Matrix:
        return self.B.to_matrix()

    def _parse_k(self):
        t = self.k_edit.text().strip()
        if t == '':
            t = '1'
        return parse_number(t)

    def _fmt_head(self, title: str) -> str:
        return f"=== {title} ===\n"

    def _matrix_block(self, label: str, M: Matrix) -> str:
        return f"{label}:\n" + '\n'.join(pretty_matrix(M)) + '\n'

    def _do_sum(self):
        try:
            A = self._get_A(); B = self._get_B()
            out = [self._fmt_head('Suma de matrices A + B')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}, B es {B.m}×{B.n}")
            if A.m == B.m and A.n == B.n:
                out.append('✔️ Las matrices son compatibles para la suma (mismos tamaños).\n')
                C = A.add(B)
                out.append(self._matrix_block('A', A))
                out.append('+\n')
                out.append(self._matrix_block('B', B))
                out.append('=\n')
                out.append(self._matrix_block('A + B', C))
                # Paso a paso elemento a elemento
                steps = ['C[i,j] = A[i,j] + B[i,j]']
                for i in range(A.m):
                    for j in range(A.n):
                        a = frac_to_str(A.at(i,j)); b = frac_to_str(B.at(i,j)); c = frac_to_str(C.at(i,j))
                        steps.append(f"c[{i+1},{j+1}] = {a} + {b} = {c}")
                out.append('\n' + '\n'.join(steps))
            else:
                out.append('❌ No se puede sumar: las matrices deben tener igual número de filas y columnas.')
            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_sub(self):
        try:
            A = self._get_A(); B = self._get_B()
            out = [self._fmt_head('Resta de matrices A - B')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}, B es {B.m}×{B.n}")
            if A.m == B.m and A.n == B.n:
                out.append('✔️ Las matrices son compatibles para la resta (mismos tamaños).\n')
                C = A.sub(B)
                out.append(self._matrix_block('A', A))
                out.append('-\n')
                out.append(self._matrix_block('B', B))
                out.append('=\n')
                out.append(self._matrix_block('A - B', C))
                steps = ['C[i,j] = A[i,j] - B[i,j]']
                for i in range(A.m):
                    for j in range(A.n):
                        a = frac_to_str(A.at(i,j)); b = frac_to_str(B.at(i,j)); c = frac_to_str(C.at(i,j))
                        steps.append(f"c[{i+1},{j+1}] = {a} - {b} = {c}")
                out.append('\n' + '\n'.join(steps))
            else:
                out.append('❌ No se puede restar: las matrices deben tener igual número de filas y columnas.')
            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_scal(self):
        try:
            A = self._get_A(); k = self._parse_k()
            out = [self._fmt_head('Multiplicación por escalar k · A')]
            out.append(f"Escalar ingresado: k = {frac_to_str(k)}")
            out.append('✔️ Siempre es posible multiplicar una matriz por un escalar.\n')
            C = A.scalar(k)
            out.append(self._matrix_block('A', A))
            out.append('→\n')
            out.append(self._matrix_block('k · A', C))
            steps = ['C[i,j] = k · A[i,j]']
            for i in range(A.m):
                for j in range(A.n):
                    a = frac_to_str(A.at(i,j)); c = frac_to_str(C.at(i,j))
                    steps.append(f"c[{i+1},{j+1}] = {frac_to_str(k)} · {a} = {c}")
            out.append('\n' + '\n'.join(steps))
            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_mul(self):
        try:
            A = self._get_A(); B = self._get_B()
            out = [self._fmt_head('Producto de matrices A · B')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}, B es {B.m}×{B.n}")
            if A.n != B.m:
                out.append('❌ El producto AB no es posible: columnas de A deben coincidir con filas de B (n = p).')
                self.output.setPlainText('\n'.join(out))
                return
            out.append('✔️ El producto AB es posible. El resultado será de tamaño ' + f"{A.m}×{B.n}.\n")
            C = A.mul(B)
            out.append(self._matrix_block('A', A))
            out.append('×\n')
            out.append(self._matrix_block('B', B))
            out.append('=\n')
            out.append(self._matrix_block('A · B', C))
            # Paso a paso por entradas
            steps = ['C[i,j] = Σ_{k=1..n} A[i,k] · B[k,j]']
            for i in range(A.m):
                for j in range(B.n):
                    terms = []
                    for k in range(A.n):
                        terms.append(f"{frac_to_str(A.at(i,k))}·{frac_to_str(B.at(k,j))}")
                    s_terms = ' + '.join(terms) if terms else '0'
                    steps.append(f"c[{i+1},{j+1}] = {s_terms} = {frac_to_str(C.at(i,j))}")
            out.append('\n' + '\n'.join(steps))
            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_transpose(self):
        try:
            A = self._get_A()
            AT = A.transpose()
            ATT = AT.transpose()
            out = [self._fmt_head('Traspuesta de A')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}; Aᵀ será {A.n}×{A.m}.\n")
            out.append('Intercambiamos filas por columnas: (Aᵀ)[i,j] = A[j,i].\n')
            out.append(self._matrix_block('A', A))
            out.append('→\n')
            out.append(self._matrix_block('Aᵀ', AT))
            # Verificación de propiedad
            ok = (ATT.rows() == A.rows())
            out.append('Propiedad: (Aᵀ)ᵀ = A → ' + ('✔️ Se cumple.' if ok else '❌ No se cumple.'))
            if not ok:
                out.append(self._matrix_block('(Aᵀ)ᵀ', ATT))
            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_transpose_sum_compare(self):
        try:
            A = self._get_A(); B = self._get_B()
            out = [self._fmt_head('Comparación: (A + B)ᵀ  vs  Aᵀ + Bᵀ')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}, B es {B.m}×{B.n}")
            if A.m != B.m or A.n != B.n:
                out.append('❌ No se puede sumar A + B: las matrices deben tener igual número de filas y columnas.')
                self.output.setPlainText('\n'.join(out))
                return

            # Calcular A + B y su traspuesta
            C = A.add(B)
            Ct = C.transpose()

            # Calcular A^T + B^T
            At = A.transpose()
            Bt = B.transpose()
            sum_trans = At.add(Bt)

            out.append('\nMatriz A:')
            out.extend(pretty_matrix(A))
            out.append('\nMatriz B:')
            out.extend(pretty_matrix(B))
            out.append('\nA + B:')
            out.extend(pretty_matrix(C))
            out.append('\n(A + B)ᵀ:')
            out.extend(pretty_matrix(Ct))
            out.append('\nAᵀ:')
            out.extend(pretty_matrix(At))
            out.append('\nBᵀ:')
            out.extend(pretty_matrix(Bt))
            out.append('\nAᵀ + Bᵀ:')
            out.extend(pretty_matrix(sum_trans))

            # Comprobación de igualdad elemento a elemento
            equal = (Ct.rows() == sum_trans.rows())
            out.append('\nResultado de la verificación: ' + ('✔️ (A + B)ᵀ = Aᵀ + Bᵀ' if equal else '❌ No son iguales'))

            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _do_transpose_diff_compare(self):
        try:
            A = self._get_A(); B = self._get_B()
            out = [self._fmt_head('Comparación: (A - B)ᵀ  vs  Aᵀ - Bᵀ')]
            out.append(f"Dimensiones: A es {A.m}×{A.n}, B es {B.m}×{B.n}")
            if A.m != B.m or A.n != B.n:
                out.append('❌ No se puede restar A - B: las matrices deben tener igual número de filas y columnas.')
                self.output.setPlainText('\n'.join(out))
                return

            # Calcular A - B y su traspuesta
            D = A.sub(B)
            Dt = D.transpose()

            # Calcular A^T - B^T
            At = A.transpose()
            Bt = B.transpose()
            diff_trans = At.sub(Bt)

            out.append('\nMatriz A:')
            out.extend(pretty_matrix(A))
            out.append('\nMatriz B:')
            out.extend(pretty_matrix(B))
            out.append('\nA - B:')
            out.extend(pretty_matrix(D))
            out.append('\n(A - B)ᵀ:')
            out.extend(pretty_matrix(Dt))
            out.append('\nAᵀ:')
            out.extend(pretty_matrix(At))
            out.append('\nBᵀ:')
            out.extend(pretty_matrix(Bt))
            out.append('\nAᵀ - Bᵀ:')
            out.extend(pretty_matrix(diff_trans))

            # Comprobación de igualdad elemento a elemento
            equal = (Dt.rows() == diff_trans.rows())
            out.append('\nResultado de la verificación: ' + ('✔️ (A - B)ᵀ = Aᵀ - Bᵀ' if equal else '❌ No son iguales'))

            self.output.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))
            
    """Inversa de una matriz cuadrada mediante Gauss–Jordan con pasos."""
class MatrixInverseWidget(QWidget):
    """Calcular la inversa de una matriz cuadrada A mediante Gauss–Jordan [A | I] → [I | A^{-1}] con pasos."""
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        # Tamaño
        size = QHBoxLayout()
        size.addWidget(QLabel('Tamaño n (matriz A es n×n):'))
        self.n_spin = QSpinBox(); self.n_spin.setRange(1, 8); self.n_spin.setValue(3)
        size.addWidget(self.n_spin)
        self.resizeBtn = QPushButton('Redimensionar')
        size.addWidget(self.resizeBtn)
        v.addLayout(size)

        # Entrada de A
        self.A = MatrixInput(self.n_spin.value(), self.n_spin.value())
        mat_box = QVBoxLayout(); mat_box.addWidget(QLabel('Matriz A')); mat_box.addWidget(self.A)
        v.addLayout(mat_box)

        # Acciones
        btns = QHBoxLayout()
        self.btn_inv = QPushButton('Calcular A^{-1} (Gauss–Jordan)')
        btns.addWidget(self.btn_inv)
        btns.addStretch()
        v.addLayout(btns)

        # Salidas
        self.output = QTextEdit(); self.output.setReadOnly(True); self.output.setFont(QFont('Consolas', 11))
        v.addWidget(QLabel('Pasos (matriz aumentada [A | I]):'))
        v.addWidget(self.output)

        self.result = QTextEdit(); self.result.setReadOnly(True); self.result.setFont(QFont('Consolas', 11))
        v.addWidget(QLabel('Resultado y verificaciones:'))
        v.addWidget(self.result)

        # Propiedades/Teoremas de invertibilidad
        self.props = QTextEdit(); self.props.setReadOnly(True); self.props.setFont(QFont('Consolas', 11))
        v.addWidget(QLabel('Teoremas y propiedades asociadas a la invertibilidad de A:'))
        v.addWidget(self.props)

        # Conexiones
        self.resizeBtn.clicked.connect(self._on_resize)
        self.btn_inv.clicked.connect(self._compute_inverse)

    def _on_resize(self):
        n = self.n_spin.value()
        self.A.setParent(None)
        self.A = MatrixInput(n, n)
        # Reinsert A into the mat_box (which is layout index 1)
        mat_box = QVBoxLayout(); mat_box.addWidget(QLabel('Matriz A')); mat_box.addWidget(self.A)
        # Replace layout at index 1
        root = self.layout()
        old = root.itemAt(1)
        # Remove the old layout widgets if present
        if old is not None:
            lay = old.layout()
            if lay is not None:
                while lay.count():
                    it = lay.takeAt(0)
                    w = it.widget()
                    if w:
                        w.setParent(None)
                # Remove the layout itself
                root.removeItem(old)
        root.insertLayout(1, mat_box)

    def _compute_inverse(self):
        try:
            A = self.A.to_matrix()
            n = A.n
            if A.m != A.n:
                raise ValueError('A debe ser cuadrada para calcular su inversa.')
            I = Matrix.identity(n)
            aug = A.augmented_with(I)
            solver = LinearSystemSolver(aug)
            rref_mat, steps, rank, m = solver.rref()

            # Construir salida de pasos con barra | entre bloques
            buffs = []
            for step in steps:
                full = step.matrix
                rows = full.rows()
                left = Matrix([row[:n] for row in rows])
                right = Matrix([row[n:] for row in rows])
                lines = join_augmented(left, right)
                buffs.append(with_op(lines, step.op))
            self.output.setPlainText(('\n\n~\n\n').join(buffs))

            # Verificar si la parte izquierda es identidad
            rows = rref_mat.rows()
            left_final = Matrix([row[:n] for row in rows])
            right_final = Matrix([row[n:] for row in rows])

            # Checar identidad
            is_identity = left_final.rows() == Matrix.identity(n).rows()
            # Siempre evaluar propiedades de invertibilidad (pivot count, etc.)
            self._update_invertibility_properties(left_final, n)
            if not is_identity:
                # Mensaje explícito solicitado: no tiene pivote en cada fila
                piv = getattr(self, '_last_pivot_info', None)
                if piv and not piv.get('has_n_pivots', False):
                    head = 'Conclusión: ❌ La matriz no es invertible porque no tiene pivote en cada fila.'
                    detalle = f"Pivotes encontrados: {piv.get('piv_count', 0)} de {n}."
                else:
                    head = 'Conclusión: ❌ A no es invertible (la izquierda no es Iₙ).'
                    detalle = None
                msg = [head]
                if detalle:
                    msg.append(detalle)
                msg.append('\nMatriz final [A_rref | ? ]:')
                msg.extend(pretty_matrix(left_final))
                self.result.setPlainText('\n'.join(msg))
                return

            # A inversa
            inv = right_final
            out = []
            out.append('Conclusión: ✔️ A es invertible. Se obtuvo A^{-1}.')
            out.append('\nA^{-1} =')
            out.extend(pretty_matrix(inv))

            # Verificaciones de propiedades: A·A^{-1} = I y A^{-1}·A = I
            try:
                prod1 = A.mul(inv)
                prod2 = inv.mul(A)
                I1 = Matrix.identity(n)
                ok1 = (prod1.rows() == I1.rows())
                ok2 = (prod2.rows() == I1.rows())
                out.append('\nVerificación: A · A^{-1} = I → ' + ('✔️' if ok1 else '❌'))
                out.extend(pretty_matrix(prod1))
                out.append('\nVerificación: A^{-1} · A = I → ' + ('✔️' if ok2 else '❌'))
                out.extend(pretty_matrix(prod2))
            except Exception:
                pass

            self.result.setPlainText('\n'.join(out))
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))

    def _update_invertibility_properties(self, left_final: Matrix, n: int):
        rows = left_final.rows()
        # Contar pivotes (columnas con un 1 y ceros en esa columna, formato RREF)
        pivot_cols = []
        for j in range(n):
            found = False
            for i in range(len(rows)):
                if rows[i][j] == 1:
                    # Verificar que en esa fila las demás entradas (en el bloque izquierda) sean 0 excepto en j
                    if all(rows[i][k] == 0 for k in range(n) if k != j) and all(rows[r][j] == 0 for r in range(len(rows)) if r != i):
                        found = True
                        break
            if found:
                pivot_cols.append(j)
        piv_count = len(pivot_cols)
        has_n_pivots = (piv_count == n)

        # Ax = 0 solo tiene solución trivial <=> rango(A) = n <=> n pivotes
        trivial_only = has_n_pivots
        # Columnas linealmente independientes <=> rango(A) = n
        cols_independent = has_n_pivots

        lines = []
        # (c) n pivotes
        lines.append('(c) La matriz A tiene n posiciones pivote: ' + ('✔️ Sí' if has_n_pivots else '❌ No'))
        lines.append(f"   Pivotes encontrados: {piv_count} de {n}. Columnas pivote (1-indexadas): " + (', '.join(str(j+1) for j in pivot_cols) if pivot_cols else '—'))
        lines.append('   Interpretación: Si A tiene n pivotes, entonces A es invertible.')
        # (d) Ax=0 solo trivial
        lines.append('(d) La ecuación A·x = 0 tiene solamente la solución trivial: ' + ('✔️ Sí' if trivial_only else '❌ No'))
        lines.append('   Interpretación: Si A·x=0 solo tiene la solución trivial, entonces A^{-1} existe.')
        # (e) Columnas LI
        lines.append('(e) Las columnas de A forman un conjunto linealmente independiente: ' + ('✔️ Sí' if cols_independent else '❌ No'))
        lines.append('   Interpretación: Si las columnas son linealmente independientes, entonces A es una matriz invertible.')

        self.props.setPlainText('\n'.join(lines))
        # Guardar para mensajes posteriores
        self._last_pivot_info = {
            'pivot_cols': pivot_cols,
            'piv_count': piv_count,
            'has_n_pivots': has_n_pivots,
        }
