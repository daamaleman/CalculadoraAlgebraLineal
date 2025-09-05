from PySide6.QtWidgets import QWidget, QGridLayout, QLineEdit, QLabel, QVBoxLayout, QHBoxLayout, QPushButton, QMessageBox, QTextEdit, QSpinBox
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from core.matrix import Matrix, parse_number
from core.formatter import block_from_matrix, with_op, pretty_matrix
from core.linsys import LinearSystemSolver
from .validators import number_validator

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
            if typ == 'unique':
                sol = res['solution']
                sol_str = ', '.join([f"x{i+1} = {sol[i]}" for i in range(len(sol))])
                text += f"\n\nSolución única: {sol_str}"
            elif typ == 'infinite':
                text += "\n\nSistema con infinitas soluciones (libertad en variables)."
            elif typ == 'inconsistent':
                text += "\n\nSistema inconsistente (sin solución)."
            self.output.setPlainText(text)
        except Exception as ex:
            QMessageBox.critical(self, 'Error', str(ex))
