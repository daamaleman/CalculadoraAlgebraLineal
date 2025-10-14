from PySide6.QtWidgets import QWidget, QLabel, QTabWidget, QVBoxLayout, QTextEdit
from .matrix_widgets import AugmentedSystemWidget, VectorPropertiesWidget, LinearCombinationWidget, MatrixEquationWidget, VectorArithmeticWidget, SystemsSolverWidget, LinearDependenceWidget, MatrixOpsWidget

class HomeView(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        tabs = QTabWidget()

        # Dashboard tab with welcome and instructions
        dashboard = QTextEdit()
        dashboard.setReadOnly(True)
        dashboard.setHtml('''
        <h1>Bienvenido a la Calculadora de Álgebra Lineal</h1>
        <p>Esta herramienta le permite practicar y resolver operaciones de álgebra lineal.</p>
        <h2>¿Qué puedes hacer aquí?</h2>
        <ul>
            <li><b>Matrices:</b> suma, resta, multiplicación por escalar, producto, transposición, identidad.</li>
            <li><b>Operaciones elementales:</b> filas y columnas, operaciones elementales.</li>
            <li><b>Sistemas de ecuaciones lineales:</b> matrices aumentadas, resolución paso a paso.</li>
            <li><b>Forma escalonada y reducida:</b> método de Gauss–Jordan con pasos tipo pizarrón.</li>
        </ul>
        <h2>¿Cómo usar la calculadora?</h2>
        <ol>
            <li>Selecciona la pestaña de la función que deseas usar.</li>
            <li>Ingresa los valores en las casillas correspondientes. Puedes usar fracciones (ejemplo: 2/3) o decimales (ejemplo: 1.5).</li>
            <li>Presiona el botón correspondiente para realizar la operación.</li>
            <li>Observa los resultados y, si corresponde, los pasos intermedios.</li>
        </ol>
        <p>¡Explora las pestañas y aprende álgebra lineal de manera interactiva!</p>
        ''')
        tabs.addTab(dashboard, 'Dashboard')

        # Tabs para cada funcionalidad
        tabs.addTab(VectorArithmeticWidget(), 'Suma/Resta/Escalar de Vectores')
        tabs.addTab(MatrixOpsWidget(), 'Operaciones con Matrices')
        tabs.addTab(VectorPropertiesWidget(), 'Propiedades de ℝⁿ')
        tabs.addTab(LinearCombinationWidget(), 'Combinación Lineal / Ecuación Vectorial')
        tabs.addTab(MatrixEquationWidget(), 'Ecuación Matricial (AX=B)')
        tabs.addTab(SystemsSolverWidget(), 'Sistemas AX=0 / AX=B (paso a paso)')
        tabs.addTab(LinearDependenceWidget(), 'Dependencia Lineal (R^n)')
        tabs.addTab(AugmentedSystemWidget(), 'Sistemas de ecuaciones (Gauss–Jordan)')
        v.addWidget(tabs)
