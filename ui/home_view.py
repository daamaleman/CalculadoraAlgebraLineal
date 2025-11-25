from PySide6.QtWidgets import QWidget, QLabel, QTabWidget, QVBoxLayout, QTextEdit
from .matrix_widgets import AugmentedSystemWidget, VectorPropertiesWidget, LinearCombinationWidget, MatrixEquationWidget, VectorArithmeticWidget, SystemsSolverWidget, LinearDependenceWidget, MatrixOpsWidget, MatrixInverseWidget, DeterminantWidget
from .errores_view import ErrorsWidget
from .numericos_view import NumericalMethodsWidget

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
        <p>Explora operaciones y propiedades clave de álgebra lineal con resultados exactos y pasos tipo pizarrón.</p>

        <h2>Funciones destacadas</h2>
        <ul>
            <li>
                <b>Operaciones con matrices:</b>
                suma (A+B), resta (A−B), escalar (k·A), producto (A·B), traspuesta (A<sup>T</sup>) e identidad (I).
                Verificaciones incluidas:
                <ul>
                    <li>(A+B)<sup>T</sup> = A<sup>T</sup> + B<sup>T</sup></li>
                    <li>(A−B)<sup>T</sup> = A<sup>T</sup> − B<sup>T</sup></li>
                    <li>(AB)<sup>T</sup> = B<sup>T</sup> A<sup>T</sup></li>
                    <li>En k·A: se muestra (k·A)<sup>T</sup> y la propiedad de rango r(A) = r(A<sup>T</sup>).</li>
                </ul>
            </li>
            <li>
                <b>Inversa Matriz:</b>
                cálculo por Gauss–Jordan de [A | I] → [I | A<sup>−1</sup>], con pasos detallados. Incluye:
                <ul>
                    <li>Determinante det(A) y clasificación: <i>singular</i> si det(A)=0 (no invertible) vs <i>no singular</i> si det(A)≠0 (invertible).</li>
                    <li>Propiedades de invertibilidad: (c) pivotes en cada fila, (d) única solución a AX=0 (trivial), (e) columnas linealmente independientes.</li>
                    <li>Mensaje explícito cuando A <u>no tiene pivote en cada fila</u>.</li>
                </ul>
            </li>
            <li>
                <b>Sistemas y Gauss–Jordan:</b>
                resolución de AX=0 y AX=B con matrices aumentadas y forma escalonada reducida, mostrando cada operación elemental.
            </li>
            <li>
                <b>Vectores y ℝ<sup>n</sup>:</b>
                suma, resta y escalar de vectores; combinación lineal y ecuación vectorial; propiedades en ℝ<sup>n</sup>; dependencia lineal.
            </li>
            <li>
                <b>Determinante de una matriz:</b> pestaña "Determinante de una Matriz" con métodos educativos y pasos:
                <ul>
                    <li><b>Cramer (2×2):</b> det(A) = ad − bc (ilustrativo para 2×2).</li>
                    <li><b>Regla de Sarrus (3×3):</b> suma de diagonales principales menos secundarias.</li>
                    <li><b>Expansión por cofactores (n×n):</b> desarrollo con signos y menores, para cualquier tamaño.</li>
                </ul>
                Se muestra det(A), interpretación de invertibilidad y verificación de propiedades (filas/columnas cero o proporcionales ⇒ det=0, cambio de signo por intercambio de filas, efecto de k en una fila, y det(AB)=det(A)·det(B)).
            </li>
        </ul>

        <h2>Cómo usarla</h2>
        <ol>
            <li>Ve a la pestaña de la función que necesites.</li>
            <li>Ingresa matrices/vectores. Puedes usar fracciones (p. ej., 2/3) o decimales (p. ej., 1.5).</li>
            <li>Ejecuta la operación y revisa los resultados y los pasos intermedios cuando apliquen.</li>
        </ol>

        <p>Consejo: cuando compares propiedades (por ejemplo, traspuestas), la herramienta muestra ambos lados y si son iguales.</p>

        <h2>Mini ayuda: determinante</h2>
        <p>Para calcular y estudiar det(A), abre la pestaña <b>“Determinante de una Matriz”</b>:</p>
        <ol>
            <li>Elige el tamaño n y el método (Cramer 2×2, Sarrus 3×3 o Cofactores n×n).</li>
            <li>Ingresa A (puedes <b>copiar/pegar</b> matrices entre pestañas).</li>
            <li>Presiona <b>Calcular determinante</b> para ver los <i>pasos</i> y el <i>valor</i>.</li>
        </ol>
        <p>Además, la vista verifica propiedades teóricas y ofrece un bloque para comprobar <b>det(AB) = det(A)·det(B)</b> ingresando una matriz B.</p>
        ''')
        tabs.addTab(dashboard, 'Dashboard')

        # Tabs para cada funcionalidad
        tabs.addTab(VectorArithmeticWidget(), 'Suma/Resta/Escalar de Vectores')
        tabs.addTab(MatrixOpsWidget(), 'Operaciones con Matrices')
        tabs.addTab(MatrixInverseWidget(), 'Inversa Matriz')
        tabs.addTab(VectorPropertiesWidget(), 'Propiedades de ℝⁿ')
        tabs.addTab(LinearCombinationWidget(), 'Combinación Lineal / Ecuación Vectorial')
        tabs.addTab(MatrixEquationWidget(), 'Ecuación Matricial (AX=B)')
        tabs.addTab(DeterminantWidget(), 'Determinante de una Matriz')
        tabs.addTab(SystemsSolverWidget(), 'Sistemas AX=0 / AX=B (paso a paso)')
        tabs.addTab(LinearDependenceWidget(), 'Dependencia Lineal (R^n)')
        tabs.addTab(AugmentedSystemWidget(), 'Sistemas de ecuaciones (Gauss–Jordan)')
        tabs.addTab(ErrorsWidget(), 'Errores numéricos')
        tabs.addTab(NumericalMethodsWidget(), 'Métodos numéricos')
        v.addWidget(tabs)
