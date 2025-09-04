from PySide6.QtWidgets import QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout, QTextEdit, QMessageBox
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt, Signal

class LoginView(QWidget):
    authenticated = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)

        # Left: description (español)
        desc = QTextEdit()
        desc.setReadOnly(True)
        desc.setFont(QFont('Segoe UI', 11))
        desc.setHtml('''
        <h2>Calculadora de Álgebra Lineal</h2>
        <p>Bienvenido. Esta herramienta le permite practicar y resolver operaciones de álgebra lineal de acuerdo con las sesiones:</p>
        <ul>
            <li><b>Sesión 1:</b> Matrices (suma, resta, escalar, producto, transposición, identidad).</li>
            <li><b>Sesión 1.2:</b> Propiedades de filas y columnas; operaciones elementales.</li>
            <li><b>Sesión 2:</b> Sistemas de ecuaciones lineales y matrices aumentadas.</li>
            <li><b>Sesión 3:</b> Forma escalonada y forma escalonada reducida (Gauss–Jordan).</li>
        </ul>
        <p>Además, podrá ver <b>paso a paso</b> las operaciones con matrices como en el pizarrón.</p>
        ''')
        layout.addWidget(desc, 2)

        # Right: login form (español)
        form = QVBoxLayout()
        title = QLabel('Iniciar sesión')
        title.setFont(QFont('Segoe UI', 14, QFont.Bold))
        form.addWidget(title)
        self.user = QLineEdit(); self.user.setPlaceholderText('Usuario')
        self.pwd = QLineEdit(); self.pwd.setPlaceholderText('Contraseña'); self.pwd.setEchoMode(QLineEdit.Password)
        self.btn = QPushButton('Entrar')
        form.addWidget(self.user)
        form.addWidget(self.pwd)
        form.addWidget(self.btn)
        form.addStretch()
        right = QWidget(); right.setLayout(form)
        layout.addWidget(right, 1)

        self.btn.clicked.connect(self.try_login)

    def try_login(self):
        u = self.user.text().strip(); p = self.pwd.text().strip()
        if not u or not p:
            QMessageBox.warning(self, 'Validación', 'Por favor ingrese usuario y contraseña.')
            return
        if len(p) < 3:
            QMessageBox.warning(self, 'Validación', 'La contraseña debe tener al menos 3 caracteres.')
            return
        self.authenticated.emit(u)
