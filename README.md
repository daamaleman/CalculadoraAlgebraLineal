# 📘 Calculadora de Álgebra Lineal

<p align="center">
    <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" alt="Python">
    <img src="https://img.shields.io/badge/Framework-PySide6-green?logo=qt" alt="Framework">
    <img src="https://img.shields.io/badge/Estado-En%20Desarrollo-orange" alt="Estado">
</p>

---

## 🎯 Objetivo

Desarrollar una **calculadora de Álgebra Lineal** con interfaz gráfica en español y código modular en inglés, capaz de resolver sistemas de ecuaciones lineales y operaciones con matrices.  
Este proyecto integra los contenidos de la **Unidad I: Ecuaciones Lineales en Álgebra Lineal**, combinando teoría y práctica con Python.

---

## 📂 Estructura del Proyecto

```text
app/
├── core/                # Lógica matemática
│   ├── matrix.py        # Operaciones básicas con matrices
│   ├── formatter.py     # Mostrar matrices “como en el pizarrón”
│   └── linsys.py        # Resolver sistemas (Gauss–Jordan)
│
├── ui/                  # Interfaz gráfica en español
│   ├── theme.py         # Tema oscuro con #0099A8
│   ├── validators.py    # Validación de entradas
│   ├── matrix_widgets.py# Captura y resolución de sistemas [A|b]
│   ├── login_view.py    # Pantalla de inicio de sesión
│   └── home_view.py     # Navegación por pestañas
│
└── main.py              # Punto de entrada
```

---

## ✨ Funcionalidades

<details>
    <summary><strong>🔑 Login</strong></summary>
    <ul>
        <li>Pantalla inicial con descripción de la calculadora.</li>
        <li>Validación en tiempo real (usuario y contraseña).</li>
    </ul>
</details>

<details>
    <summary><strong>➕ Matrices</strong></summary>
    <ul>
        <li>Suma, resta y multiplicación.</li>
        <li>Producto por un escalar.</li>
        <li>Transpuesta e identidad.</li>
        <li>Inversa de una matriz n×n con pasos de Gauss–Jordan y verificación A·A^{-1}=I.</li>
        <li>Representación visual con notación matemática.</li>
    </ul>
</details>

<details>
    <summary><strong>🔄 Operaciones Elementales</strong></summary>
    <ul>
        <li>Intercambio de filas.</li>
        <li>Multiplicación de una fila por un escalar.</li>
        <li>Combinaciones lineales de filas.</li>
    </ul>
</details>

<details>
    <summary><strong>📊 Sistemas de Ecuaciones</strong></summary>
    <ul>
        <li>Planteamiento de sistemas lineales (2x2, 3x3, …).</li>
        <li>Resolución paso a paso con <b>Gauss–Jordan</b>.</li>
        <li>Clasificación automática:
            <ul>
                <li>✅ Única solución</li>
                <li>♾️ Infinitas soluciones</li>
                <li>❌ Sistema inconsistente</li>
            </ul>
        </li>
    </ul>
</details>

---

## 🚀 Instalación y Uso

### Requisitos

- Python 3.10 o superior
- Dependencias:

```bash
pip install PySide6
```

### Ejecución

```bash
python .\main.py
```

### Nueva pestaña: Inversa Matriz

- Abre la pestaña "Inversa Matriz".
- Ingresa la matriz cuadrada A (puedes usar fracciones como 2/3 o decimales 1.5).
- Presiona "Calcular A^{-1} (Gauss–Jordan)" para ver los pasos sobre la matriz aumentada [A | I].
- Si A es invertible, se mostrará A^{-1} y se verificarán las propiedades A·A^{-1}=I y A^{-1}·A=I.

---

## 📚 Contenidos Relacionados

La aplicación integra los temas trabajados en las sesiones de clase:

- **Matrices** – definición, notación y operaciones básicas  
- **Propiedades de filas y columnas** – operaciones elementales  
- **Ecuaciones lineales** – concepto, notación matricial y clasificación  
- **Forma escalonada y reducida** – pivotes, variables básicas y libres  

---

## 🏫 Contexto Académico

Este proyecto se desarrolla en el marco de la asignatura **Álgebra Lineal – Unidad I (Ecuaciones Lineales)**,  
Facultad de Ingeniería y Arquitectura – UAM Managua, bajo la coordinación del docente José Andrés Munguía Cortez.

---

<p align="center">
    <sub>Desarrollado con ❤️ para la comunidad académica</sub>
</p>