# 📘 Calculadora de Álgebra Lineal

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)  
![Framework](https://img.shields.io/badge/Framework-PySide6-green?logo=qt)  
![Estado](https://img.shields.io/badge/Estado-En%20Desarrollo-orange)

---

## 🎯 Objetivo
Desarrollar una **calculadora de Álgebra Lineal** con interfaz gráfica en español y código modular en inglés, que permita resolver sistemas de ecuaciones lineales y operaciones con matrices.  
Este proyecto integra los contenidos de la **Unidad I: Ecuaciones Lineales en Álgebra Lineal**, combinando teoría y práctica con Python:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 📂 Estructura del Proyecto

app/
├── core/ # Lógica matemática
│ ├── matrix.py → Operaciones básicas con matrices
│ ├── formatter.py → Mostrar matrices “como en el pizarrón”
│ └── linsys.py → Resolver sistemas (Gauss–Jordan)
│
├── ui/ # Interfaz gráfica en español
│ ├── theme.py → Tema oscuro con #0099A8
│ ├── validators.py → Validación de entradas
│ ├── matrix_widgets.py → Captura y resolución de sistemas [A|b]
│ ├── login_view.py → Pantalla de inicio de sesión
│ └── home_view.py → Navegación por pestañas
│
└── main.py # Punto de entrada

---

## ✨ Funcionalidades

### 🔑 Login
- Pantalla inicial con descripción de lo que hace la calculadora.  
- Validación en tiempo real (usuario y contraseña).

### ➕ Matrices
- Suma, resta y multiplicación.  
- Producto por un escalar.  
- Transpuesta e identidad.  
- Representación visual con notación matemática:contentReference[oaicite:2]{index=2}.

### 🔄 Operaciones Elementales
- Intercambio de filas.  
- Multiplicación de una fila por un escalar.  
- Combinaciones lineales de filas:contentReference[oaicite:3]{index=3}.  

### 📊 Sistemas de Ecuaciones
- Planteamiento de sistemas lineales (2x2, 3x3, …).  
- Resolución paso a paso con **Gauss–Jordan**.  
- Clasificación automática:
  - ✅ Única solución  
  - ♾️ Infinitas soluciones  
  - ❌ Sistema inconsistente:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}

---

## 🚀 Instalación y Uso

### Requisitos
- Python 3.10 o superior  
- Dependencias:
```bash
pip install PySide6
Ejecución
bash
Copiar código
python -m app.main

📚 Contenidos Relacionados
La aplicación integra los temas trabajados en las sesiones de clase:

Matrices – definición, notación y operaciones básicasSesion 1_Matrices_Parte….

Propiedades de filas y columnas – operaciones elementalesSesion 1_Propiedades de….

Ecuaciones lineales – concepto, notación matricial y clasificaciónSesion 2_Ecuaciones Lin….

Forma escalonada y reducida – pivotes, variables básicas y libresSesion 3_Forma escalona….

🏫 Contexto Académico
Este proyecto se desarrolla en el marco de la asignatura Álgebra Lineal – Unidad I (Ecuaciones Lineales),
Facultad de Ingeniería y Arquitectura – UAM Managua, bajo la coordinación del docente José Andrés Munguía Cortez