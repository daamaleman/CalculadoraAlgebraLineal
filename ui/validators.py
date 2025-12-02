from PySide6.QtGui import QRegularExpressionValidator, QIntValidator
from PySide6.QtCore import QRegularExpression

def number_validator():
    # Permite enteros, decimales, fracciones y exponentes simples, incluyendo superíndices unicode.
    # Nota: la validación final se hace al parsear; aquí solo permitimos teclear.
    supers = '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁽⁾'
    # Patrón permisivo: dígitos, espacio, coma/punto, slash, paréntesis, ^ y superíndices
    pattern = r'^[\s0-9.,/()^\-+' + supers + r']*$'
    regex = QRegularExpression(pattern)
    return QRegularExpressionValidator(regex)

def int_validator(minv=1, maxv=12):
    return QIntValidator(minv, maxv)
