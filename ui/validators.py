from PySide6.QtGui import QRegularExpressionValidator, QIntValidator
from PySide6.QtCore import QRegularExpression

def number_validator():
    # Accept integers, decimals with dot/comma, and fractions a/b
    regex = QRegularExpression(r'^(-?(?:\d+(?:[.,]\d+)?)|-?\d+/\d+)$')
    return QRegularExpressionValidator(regex)

def int_validator(minv=1, maxv=12):
    return QIntValidator(minv, maxv)
