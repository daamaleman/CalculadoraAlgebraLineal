class Vector:
    def __init__(self, values):
        self.values = list(map(float, values))

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.values, other.values)])

    def __mul__(self, scalar):
        return Vector([scalar * a for a in self.values])

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __neg__(self):
        return Vector([-a for a in self.values])

    def __eq__(self, other):
        return all(abs(a - b) < 1e-9 for a, b in zip(self.values, other.values))

    def is_zero(self):
        return all(abs(a) < 1e-9 for a in self.values)

    @staticmethod
    def zero(n):
        return Vector([0.0] * n)

    @staticmethod
    def check_commutative(v1, v2):
        return v1 + v2 == v2 + v1

    @staticmethod
    def check_associative(v1, v2, v3):
        return (v1 + (v2 + v3)) == ((v1 + v2) + v3)

    @staticmethod
    def check_zero(v):
        zero = Vector.zero(len(v.values))
        return v + zero == v

    @staticmethod
    def check_opposite(v):
        return v + (-v) == Vector.zero(len(v.values))

    def __repr__(self):
        return f"Vector({self.values})"
