import functools
import itertools
import math
import operator


def superscript(x, /) -> str:
    """Superscript version of a numeric string."""
    return str(x).translate(str.maketrans('0123456789', '⁰¹²³⁴⁵⁶⁷⁸⁹'))


def rstrip(seq, *items):
    """Copy of subscriptable sequence `seq` with any of `items` removed from the end."""
    return rstrip(seq[:-1], *items) if (seq and seq[-1] in items) else seq


def groupwise(iterable, size: int = 3) -> tuple:
    """Iterator of overlapping subgroups of size `size` taken from `iterable`."""
    iterator = iter(iterable)
    group = [item for i, item in zip(range(size-1), iterator)]
    for i in iterator:
        group.append(i)
        yield tuple(group)
        group.pop(0)


def stringify_power(base: int, power: int) -> str:
    """A pretty string representation of expression in the form `b^p`."""
    if power == 0:
        return ''
    if power == 1:
        return str(base)
    return f'{base}{superscript(power)}'


def stringify_poly_term(a: int, p: int) -> str:
    """A pretty string representing polynomial's term in the format `±ax^p`."""
    if a == 0:
        return ''
    power = stringify_power('x', p)
    sign = '-' if a < 0 else '+'
    c = abs(a) if abs(a) != 1 or not power else ''
    return f"{sign}{c}{power}"


def convolution(a, b):
    """List resulting from the convolution of sequences `a` and `b`."""
    output = [0 for _ in range(len(a) + len(b) - 1)]
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            output[i+j] += x*y
    return output


class Function:
    """Abstract base class of pure mathematical function."""

    def __contains__(self, point):
        x, y = point
        return y == self(x)

    # not well tested
    def bisect(self, left_bound, right_bound):
        """Approximation of the root in the interval (left_bound, right_bound) by bisection method."""
        while right_bound - left_bound:
            mid = (left_bound + right_bound) / 2
            if self(left_bound) * self(mid) < 0:
                right_bound = mid
            elif self(left_bound) * self(mid) > 0:
                left_bound = mid
            else:
                return mid
        return (left_bound + right_bound) / 2

    def newton(self, x):
        """Approximation of the root by Newton's method, starting at `x`."""
        velocity = self.derivative()
        hist = set()
        while x not in hist:
            hist.add(x)
            x -= self(x) / velocity(x)
        return x

    def integral(self, l, r):
        """Definite integral over [l, r]."""
        antiderivative = self.derivative(-1)
        return antiderivative(r) - antiderivative(l)

    def riemann(self, l, r, n):
        """Riemann summ with `n` rectangles over [l, r]."""
        dx = (r - l) / n
        return sum(map(lambda i: self(l + dx * (i + 0.5)) * dx, range(n)))

    def trapezoids(self, l, r, n):
        """Approximation of the definite integral with `n` trapezoids over [l, r]."""
        dx = (r - l) / n
        points = list(map(lambda i: self(l + dx * i), range(n + 1)))
        points[0] /= 2
        points[-1] /= 2
        return sum(points) * dx

    def avg(self, l, r):
        """Average value over [l, r]."""
        return self.integral(l, r) / (r - l)

    def taylor(self, x, degree):
        """Approximation with Taylor's polynomial of degree `degree` at `x`."""
        return sum(map(lambda n: self.derivative(n)(x) / math.factorial(n) * pow(Poly(-x, 1), n), range(degree + 1)), Poly())


class Poly(Function):
    """A polynomial function."""

    def __init__(self, *factors):
        self.factors = rstrip(factors, 0) or (0,)

    def __call__(self, x):
        return sum(map(lambda a, p: a*x**p, self.factors, itertools.count()))

    def __eq__(self, other):
        return self.factors == other.factors if isinstance(other, Poly) else NotImplemented

    def __ne__(self, other):
        return self.factors != other.factors if isinstance(other, Poly) else NotImplemented

    def __hash__(self):
        return hash(self.factors)

    def __add__(self, other):
        if isinstance(other, Poly):
            return Poly(*itertools.starmap(operator.add, itertools.zip_longest(self.factors, other.factors, fillvalue=0)))
        else:
            try:
                return Poly(self.factors[0] + other, *self.factors[1:])
            except TypeError:
                return NotImplemented

    __radd__ = __add__

    def __pos__(self):
        return self

    def __neg__(self):
        return Poly(*map(operator.neg, self.factors))

    def __sub__(self, other):
        return self + -other

    def __rsub__(self, other):
        return other + -self

    def __mul__(self, other):
        if isinstance(other, Poly):
            return Poly(*convolution(self.factors, other.factors))
        else:
            try:
                return Poly(*map(lambda x: other * x, self.factors))
            except TypeError:
                return NotImplemented

    __rmul__ = __mul__

    # almost works
    # def __divmod__(self, other: Poly | int | float):
    #     if type(other) in (int, float):
    #         return self / other, Poly()
    #     if type(other) != Poly:
    #         return NotImplemented
    #     if self.degree() < other.degree():
    #         return Poly(), self
    #     factor = self.factors[-1] / other.factors[-1]
    #     div, mod = divmod(self - other * Poly(0, 1) ** (self.degree() - other.degree()) * factor, other)
    #     return Poly(*div.factors, factor), mod

    #     # if self.degree() == other.degree():
    #     #     return 
    #     # 


    def __truediv__(self, factor: int | float):
        if type(factor) not in (int, float):
            return NotImplemented
        return Poly(*map(lambda x: x / factor, self.factors))

    def __floordiv__(self, other):
        return divmod(self, other)[0]

    def __mod__(self, other):
        return divmod(self, other)[1]

    def __pow__(self, exp: int):
        return functools.reduce(operator.mul, itertools.repeat(self, exp), Poly(1))

    def __round__(self, precision=None):
        return Poly(*map(lambda factor: round(factor, precision), self.factors))

    def __repr__(self):
        return f'Poly{self.factors}'

    def __str__(self):
        output = ''.join(map(stringify_poly_term, reversed(self.factors), itertools.count(self.degree(), -1)))
        return output[1:] if output[0] == '+' else output

    @classmethod
    def with_roots(cls, *roots):
        """New polynomial created with roots `roots`."""
        return functools.reduce(lambda poly, root: poly * Poly(-root, 1), roots, Poly(1))

    def derivative(self, degree: int = 1):
        """A derivative polynomial function of degree `degree` (negative degrees mean antiderivatives)."""
        if degree == 0:
            return self
        elif degree >= 1:
            return Poly(*map(operator.mul, self.factors[1:], itertools.count(1))).derivative(degree-1)
        elif degree <= 1:
            return Poly(0, *map(operator.truediv, self.factors, itertools.count(1))).derivative(degree+1)

    def limit(self, x):
        """Limit as argument approaches `x`."""
        return self.factors[-1] * x ** self.degree() if abs(x) == math.inf else self(x)

    def criticals(self) -> list:
        """List of points in ascending order where the derivative is zero or does not exist."""
        return self.derivative().roots()

    def extrema(self) -> list:
        """List of local minima and maxima in ascending order."""
        points = []
        for l, c, r in groupwise((-math.inf, *self.criticals(), math.inf)):
            if not (self.limit(l) > self(c) and self(c) > self.limit(r)):
                points.append(c)
        return points

    def minima(self) -> list:
        """List of local minima in ascending order."""
        points = []
        for l, c, r in groupwise((-math.inf, *self.criticals(), math.inf)):
            if self.limit(l) > self(c) < self.limit(r):
                points.append(c)
        return points

    def maxima(self) -> list:
        """List of local maxima in ascending order."""
        points = []
        for l, c, r in groupwise((-math.inf, *self.criticals(), math.inf)):
            if self.limit(l) < self(c) > self.limit(r):
                points.append(c)
        return points

    def degree(self) -> int:
        """Degree of the polynomial."""
        return len(self.factors) - 1

    def roots(self) -> list:
        """List of roots in ascending order."""
        match self.degree():
            case 0:  # a = 0
                return []
            case 1:  # ax + b = 0
                b, a = self.factors
                return [-b/a]
            case 2:  # ax^2 + bx + c = 0
                c, b, a = self.factors
                d = b**2 - 4*a*c
                if d < 0:
                    return []
                if d == 0:
                    return [-b / (2*a)]
                sqrt_d = math.sqrt(d)
                return sorted([(-b - sqrt_d) / (2*a), (-b + sqrt_d) / (2*a)])
            case _:
                criticals = self.criticals()
                if len(criticals) == 0:  # strictly increasing or decreasing function
                    return [self.newton(0)]
                else:
                    roots = []
                    # root to the left from the leftmost critical point
                    if self(criticals[0]) * self.limit(-math.inf) < 0:
                        roots.append(self.newton(criticals[0] - 1))
                    # roots on critical points
                    roots.extend(filter(lambda x: self(x) == 0, criticals))
                    # roots between critical points
                    for a, b in itertools.pairwise(criticals):
                        if self(a) * self(b) < 0:
                            roots.append(self.newton((a+b)/2))
                    # root to the right from the rightmost critical point
                    if self(criticals[-1]) * self.limit(math.inf) < 0:
                        roots.append(self.newton(criticals[-1] + 1))
                    roots.sort()
                    return roots
