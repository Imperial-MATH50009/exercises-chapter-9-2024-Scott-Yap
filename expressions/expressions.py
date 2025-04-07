"""Expression module."""
from functools import singledispatch
import numbers


class Expression:
    """Expression class."""

    def __init__(self, *operands):
        """Init method."""
        self.operands = operands

    def __add__(self, other):
        """Add method."""
        if not isinstance(other, Expression):
            other = Number(other)
        return Add(self, other)

    def __sub__(self, other):
        """Sub method."""
        if not isinstance(other, Expression):
            other = Number(other)
        return Sub(self, other)

    def __mul__(self, other):
        """Mul method."""
        if not isinstance(other, Expression):
            other = Number(other)
        return Mul(self, other)

    def __truediv__(self, other):
        """Div method."""
        if not isinstance(other, Expression):
            other = Number(other)
        return Div(self, other)

    def __pow__(self, other):
        """Pow method."""
        if not isinstance(other, Expression):
            other = Number(other)
        return Pow(self, other)

    def __radd__(self, other):
        """Radd method."""
        return self.__add__(other)

    def __rsub__(self, other):
        """Rsub method."""
        return Sub(Number(other), self)

    def __rmul__(self, other):
        """Rmul method."""
        return self.__mul__(other)

    def __rtruediv__(self, other):
        """Rdiv method."""
        return Div(Number(other), self)

    def __rpow__(self, other):
        """Rpow method."""
        return Pow(Number(other), self)


class Terminal(Expression):
    """Terminal subclass."""

    precedence = 0

    def __init__(self, value):
        """Init method."""
        self.value = value
        super().__init__()

    def __repr__(self):
        """Repr method."""
        return repr(self.value)

    def __str__(self):
        """Str method."""
        return str(self.value)


class Number(Terminal):
    """Number subclass."""

    def __init__(self, value):
        """Init method."""
        if not isinstance(value, numbers.Number):
            raise ValueError("Number must be a numeric value.")
        super().__init__(value)


class Symbol(Terminal):
    """Symbol subclass."""

    def __init__(self, value):
        """Initialize."""
        if not isinstance(value, str):
            raise ValueError("Symbol must be a string.")
        super().__init__(value)


class Operator(Expression):
    """Operator subclass."""

    precedence = 0

    def __repr__(self):
        """Repr method."""
        return type(self).__name__ + repr(self.operands)

    def __str__(self):
        """Set string."""
        def bracket(operand, precedence):
            if isinstance(operand, Operator) and operand.precedence \
                    < precedence:
                return f"({operand})"
            return str(operand)

        first = bracket(self.operands[0], self.precedence)
        second = bracket(self.operands[1], self.precedence)
        return f"{first} {self.symbol} {second}"


class Add(Operator):
    """Add subclass."""

    symbol = '+'
    precedence = 1


class Sub(Operator):
    """Sub subclass."""

    symbol = '-'
    precedence = 1


class Mul(Operator):
    """Mul subclass."""

    symbol = '*'
    precedence = 2

    def __str__(self):
        """Str method."""
        a, b = self.operands
        # Reorder to put the Number first if applicable
        if isinstance(b, Number) and not isinstance(a, Number):
            a, b = b, a

        def bracket(operand, precedence):
            if isinstance(operand, Operator) \
                    and operand.precedence < precedence:
                return f"({operand})"
            return str(operand)
        return (
            f"{bracket(a, self.precedence)} "
            f"{self.symbol} "
            f"{bracket(b, self.precedence)}"
        )


class Div(Operator):
    """Div subclass."""

    symbol = '/'
    precedence = 2


class Pow(Operator):
    """Pow subclass."""

    symbol = '^'
    precedence = 3


# Postvisitor Function
def postvisitor(expr, visitor, **kwargs):
    """
    Post-order traversal of an expression tree.

    Visiting each subexpression only once.
    """
    stack = [expr]
    visited = {}

    while stack:
        e = stack.pop()
        unvisited_children = [o for o in e.operands if o not in visited]

        if unvisited_children:
            stack.append(e)
            stack.extend(unvisited_children)
        else:
            visited[e] = visitor(
                e,
                *(visited[o] for o in e.operands),  # Operand results
                **kwargs  # Additional arguments
            )

    return visited[expr]


# Differentiation Functions
@singledispatch
def differentiate(expr, *, var):
    """Differentiate an expression with respect to a given variable."""
    raise NotImplementedError(
        f"Cannot differentiate a {type(expr).__name__}"
    )


@differentiate.register(Number)
def _(expr, *, var):
    return Number(0)  # Derivative of a constant is 0


@differentiate.register(Symbol)
def _(expr, *, var):
    return Number(1) if expr.value == var else Number(0)
# 1 if variable matches


@differentiate.register(Add)
def _(expr, *operands, var):
    # Differentiate sum: (f + g)' = f' + g'
    return Add(*[differentiate(op, var=var) for op in expr.operands])


@differentiate.register(Mul)
def _(expr, *operands, var):
    # Differentiate product: (f * g)' = f' * g + f * g'
    f, g = expr.operands
    return Add(
        Mul(differentiate(f, var=var), g),  # f' * g
        Mul(differentiate(g, var=var), f)   # f * g'
    )


@differentiate.register(Div)
def _(expr, *operands, var):
    # Differentiate division: (f / g)' = (f' * g - f * g') / g^2
    f, g = expr.operands
    return Div(
        Sub(
            Mul(differentiate(f, var=var), g),  # f' * g
            Mul(differentiate(g, var=var), f)   # f * g'
        ),
        Pow(g, Number(2))  # g^2
    )


@differentiate.register(Pow)
def _(expr, *operands, var):
    """Differentiate f^n: (f^n)' = n * f^(n-1) * f'."""
    base, exponent = expr.operands
    if isinstance(exponent, Number):  # Constant exponent
        return Mul(
            Mul(exponent, Pow(base, Sub(exponent, Number(1)))),  # n * f^(n-1)
            differentiate(base, var=var)  # * f'
        )
    else:
        raise NotImplementedError("Variable exponents are not yet supported.")
