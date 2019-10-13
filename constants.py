"""Constant values"""

from enum import Enum


class VarType(Enum):
    """Variable types"""

    Global = 'global'
    Local = 'local'
    Param = 'param'


class BaseTypes(Enum):
    """Base Types"""

    Int = 'int'
    Double = 'double'
    # String = 'string'
    Bool = 'bool'
    Void = 'void'
    Char = 'char'


class BinOp(Enum):
    """Binary operations"""

    ADD = '+'
    SUB = '-'
    MOD = '%'
    MUL = '*'
    DIV = '/'
    GE = '>='
    LE = '<='
    NEQUALS = '!='
    EQUALS = '=='
    GT = '>'
    LT = '<'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'


class UnOp(Enum):
    """Unary operations"""

    NOT = '!'
    SUB = '-'
    ADD = '+'


LLVM_TYPES = {
    'int': ['i32', 4],
    'double': ['double', 8],
    'bool': ['i8', 1],
    'void': ['void', ],
    'char': ['i8', 1],
    'char[]': ['i8*', 8]
}

DEFAULT_VALUES = {
    'int': '0',
    'double': '0.0',
    'bool': '0',
}

LLVM_INT_OPS = {
    '/': 'sdiv',
    '+': 'add',
    '-': 'sub',
    '*': 'mul',
    '%': 'srem',
    '>': 'icmp sgt',
    '<': 'icmp slt',
    '==': 'icmp eq',
    '>=': 'icmp sge',
    '<=': 'icmp sle',
    '!=': 'icmp ne',
}

LLVM_FLOAT_OPS = {
    '/': 'fdiv',
    '+': 'fadd',
    '-': 'fsub',
    '*': 'fmul',
    '%': 'frem',
    '>': 'fcmp ogt',
    '<': 'fcmp olt',
    '==': 'fcmp oeq',
    '>=': 'fcmp oge',
    '<=': 'fcmp ole',
    '!=': 'fcmp one',
}
