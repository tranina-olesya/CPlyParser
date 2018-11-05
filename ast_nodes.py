from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Union
from enum import Enum


class AstNode(ABC):
    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__()
        self.row = row
        self.line = line
        for k, v in props.items():
            setattr(self, k, v)


    @property
    def childs(self) -> Tuple['AstNode', ...]:
        return ()

    @abstractmethod
    def __str__(self) -> str:
        pass

    @property
    def tree(self) -> [str, ...]:
        res = [str(self)]
        childs_temp = self.childs
        for i, child in enumerate(childs_temp):
            ch0, ch = '├', '│'
            if i == len(childs_temp) - 1:
                ch0, ch = '└', ' '
            res.extend(((ch0 if j == 0 else ch) + ' ' + s for j, s in enumerate(child.tree)))
        return res

    def visit(self, func: Callable[['AstNode'], None]) -> None:
        func(self)
        map(func, self.childs)

    def __getitem__(self, index):
        return self.childs[index] if index < len(self.childs) else None


class ExprNode(AstNode):
    pass


class Type(ExprNode):
    def __init__(self, type: str, rang: int = 0):
        self.base_types = ('int', 'string', 'float')
        self.type = type
        self.rang = rang

    def check(self):
        return self.type in self.base_types

    def __str__(self) -> str:
        return self.type + ('[]' if self.rang else '')


class LiteralNode(ExprNode):
    def __init__(self, literal: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.literal = literal
        if literal in ('true', 'false'):
            literal = literal.title()
        self.value = eval(literal)

    def __str__(self) -> str:
        return '{0} ({1})'.format(self.literal, type(self.value).__name__)


class IdentNode(ExprNode):
    def __init__(self, name: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = str(name)

    def __str__(self) -> str:
        return str(self.name)


class BinOp(Enum):
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
    BIT_AND = '&'
    BIT_OR = '|'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'


class BinOpNode(ExprNode):
    def __init__(self, op: BinOp, arg1: ExprNode, arg2: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2

    @property
    def childs(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def __str__(self) -> str:
        return str(self.op.value)


class UnOp(Enum):
    NOT = '!'
    SUB = '-'
    ADD = '+'
    INC_OP = '++'
    DEC_OP = '--'


class UnOpNode(ExprNode):
    def __init__(self, op: UnOp, arg: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg = arg

    @property
    def childs(self) -> Tuple[ExprNode]:
        return self.arg,

    def __str__(self) -> str:
        return str(self.op.value)


class StmtNode(ExprNode):
    pass


class VarsDeclNode(StmtNode):
    def __init__(self, vars_type: Type, *vars_list: Tuple[AstNode, ...],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars_type = vars_type
        self.vars_list = vars_list

    @property
    def childs(self) -> Tuple[ExprNode, ...]:
        return self.vars_type, (*self.vars_list)

    def __str__(self) -> str:
        return 'var'


class CallNode(StmtNode):
    def __init__(self, func: IdentNode, *params: Tuple[ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.func = func
        self.params = params

    @property
    def childs(self) -> Tuple[IdentNode, ...]:
        return self.func, (*self.params)

    def __str__(self) -> str:
        return 'call'


class Param(object):
    def __init__(self, type: Type, name: IdentNode):
        self.type = type
        self.name = name


class FunctionNode(StmtNode):
    def __init__(self, var: Type, name: IdentNode, params: Tuple = None, *body: Tuple[StmtNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.name = name
        self.params = self.check_params(params) if params else tuple()
        self.body = body if body else (_empty,)

    @staticmethod
    def check_params(params):
        return tuple(Param(Type(str(param.vars_type)), param.childs[1]) for param in params)

    @property
    def childs(self) -> Tuple[AstNode]:
        return (*self.body,)

    def __str__(self) -> str:
        params = ''
        if len(self.params):
            params += self.params[0].type.value + ' ' + self.params[0].name.name
            for item in self.params[1:]:
                params += ', ' + item.type.value + ' ' + item.name.name
        return str(self.var) + ' ' + str(self.name) + '(' + params + ')'


class AssignNode(StmtNode):
    def __init__(self, var: IdentNode, val: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.val = val

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.var, self.val

    def __str__(self) -> str:
        return '='


class ElementNode(ExprNode):
    def __init__(self, name: IdentNode, num: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = name
        self.num = num

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.name, self.num

    def __str__(self) -> str:
        return '[]'


class ArrayIdentNode(ExprNode):
    def __init__(self, type: Type, num: ExprNode = None, *elements: Tuple[ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.type = type
        self.num = num if num else _empty
        self.elements = elements if elements else (_empty,)

    @property
    def childs(self) -> Tuple[AstNode]:
        return (*self.elements,)

    def __str__(self) -> str:
        return self.type.type + '[' + str(self.num) + ']'


class ReturnNode(StmtNode):
    def __init__(self, val: ExprNode = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.val = val

    @property
    def childs(self) -> Tuple[ExprNode]:
        if self.val:
            return self.val,
        else:
            return tuple()

    def __str__(self) -> str:
        return 'return'


class ContinueNode(StmtNode):
    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

    @property
    def childs(self) -> Tuple[ExprNode]:
        return tuple()

    def __str__(self) -> str:
        return 'continue'


class BreakNode(StmtNode):
    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

    @property
    def childs(self) -> Tuple[ExprNode]:
        return tuple()

    def __str__(self) -> str:
        return 'break'



class IfNode(StmtNode):
    def __init__(self, cond: ExprNode, then_stmt: StmtNode, else_stmt: Optional[StmtNode] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.then_stmt = then_stmt if then_stmt else _empty
        self.else_stmt = else_stmt

    @property
    def childs(self) -> Tuple[ExprNode, StmtNode, Optional[StmtNode]]:
        return (self.cond, self.then_stmt) + ((self.else_stmt,) if self.else_stmt else tuple())

    def __str__(self) -> str:
        return 'if'


class ForNode(StmtNode):
    def __init__(self, init: Union[StmtNode, None], cond: Union[ExprNode, StmtNode, None],
                 step: Union[StmtNode, None], body: Union[StmtNode, None] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.init = init if init else _empty
        self.cond = cond if cond else _empty
        self.step = step if step else _empty
        self.body = body if body else _empty

    @property
    def childs(self) -> Tuple[AstNode, ...]:
        return self.init, self.cond, self.step, self.body

    def __str__(self) -> str:
        return 'for'


class WhileNode(StmtNode):
    def __init__(self, cond: ExprNode, body: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.body = body

    @property
    def childs(self) -> Tuple[ExprNode, StmtNode]:
        return (self.cond, self.body)

    def __str__(self) -> str:
        return 'while'


class DoWhileNode(StmtNode):
    def __init__(self, body: StmtNode, cond: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.body = body
        self.cond = cond

    @property
    def childs(self) -> Tuple[StmtNode, ExprNode]:
        return ( self.body, self.cond)

    def __str__(self) -> str:
        return 'do while'


class StatementListNode(StmtNode):
    def __init__(self, *exprs: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs


    def add_child(self, *ch):
        self.exprs += ch

    @property
    def childs(self) -> Tuple[StmtNode, ...]:
        return self.exprs

    def __str__(self) -> str:
        return '...'


_empty = StatementListNode()
