from abc import ABC, abstractmethod
from typing import Callable, Tuple, Optional, Union
from enum import Enum
from logger import *


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

    def semantic_check(self, context: 'Context' = None):
        for each in self.childs:
            each.semantic_check(context)

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


class VarType(Enum):
    Global = 'global'
    Local = 'local'
    Param = 'param'


class Context(object):
    def __init__(self, parent: 'Context' = None):
        self.vars = []
        self.parent = parent

    def add_var(self, var: 'IdentNode'):
        self.vars.append(var)

    def get_var(self, var_name):
        loc_var = None
        loc_context = self
        while not loc_var and loc_context:
            loc_var = list(filter(lambda x: x.name == var_name, loc_context.vars))
            if loc_var:
                loc_var = loc_var[0]
            loc_context = loc_context.parent

        if loc_var:
            return loc_var


class BaseTypes(Enum):
    Int = 'int'
    Float = 'float'
    String = 'str'


class FunctionTypes(Enum):
    Int = 'int'
    Float = 'float'
    String = 'str'
    Void = 'void'


class Type(AstNode):
    def __init__(self, data_type: str, rang: int = 0,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

        if data_type in (e.value for e in BaseTypes):
            self.name = BaseTypes(data_type)
        else:
            self.name = None
            logger.error(data_type + ': неизвестный тип')
        self.rang = rang

    def is_castable_to(self, cast_type)->bool:
        if cast_type.name == self.name:
            return True
        elif cast_type.name == BaseTypes.String:
            return True
        elif cast_type.name == BaseTypes.Float:
            return self.name == BaseTypes.Int
        elif cast_type.name == BaseTypes.Int:
            return False
        return False

    def __str__(self) -> str:
        return self.name.value + ('[]' if self.rang else '')


class ExprNode(AstNode):
    def __init__(self, data_type: Type = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type


class LiteralNode(ExprNode):
    def __init__(self, literal: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.literal = literal
        if literal in ('true', 'false'):
            literal = literal.title()
        self.value = eval(literal)
        self.data_type = Type(type(self.value).__name__)

    def __str__(self) -> str:
        return '{0} ({1})'.format(self.literal, type(self.value).__name__)


class IdentNode(ExprNode):
    def __init__(self, name: str, data_type: Type = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = str(name)
        self.index = None
        self.data_type = data_type
        self.var_type: VarType = None

    def semantic_check(self, context: Context = None):
        v: IdentNode = context.get_var(self.name)
        if not v:
            logger.error(self.name + ': необъявленный идентификатор')
        elif v != self:
            self.index = v.index
            self.var_type = v.var_type
            self.data_type = v.data_type

    def __str__(self) -> str:
        if self.index is not None and self.data_type and self.var_type:
            return str(self.name + ' (' + 'dtype=' + self.data_type.name.value +
                        ', vtype=' + self.var_type.value + ', index=' + str(self.index) + ')')
        else:
            return str(self.name)


class CastNode(ExprNode):
    def __init__(self, var: ExprNode, data_type: Type,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.data_type = data_type

    @property
    def childs(self) -> Tuple[ExprNode, Type]:
        return self.var, self.data_type

    def __str__(self) -> str:
        return 'cast'


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

    def semantic_check(self, context: Context = None):
        for each in self.childs:
            each.semantic_check(context)
        if self.arg1.data_type and self.arg2.data_type:
            if self.arg1.data_type.name == self.arg2.data_type.name:
                self.data_type = self.arg1.data_type
            elif self.arg1.data_type.is_castable_to(self.arg2.data_type):
                self.data_type = self.arg2.data_type
                self.arg1 = CastNode(self.arg1, self.data_type)
            elif self.arg2.data_type.is_castable_to(self.arg1.data_type):
                self.data_type = self.arg1.data_type
                self.arg2 = CastNode(self.arg2, self.data_type)

    @property
    def childs(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def __str__(self) -> str:
        if self.data_type:
            return str(self.op.value) + ' (dtype=' + str(self.data_type) + ')'
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
    def childs(self) -> Tuple[Type, ...]:
        return self.vars_type, (*self.vars_list)

    def __str__(self) -> str:
        return 'var'

    def semantic_check(self, context: Context = None):
        for el in self.vars_list:
            if type(el) is IdentNode:
                el: IdentNode
                context.add_var(el)
                el.index = len(context.vars) - 1
                el.data_type = self.vars_type
                el.var_type = VarType.Local if context.parent else VarType.Global
            elif type(el) is AssignNode:
                el: AssignNode
                context.add_var(el.var)
                el.var.index = len(context.vars) - 1
                el.var.data_type = self.vars_type
                el.var.var_type = VarType.Local if context.parent else VarType.Global

        for each in self.childs:
            each.semantic_check(context)


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
        return tuple(IdentNode(param.childs[1], Type(str(param.vars_type))) for param in params)

    @property
    def childs(self) -> Tuple[AstNode]:
        return (*self.body,)

    def __str__(self) -> str:
        params = ''
        if len(self.params):
            params += str(self.params[0].data_type) + ' ' + self.params[0].name
            for item in self.params[1:]:
                params += ', ' + str(item.data_type) + ' ' + item.name
        s = ''
        for e in self.params:
            if e.data_type:
                if s:
                    s += ', '
                s += '(dtype=' + str(e.data_type) + ', vtype=' + str(e.var_type.value) + ', index=' + str(e.index) + ')'
        if s:
            return str(self.var) + ' ' + str(self.name) + '(' + params + ') (' + s + '}'
        else:
            return str(self.var) + ' ' + str(self.name) + '(' + params + ')'

    def semantic_check(self, context: Context = None):
        context = Context(context)
        for el in self.params:
            context.add_var(el)
            el.index = len(context.vars) - 1
            el.var_type = VarType.Param
        for each in self.childs:
            each.semantic_check(context)


class AssignNode(StmtNode):
    def __init__(self, var: IdentNode, val: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.val = val

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.var, self.val

    def semantic_check(self, context: Context = None):
        # if self.var.type != self.val.type:
        for each in self.childs:
            each.semantic_check(context)

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
    def __init__(self, data_type: Type, num: ExprNode = None, *elements: Tuple[ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.num = num if num else _empty
        self.elements = elements if elements else (_empty,)

    @property
    def childs(self) -> Tuple[AstNode]:
        return (*self.elements,)

    def __str__(self) -> str:
        return self.data_type.name.value + '[' + str(self.num) + ']'


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

    def semantic_check(self, context: 'Context' = None):
        context = Context(context)
        for each in self.childs:
            each.semantic_check(context)


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

    def semantic_check(self, context: 'Context' = None):
        context = Context(context)
        for each in self.childs:
            each.semantic_check(context)


class WhileNode(StmtNode):
    def __init__(self, cond: ExprNode, body: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.body = body

    @property
    def childs(self) -> Tuple[ExprNode, StmtNode]:
        return self.cond, self.body

    def __str__(self) -> str:
        return 'while'

    def semantic_check(self, context: 'Context' = None):
        self.cond.semantic_check(context)
        context = Context(context)
        self.body.semantic_check(context)


class DoWhileNode(StmtNode):
    def __init__(self, body: StmtNode, cond: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.body = body
        self.cond = cond

    @property
    def childs(self) -> Tuple[StmtNode, ExprNode]:
        return self.body, self.cond

    def __str__(self) -> str:
        return 'do while'

    def semantic_check(self, context: 'Context' = None):
        context = Context(context)
        for each in self.childs:
            each.semantic_check(context)


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

    def semantic_check(self, context: Context = None):
        for each in self.childs:
            each.semantic_check(context)

    def __str__(self) -> str:
        return '...'


class Program(StatementListNode):
    def __init__(self, *exprs: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(exprs=exprs, row=row, line=line, **props)
        self.exprs = exprs

    def semantic_check(self, context: Context = None):
        context = Context()
        for each in self.childs:
            each.semantic_check(context)


_empty = StatementListNode()
