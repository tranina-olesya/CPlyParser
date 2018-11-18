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
        self.functions = [FunctionNode(Type('int'), 'input')]
        self.parent = parent

    def add_var(self, var: 'IdentNode'):
        self.vars.append(var)

    def add_function(self, func: 'FunctionNode'):
        self.functions.append(func)

    def find_var(self, var_name):
        loc_var = None
        loc_context = self
        while not loc_var and loc_context:
            loc_var = list(filter(lambda x: x.name == var_name, loc_context.vars))
            if loc_var:
                loc_var = loc_var[0]
            loc_context = loc_context.parent

        return loc_var

    def find_function(self, func_name):
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        func = list(filter(lambda x: x.name == func_name, loc_context.functions))
        return func[0] if func else None


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
            logger.error(str(self.row) + ': ' + data_type + ': unknown identifier')
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
        return str(self.name.value) + ('[]' if self.rang else '')


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
        self.const = eval(literal)
        self.data_type = Type(type(self.const).__name__, row=self.row)

    def __str__(self) -> str:
        return '{0} ({1})'.format(self.literal, self.data_type)


class IdentNode(ExprNode):
    def __init__(self, name: str, data_type: Type = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.name = str(name)
        self.index = None
        self.data_type = data_type
        self.var_type: VarType = None

    def semantic_check(self, context: Context = None):
        v: IdentNode = context.find_var(self.name)
        if v:
            if v != self:
                self.index = v.index
                self.var_type = v.var_type
                self.data_type = v.data_type
        else:
            v = context.find_function(self.name)
            if v:
                self.data_type = v.data_type
            else:
                logger.error(str(self.row) + ': ' + self.name + ': unknown identifier')

    def __str__(self) -> str:
        if self.index is not None and self.data_type and self.var_type:
            return str(self.name + ' (dtype=' + str(self.data_type) +
                        ', vtype=' + self.var_type.value + ', index=' + str(self.index) + ')')
        elif self.data_type:
            return str(self.name + ' (dtype=' + str(self.data_type) + ')')
        else:
            return str(self.name)


class CastNode(ExprNode):
    def __init__(self, var: ExprNode, data_type: Type, const = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.data_type = data_type
        self.const = const

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
    #BIT_AND = '&'
    #BIT_OR = '|'
    LOGICAL_AND = '&&'
    LOGICAL_OR = '||'


class BinOpNode(ExprNode):
    def __init__(self, op: BinOp, arg1: ExprNode, arg2: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.const = None

    def semantic_check(self, context: Context = None):
        for each in self.childs:
            each.semantic_check(context)
        if self.arg1.data_type and self.arg2.data_type:
            if str(self.arg1.data_type) == str(self.arg2.data_type):
                self.data_type = self.arg1.data_type
            elif self.arg1.data_type.is_castable_to(self.arg2.data_type):
                self.data_type = self.arg2.data_type
                c = None
                if type(self.arg1) in (LiteralNode, BinOpNode, UnOpNode, CastNode):
                    c = self.arg1.const
                self.arg1 = CastNode(self.arg1, self.data_type, c)
            elif self.arg2.data_type.is_castable_to(self.arg1.data_type):
                self.data_type = self.arg1.data_type
                c = None
                if type(self.arg2) in (LiteralNode, BinOpNode, UnOpNode, CastNode):
                    c = self.arg2.const
                self.arg2 = CastNode(self.arg2, self.data_type, c)

            if (type(self.arg1) in (LiteralNode, BinOpNode, UnOpNode, CastNode)) \
                    and (type(self.arg2) in (LiteralNode, BinOpNode , UnOpNode, CastNode))\
                    and self.arg1.const is not None and self.arg2.const is not None and self.data_type:
                if self.op.value == '&&':
                    self.const = 1 if self.arg1.const and self.arg2.const else 0
                elif self.op.value == '||':
                    self.const = 1 if self.arg1.const or self.arg2.const else 0
                else:
                    self.const = eval(str(self.data_type) + '(' + str(self.arg1.const) + self.op.value + str(self.arg2.const) + ')')

    @property
    def childs(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def __str__(self) -> str:
        c = d = sb = ''
        if self.data_type:
            d = 'dtype=' + str(self.data_type)
        if self.const is not None:
            c = 'const=' + str(self.const)
        if d: sb = d
        if c: sb = sb + (', ' if sb else '') + c
        return str(self.op.value) + (' (' + sb + ')' if sb else '')


class UnOp(Enum):
    NOT = '!'
    SUB = '-'
    ADD = '+'


class UnOpNode(ExprNode):
    def __init__(self, op: UnOp, arg: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg = arg
        self.const = None

    @property
    def childs(self) -> Tuple[ExprNode]:
        return self.arg,

    def __str__(self) -> str:
        c = d = sb = ''
        if self.data_type:
            d = 'dtype=' + str(self.data_type)
        if self.const is not None:
            c = 'const=' + str(self.const)
        if d: sb = d
        if c: sb = sb + (', ' if sb else '') + c
        return str(self.op.value) + (' (' + sb + ')' if sb else '')

    def semantic_check(self, context: Context = None):
        self.arg.semantic_check(context)
        self.data_type = self.arg.data_type
        self.data_type = self.arg.data_type
        if type(self.arg) in (LiteralNode, BinOpNode, UnOpNode, CastNode):
            if self.op.value == '!':
                self.const = 1 if not self.arg else 0
            else:
                self.const = eval(str(self.data_type) + '(' + self.op.value + str(self.arg.const) + ')')


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
            if type(el) is AssignNode:
                el: AssignNode = el.var
                el: IdentNode
            if context.find_var(el.name) not in context.vars:
                context.add_var(el)
                el.index = len(context.vars) - 1
                el.data_type = self.vars_type
                el.var_type = VarType.Local if context.parent else VarType.Global
            else:
                logger.error(str(self.row) + ': ' + str(el.name) + ': identifier was already declared')
        for each in self.childs:
            each.semantic_check(context)


class CallNode(StmtNode):
    def __init__(self, func: IdentNode, *params: [ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.func = func
        self.params = list(params)

    @property
    def childs(self) -> Tuple[IdentNode, ...]:
        return self.func, (*self.params)

    def __str__(self) -> str:
        if self.data_type:
            return 'call (dtype=' + str(self.data_type) + ')'
        return 'call'

    def semantic_check(self, context: Context = None):
        for each in self.childs:
            each.semantic_check(context)

        v: FunctionNode = context.find_function(self.func.name)
        if v:
            self.data_type = v.data_type
            if len(self.params) < len(v.params):
                logger.error(str(self.row) + ': ' + str(self.func.name) + ': not enough arguments in the function call')
            elif len(self.params) > len(v.params):
                logger.error(str(self.row) + ': ' + str(self.func.name) + ': too many arguments in the function call')
            else:
                for i, p in enumerate(self.params):
                    if str(v.params[i].data_type) != str(p.data_type):
                        if p.data_type.is_castable_to(v.params[i].data_type):
                            self.params[i] = CastNode(self.params[i], v.params[i].data_type)
                        else:
                            logger.error(str(self.row) + ': ' + str(self.func.name) + ': argument (' + str(i + 1) +
                                         ') of type \'' + str(p.data_type) + '\' is not compatible with type \'' +
                                         str(v.params[i].data_type) + '\'')
        else:
            logger.error(str(self.row) + ': ' + str(self.func.name) + ': unknown identifier')


class FunctionNode(StmtNode):
    def __init__(self, data_type : Type, name: str, params: Tuple = None, *body: Tuple[StmtNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.name = name
        self.params = self.check_params(params) if params else tuple()
        self.body = body if body else (_empty,)

    def check_params(self, params):
        return tuple(IdentNode(param.childs[1], Type(str(param.vars_type), row=self.row)) for param in params)

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
            if e.data_type and e.var_type and e.index is not None:
                if s:
                    s += ', '
                s += '(dtype=' + str(e.data_type) + ', vtype=' + str(e.var_type.value) + ', index=' + str(e.index) + ')'
        if s:
            return str(self.data_type) + ' ' + self.name + '(' + params + ') (' + s + ')'
        else:
            return str(self.data_type) + ' ' + self.name + '(' + params + ')'

    def semantic_check(self, context: Context = None):
        context.add_function(self)
        context = Context(context)
        for el in self.params:
            context.add_var(el)
            el.index = len(context.vars) - 1
            el.var_type = VarType.Param

        context = Context(context)
        for each in self.childs:
            each.semantic_check(context)


class AssignNode(StmtNode):
    def __init__(self, var: IdentNode, val: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.val = val
        self.const = None

    @property
    def childs(self) -> Tuple[IdentNode, ExprNode]:
        return self.var, self.val

    def semantic_check(self, context: Context = None):
        for each in self.childs:
            each.semantic_check(context)

        if self.var.data_type and self.var.data_type.name and self.val.data_type:
            self.data_type = self.var.data_type
            if str(self.var.data_type) != str(self.val.data_type):
                if self.val.data_type.is_castable_to(self.var.data_type):
                    c = None
                    if type(self.val) in (LiteralNode, BinOpNode, UnOpNode, CastNode):
                        c = self.val.const
                    self.val = CastNode(self.val, self.data_type, c)
                else:
                    logger.error(str(self.var.row) + ': ' + str(self.var.name) + ': value does not match \'' + str(self.data_type) + '\' type')
            if type(self.val) in (LiteralNode, BinOpNode, UnOpNode, CastNode):
                self.const = self.val.const

    def __str__(self) -> str:
        c = d = sb = ''
        if self.data_type:
            d = 'dtype=' + str(self.data_type)
        if self.const is not None:
            c = 'const=' + str(self.const)
        if d: sb = d
        if c: sb = sb + (', ' if sb else '') + c
        return '=' + (' (' + sb + ')' if sb else '')


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
            if each:
                each.semantic_check(context)


_empty = StatementListNode()
