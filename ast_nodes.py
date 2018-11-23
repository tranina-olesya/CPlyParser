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
        self.params = []

    def add_var(self, var: 'IdentNode'):
        self.vars.append(var)
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
            loc_context.vars.append(var)

    def add_param(self, var: 'IdentNode'):
        self.params.append(var)

    def add_function(self, func: 'FunctionNode'):
        self.functions.append(func)

    def find_var(self, var_name):
        loc_var = None
        loc_context = self
        while not loc_var and loc_context:
            loc_var = list(filter(lambda x: x.name == var_name, (*loc_context.vars, *loc_context.params)))
            if loc_var:
                loc_var = loc_var[0]
            loc_context = loc_context.parent
        return loc_var

    def get_function_context(self):
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
        return loc_context

    def find_function(self, func_name):
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        func = list(filter(lambda x: x.name == func_name, loc_context.functions))
        return func[0] if func else None

    def get_next_local_num(self):
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
        return len(loc_context.vars)

    def get_next_param_num(self):
        return len(self.params)


class BaseTypes(Enum):
    Int = 'int'
    Float = 'float'
    String = 'str'
    Bool = 'bool'
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

    def is_castable_to(self, cast_type) -> bool:
        if cast_type.rang == self.rang == 0:
            if cast_type.name == self.name:
                return True
            elif cast_type.name in (BaseTypes.String, BaseTypes.Bool):
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
        self.const = None


class LiteralNode(ExprNode):
    def __init__(self, literal: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        if literal in ('true', 'false'):
            literal = literal.capitalize()
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
        if const is not None:
            if self.data_type.name is BaseTypes.Bool:
                self.const = bool(const)
            elif self.data_type.name is BaseTypes.String:
                self.const = str(const)
            else:
                self.const = const

    @property
    def childs(self) -> Tuple[ExprNode, Type]:
        return self.var, self.data_type

    def __str__(self) -> str:
        if self.const is not None:
            return 'cast (const=' + str(self.const) + ')'
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
            '''if self.op.value in ( '&&', '||'):
                self.data_type = Type('bool')
                if self.arg1.data_type.name is not BaseTypes.Bool:
                    self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
                if self.arg2.data_type.name is not BaseTypes.Bool:
                    self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)
            el'''
            if str(self.arg1.data_type) == str(self.arg2.data_type):
                self.data_type = self.arg1.data_type
            elif self.arg1.data_type.is_castable_to(self.arg2.data_type):
                self.data_type = self.arg2.data_type
                self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
            elif self.arg2.data_type.is_castable_to(self.arg1.data_type):
                self.data_type = self.arg1.data_type
                self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)

            if self.arg1.const is not None and self.arg2.const is not None and self.data_type:
                if self.op.value == '&&':
                    self.const = bool(self.arg1.const and self.arg2.const)
                elif self.op.value == '||':
                    self.const = bool(self.arg1.const or self.arg2.const)
                elif self.arg1.data_type.name is BaseTypes.String and self.arg2.data_type.name not in (BaseTypes.Int, BaseTypes.Float, BaseTypes.Bool):
                    if self.op.value == '+':
                        self.const = self.arg1.const + self.arg2.const
                    else:
                        logger.error(str(self.row) + ': pass')
                else:
                    self.const = eval(str(self.data_type) + '(' + str(self.arg1.const) + self.op.value + str(self.arg2.const) + ')')
                if self.op.value in ('>', '<','>=','<=','==','!=', '&&', '||'):
                    self.const = bool(self.const)
                    self.data_type = Type('bool')

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
                el = el.var
            if context.find_var(el.name) in context.get_function_context().vars:
                logger.error(str(self.row) + ': ' + str(el.name) + ': identifier was already declared')
            elif context.find_var(el.name) in context.get_function_context().params:
                logger.error(str(self.row) + ': ' + str(el.name) + ': identifier(param) was already declared')
            else:
                el.index = context.get_next_local_num()
                context.add_var(el)
                el.data_type = self.vars_type
                el.var_type = VarType.Local if context.parent else VarType.Global
        for each in self.childs:
            each.semantic_check(context)
        if self.vars_type is None or self.vars_type.name == BaseTypes.Void:
            logger.error(str(self.row) + ': vars cannot be void')


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
                            self.params[i] = CastNode(self.params[i], v.params[i].data_type, self.params[i].const)
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
            el.index = context.get_next_param_num()
            context.add_param(el)
            el.var_type = VarType.Param
        context = Context(context)
        ret = None
        for each in self.childs:
            if type(each) is ReturnNode:
                ret = each
                each.data_type = self.data_type
            each.semantic_check(context)
        if self.data_type.name is not BaseTypes.Void and not ret:
            logger.error(str(self.row) + ': function must return some value' )


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
                    #c = self.val.const
                    self.val = CastNode(self.val, self.data_type, self.val.const)
                else:
                    logger.error(str(self.var.row) + ': ' + str(self.var.name) + ': value does not match \'' + str(self.data_type) + '\' type')
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
        if self.data_type:
            return '[] (dtype=' + str(self.data_type) + ')'
        return '[]'

    def semantic_check(self, context: Context = None):
        v: IdentNode = context.find_var(self.name.name)
        if v.data_type.rang == 0 and v.data_type.name is not BaseTypes.String:
            logger.error(str(self.row) + ': pass')
            return
        self.data_type = Type(v.data_type.name.value, row=self.row)
        for each in self.childs:
            each.semantic_check(context)
        if self.num.data_type.name is not BaseTypes.Int:
            logger.error(str(self.row) + ': array index must be int')


class ArrayIdentNode(ExprNode):
    def __init__(self, data_type: Type, num: ExprNode = None, *elements: [ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.num = num if num else LiteralNode(str(len(elements)), row=self.row)
        self.elements = list(elements) if elements else list()

    @property
    def childs(self) -> Tuple[AstNode]:
        a = (*self.elements,)
        return (*self.elements,) if self.elements else (_empty,)

    def __str__(self) -> str:
        return self.data_type.name.value + '[' + str(self.num.const) + ']'

    def semantic_check(self, context: 'Context' = None):
        for each in (*self.elements, self.num):
            each.semantic_check(context)
        if self.num.const is None:
            logger.error(str(self.row) + ': length of array must be const')
        if len(self.elements) != 0:
            if len(self.elements) > self.num.const:
                logger.error(str(self.row) + ': too many elements: ' + str(self.num.const) + ' expected, ' + str(len(self.elements)) + ' given')
            elif len(self.elements) < self.num.const:
                logger.error(str(self.row) + ': not enough elements ' + str(self.num.const) + ' expected, ' + str(len(self.elements)) + ' given')
        for i, el in enumerate(self.elements):
            el: ExprNode
            if self.data_type.name.value != str(el.data_type):
                if el.data_type.is_castable_to(Type(self.data_type.name.value)):
                    self.elements[i] = CastNode(el, Type(self.data_type.name.value), el.const)
                else:
                    logger.error(str(self.row) + ': type of element ' + str(i) + ' does not match type of array')


class ReturnNode(StmtNode):
    def __init__(self, val: ExprNode = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.val = val
        self.data_type = None

    @property
    def childs(self) -> Tuple[ExprNode]:
        if self.val:
            return self.val,
        else:
            return tuple()

    def semantic_check(self, context: 'Context' = None):
        for each in self.childs:
            each.semantic_check(context)
        if self.val:
            if self.data_type.name == BaseTypes.Void:
                logger.error(str(self.row) + ': void function must not return any value')
            elif self.val.data_type.is_castable_to(self.data_type):
                if str(self.val.data_type) != str(self.data_type):
                    self.val = CastNode(self.val, self.data_type, self.val.const, row=self.row)
            else:
                logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')
        elif self.data_type.name is not BaseTypes.Void:
            logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')


    def __str__(self) -> str:
        if self.data_type:
            return 'return (dtype=' + str(self.data_type) + ')'
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
