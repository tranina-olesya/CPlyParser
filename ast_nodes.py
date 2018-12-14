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

    def code_generate(self, func=None) -> str:
        code = ''
        for each in self.childs:
            code += each.code_generate(func)
        return code

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
        self.functions = [FunctionNode(Type('void'), 'output_int', (VarsDeclNode(Type('int'), (IdentNode('a'), )),)),
                          FunctionNode(Type('void'), 'output_double', (VarsDeclNode(Type('double'), (IdentNode('a'), )),)),
                          FunctionNode(Type('int'), 'input_int'),
                          FunctionNode(Type('double'), 'input_double')]
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

    def find_var(self, var_name) -> 'IdentNode':
        loc_var = None
        loc_context = self
        while not loc_var and loc_context:
            loc_var = list(filter(lambda x: x.name == var_name, (*loc_context.vars, *loc_context.params)))
            if loc_var:
                loc_var = loc_var[0]
            loc_context = loc_context.parent
        return loc_var

    def get_function_context(self) -> 'Context':
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
        return loc_context

    def is_param_added(self, param_name):
        return list(filter(lambda x: x.name == param_name, self.params))

    def find_function(self, func_name, func_params) -> 'FunctionNode':
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        func = None
        b = False
        for function in loc_context.functions:
            if function.name == func_name and len(function.params) == len(func_params):
                if all(func_params[i].data_type == function.params[i].data_type for i in range(len(function.params))):
                    return function
                if all(func_params[i].data_type.is_castable_to(function.params[i].data_type) for i in range(len(function.params))):
                    if func is None:
                        func = function
                    else:
                        b = True
        if b:
            raise Exception
        return func

    def check_if_was_declared(self, func_name, func_params) -> bool:
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        for function in loc_context.functions:
            if function.name == func_name and len(function.params) == len(func_params):
                if all(func_params[i].data_type == function.params[i].data_type for i in range(len(function.params))):
                    return True
        return False

    def find_function_by_name(self, func_name) -> 'FunctionNode':
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
    Double = 'double'
    String = 'string'
    Bool = 'bool'
    Void = 'void'


LLVMTypes = {
    'int': ['i32', 4],
    'double': ['double', 8],
    'bool': ['i8', 1],
    'void': ['void', ]
}

LLVMIntOps = {
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

LLVMFloatOps = {
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
            elif cast_type.name == BaseTypes.Bool:
                return True
            elif cast_type.name == BaseTypes.Double:
                return self.name == BaseTypes.Int
            elif cast_type.name == BaseTypes.Int:
                return False
        return False

    def __eq__(self, other):
        return self.name == other.name and self.rang == other.rang

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
        if type(self.const) == str:  # type(self.const).__name__
            self.data_type = Type('string', row=self.row)
        elif type(self.const) == float:
            self.data_type = Type('double', row=self.row)
        else:
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
            '''v = context.find_function_by_name(self.name)
            if v:
                self.data_type = v.data_type
            else:'''
            logger.error(str(self.row) + ': ' + self.name + ': unknown identifier')

    def code_generate(self, func=None) -> str:
        global var_number
        var_number += 1
        code = '  %{0} = load {1}, {1}* %{2}, align {3}\n'.format(var_number, LLVMTypes[self.data_type.name.value][0],
                                                                func.find_var(self.name).index, LLVMTypes[self.data_type.name.value][1])
        #self.index = var_number

        return code

    def __str__(self) -> str:
        if self.index is not None and self.data_type and self.var_type:
            return '{0} (dtype={1}, vtype={2}, index={3})'.format(self.name, self.data_type, self.var_type.value,
                                                                  self.index)
        else:
            return str(self.name)


class CastNode(ExprNode):
    def __init__(self, var: ExprNode, data_type: Type, const=None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.var = var
        if const is not None:
            if self.data_type.name is BaseTypes.Bool:
                self.const = bool(const)
            elif self.data_type.name is BaseTypes.String:
                self.const = str(const)
            else:
                t = self.data_type
                if t.name is BaseTypes.Double:
                    t = 'float'
                self.const = eval('{0}({1})'.format(t, self.var.const))

    @property
    def childs(self) -> Tuple[ExprNode, Type]:
        return self.var, self.data_type

    def code_generate(self, func=None) -> str:
        global var_number
        code = self.var.code_generate(func)
        n = var_number
        var_number += 1
        #code += '{0} = cast {1} to {2}\n'.format(var_number, str(self.var.data_type), str(self.data_type.name.value))
        code += '  %{0} = sitofp {1} %{2} to {3}\n'.format(var_number, LLVMTypes[self.var.data_type.name.value][0], n, LLVMTypes[self.data_type.name.value][0])
        return code

    def __str__(self) -> str:
        if self.const is not None:
            return 'cast (const={0})'.format(self.const)
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
        if not self.arg1.data_type or not self.arg2.data_type or not self.arg1.data_type.name or not self.arg2.data_type.name:
            return
        if self.arg1.data_type.name is BaseTypes.String and self.arg2.data_type.name is BaseTypes.String:
            if self.op.value not in ('+', '==', '!='):
                logger.error(
                    str(self.row) + ': \'{0}\' is not allowed for \'{1}\' and \'{2}\' types'.format(self.op.value, str(
                        self.arg1.data_type), str(self.arg2.data_type)))
                return
        elif self.op.value in (""" '>', '<', '>=', '<=', '==', '!=', """ '&&', '||'):
            self.data_type = Type('bool')
            if self.op.value in ('&&', '||'):
                if self.arg1.data_type.name is not BaseTypes.Bool:
                    self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
                if self.arg2.data_type.name is not BaseTypes.Bool:
                    self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)
        elif self.arg1.data_type == self.arg2.data_type:
            self.data_type = self.arg1.data_type
        elif self.arg1.data_type.is_castable_to(self.arg2.data_type):
            self.data_type = self.arg2.data_type
            self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
        elif self.arg2.data_type.is_castable_to(self.arg1.data_type):
            self.data_type = self.arg1.data_type
            self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)
        else:
            logger.error(str(self.row) + ': incompatible types: \'{0}\' and \'{1}\''.format(self.arg1.data_type,
                                                                                            self.arg2.data_type))
            return

        if self.arg1.const is not None and self.arg2.const is not None:
            if self.op.value == '&&':
                self.const = bool(self.arg1.const and self.arg2.const)
            elif self.op.value == '||':
                self.const = bool(self.arg1.const or self.arg2.const)
            elif self.arg1.data_type.name is BaseTypes.String and self.arg2.data_type.name is BaseTypes.String:
                self.const = eval(
                    '\"' + str(self.arg1.const) + '\"' + self.op.value + '\"' + str(self.arg2.const) + '\"')
            else:
                if self.op.value in ('/', '%') and int(self.arg2.const) == 0:
                    logger.error(str(self.row) + ': division by zero')
                else:
                    t = str(self.data_type)
                    if t == 'double':
                        t = 'float'
                    self.const = eval(
                        t + '(' + str(self.arg1.const) + self.op.value + str(self.arg2.const) + ')')
            if self.op.value in ('>', '<', '>=', '<=', '==', '!=', '&&', '||'):
                self.const = bool(self.const)

    @property
    def childs(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def code_generate(self, func=None) -> str:
        code = ''
        global var_number
        if self.data_type.name in (BaseTypes.Int, BaseTypes.Bool):
            op = LLVMIntOps[self.op.value]
        elif self.data_type.name is BaseTypes.Double:
            op = LLVMFloatOps[self.op.value]
        if self.arg1.const is None and self.arg2.const is None:
            code += self.arg1.code_generate(func)
            n1 = var_number
            code += self.arg2.code_generate(func)
            n2 = var_number
            var_number += 1
            code += '  %{0} = {1} {2} %{3}, %{4}\n'.format(var_number, op, LLVMTypes[self.data_type.name.value][0], n1, n2)
        elif self.arg1.const is not None:
            code += self.arg2.code_generate(func)
            n2 = var_number
            var_number += 1
            code += '  %{0} = {1} {2} {3}, %{4}\n'.format(var_number, op, LLVMTypes[self.data_type.name.value][0], self.arg1.const, n2)
        elif self.arg2.const is not None:
            code += self.arg1.code_generate(func)
            n1 = var_number
            var_number += 1
            #code += '{3} = prev{0} {1} {2}\n'.format(n1, self.op.value, self.arg2.const, var_number)
            code += '  %{0} = {1} {2} %{3}, {4}\n'.format(var_number, op, LLVMTypes[self.data_type.name.value][0], n1, self.arg2.const)
        else:
            var_number += 1
            code = '  {3} = {0} {1} {2} \n'.format(self.arg1.const, self.op.value, self.arg2.const, var_number)
        return code

    def __str__(self) -> str:
        sb = ''
        if self.data_type:
            sb = 'dtype=' + str(self.data_type)
        if self.const is not None:
            sb += (', ' if sb else '') + 'const=' + str(self.const)
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
        sb = ''
        if self.data_type:
            sb = 'dtype=' + str(self.data_type)
        if self.const is not None:
            sb += (', ' if sb else '') + 'const=' + str(self.const)
        return str(self.op.value) + (' (' + sb + ')' if sb else '')

    def semantic_check(self, context: Context = None):
        self.arg.semantic_check(context)

        if self.op.value == '!':
            self.data_type = Type('bool', row=self.row)
            self.arg = CastNode(self.arg, self.data_type, self.arg.const)
        else:
            self.data_type = self.arg.data_type
        self.const = self.arg.const

        if self.arg.data_type.rang != 0:
            logger.error(
                str(self.row) + ': \'{0}\' is not allowed for \'{1}\' type'.format(self.op.value, self.arg.data_type))
        if self.const is not None:
            if self.op.value == '!':
                self.const = not self.const
            elif self.data_type.name in (BaseTypes.Double, BaseTypes.Int):
                t = str(self.data_type)
                if t == 'double':
                    t = 'float'
                self.const = eval(t + '(' + self.op.value + str(self.arg.const) + ')')
            else:
                logger.error(str(self.row) + ': \'{0}\' is not allowed for \'{1}\' type'.format(self.op.value,
                                                                                                self.arg.data_type))


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
            if context.find_var(el.name) in context.vars:
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
            sb = ''
            for param in self.params:
                sb += (', ' if sb else '') + str(param.data_type)
            return 'call (dtype=(({0})->{1}))'.format(sb, self.data_type)
        return 'call'

    def semantic_check(self, context: Context = None):
        for each in self.params:
            each.semantic_check(context)

        try:
            v: FunctionNode = context.find_function(self.func.name, self.params)
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
        except Exception:
            str_params = ''
            for param in self.params:
                str_params += (', ' if str_params else '') + str(param.data_type)
            logger.error(str(self.row) + ': there is more than one function named \'{0}\' that match ({1})'.format(self.func.name, str_params))

    def code_generate(self, func=None):
        code = ''
        params_code = ''
        global var_number
        for arg in self.params[::-1]:
            code += arg.code_generate(func)
            t = '%' + str(var_number) if arg.const is None else arg.const
            params_code = '{0} {1}'.format(LLVMTypes[arg.data_type.name.value][0], t ) + (', ' if params_code else '') + params_code
        if self.data_type.name is not BaseTypes.Void:
            var_number += 1
            code += '  %{0} = call {1} @{2}({3})\n'.format(var_number, LLVMTypes[self.data_type.name.value][0], self.func.name, params_code)
        else:
            code += '  call {0} @{1}({2})\n'.format(LLVMTypes[self.data_type.name.value][0], self.func.name, params_code)

        return code


class FunctionNode(StmtNode):
    def __init__(self, data_type: Type, name: str, params: Tuple = None, *body: Tuple[StmtNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.name = name
        self.params = self.check_params(params) if params else list()
        self.body = body if body else (_empty,)
        self.args_list = None

    def check_params(self, params):
        return list(IdentNode(param.childs[1], Type(str(param.vars_type), row=self.row)) for param in params if
                     param.vars_type.name)

    @property
    def childs(self) -> Tuple[AstNode]:
        return (*self.body,)

    def find_var(self, var_name):
        if self.args_list is None:
            return None
        var = list(filter(lambda x: x.name == var_name, (*self.args_list, *self.params)))
        return var[0] if var else None

    def __str__(self) -> str:
        params = ''
        if len(self.params):
            params += '{0} {1}'.format(self.params[0].data_type, self.params[0].name)
            for item in self.params[1:]:
                params += ', {0} {1}'.format(item.data_type, item.name)
        s = ''
        for e in self.params:
            if e.data_type and e.var_type and e.index is not None:
                if s:
                    s += ', '
                s += '(dtype={0}, vtype={1}, index={2})'.format(e.data_type, e.var_type.value, e.index)
        if s:
            return '{0} {1} ({2}) ({3})'.format(self.data_type, self.name, params, s)
        else:
            return '{0} {1} ({2})'.format(self.data_type, self.name, params)

    def semantic_check(self, context: Context = None):
        if context.check_if_was_declared(self.name, self.params):
            str_params =''
            for param in self.params:
                str_params += (', ' if str_params else '') + str(param.data_type)
            logger.error(str(self.row) + ': function \'{0}\'({1}) was already declared'.format(self.name, str_params))
            return
        context.add_function(self)
        context = Context(context)
        for el in self.params:
            el.index = context.get_next_param_num()
            if not context.is_param_added(el.name):
                context.add_param(el)
                el.var_type = VarType.Param
            else:
                logger.error(str(self.row) + ': param with name \'{0}\' already declared'.format(el.name))
        new_context = Context(context)
        ret = self.find_return(self.childs)
        if self.data_type.name is not BaseTypes.Void and ret == 0:
            logger.error(str(self.row) + ': function must return some value')

        for each in self.childs:
            each.semantic_check(new_context)

        self.args_list = context.vars

    def find_return(self, childs, count=0):
        for each in childs:
            if type(each) is ReturnNode:
                count += 1
                each.data_type = self.data_type
            count = self.find_return(each.childs, count)
        return count

    def code_generate(self, func=None) -> str:
        sb = ''
        for param in self.params:
            sb += (', ' if sb else '') + LLVMTypes[param.data_type.name.value][0]
        code = 'define dso_local {0} @{1}({2}) {{\n'.format(LLVMTypes[self.data_type.name.value][0] ,self.name, sb)

        global var_number
        var_number = len(self.params)
        for elem in (*self.params[::-1], *self.args_list):
            var_number += 1
            code += '  %{0} = alloca {1}, align {2}\n'.format(var_number, *LLVMTypes[elem.data_type.name.value])
            elem.index = var_number

        for i in range(len(self.params), 0 , -1):
            code += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVMTypes[self.params[i-1].data_type.name.value][0], i - 1 , self.params[i-1].index, LLVMTypes[self.params[i-1].data_type.name.value][1])
        for each in self.childs:
            code += each.code_generate(self)
        if self.data_type.name is BaseTypes.Void:
            code += '  ret void\n'
        code += '}\n\n'
        return code


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

        if not self.var.data_type or not self.var.data_type or not self.val.data_type or not self.val.data_type.name or not self.var.data_type.name:
            return
        self.data_type = self.var.data_type
        if self.var.data_type != self.val.data_type:
            if self.val.data_type.is_castable_to(self.var.data_type):
                # c = self.val.const
                self.val = CastNode(self.val, self.data_type, self.val.const)
            else:
                logger.error(str(self.var.row) + ': ' + str(self.var.name) + ': value does not match \'' + str(
                    self.data_type) + '\' type')
        self.const = self.val.const

    def code_generate(self, func=None) -> str:
        code = ''
        global var_number
        if self.const is None:
            code += self.val.code_generate(func)
            n = var_number
            #var_number += 1
            code += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVMTypes[self.data_type.name.value][0], n, func.find_var(self.var.name).index, LLVMTypes[self.data_type.name.value][1])
        else:
            code += self.val.code_generate(func)
            code += '  store {0} {1}, {0}* %{2}, align {3}\n'.format(LLVMTypes[self.data_type.name.value][0], self.val.const, func.find_var(self.var.name).index, LLVMTypes[self.data_type.name.value][1])
        return code

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
            return '[] (dtype={0})'.format(self.data_type)
        return '[]'

    def semantic_check(self, context: Context = None):
        v: IdentNode = context.find_var(self.name.name)
        if v.data_type.rang == 0 and v.data_type.name is not BaseTypes.String:
            logger.error(str(self.row) + ': index can be used only with \'string\' type or array')
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
        return (*self.elements,) if self.elements else (_empty,)

    def __str__(self) -> str:
        return '{0} [{1}]'.format(self.data_type.name.value, self.num.const)

    def semantic_check(self, context: 'Context' = None):
        for each in (*self.elements, self.num):
            each.semantic_check(context)
        if self.num.const is None:
            logger.error(str(self.row) + ': length of array must be const')
        if len(self.elements) != 0:
            if len(self.elements) > self.num.const:
                logger.error(str(self.row) + ': too many elements: ' + str(self.num.const) + ' expected, ' + str(
                    len(self.elements)) + ' given')
            elif len(self.elements) < self.num.const:
                logger.error(str(self.row) + ': not enough elements ' + str(self.num.const) + ' expected, ' + str(
                    len(self.elements)) + ' given')
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
            self.const = self.val.const
            if self.data_type.name == BaseTypes.Void:
                logger.error(str(self.row) + ': void function must not return any value')
            elif self.val.data_type.is_castable_to(self.data_type):
                if str(self.val.data_type) != str(self.data_type):
                    self.val = CastNode(self.val, self.data_type, self.val.const, row=self.row)
            else:
                logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')
        elif self.data_type.name is not BaseTypes.Void:
            logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')

    def code_generate(self, func=None) -> str:
        code = ''
        if self.const is not None:
            code = '  ret {0} {1}\n'.format(LLVMTypes[self.data_type.name.value][0], self.const)
        else:
            code += self.val.code_generate(func)
            code += '  ret {0} %{1}\n'.format(LLVMTypes[self.val.data_type.name.value][0], var_number)
        return code

    def __str__(self) -> str:
        if self.data_type:
            return 'return (dtype={0})'.format(self.data_type)
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
    def childs(self) -> Tuple[Union[ExprNode, StmtNode, 'StatementListNode'], ...]:
        return (self.cond, self.then_stmt) + ((self.else_stmt,) if self.else_stmt else tuple())

    def __str__(self) -> str:
        return 'if'

    def semantic_check(self, context: 'Context' = None):
        self.cond.semantic_check(context)
        if not self.cond.data_type:
            return
        #if self.cond.data_type.name is not BaseTypes.Bool:
        #    self.cond = CastNode(self.cond, Type('bool', row=self.row), self.cond.const, row=self.row)
        context = Context(context)
        self.then_stmt.semantic_check(context)
        if self.else_stmt:
            context = Context(context)
            self.else_stmt.semantic_check(context)

    def code_generate(self, func=None) -> str:
        global var_number
        cond = self.cond.code_generate(func)
        l1 = var_number
        var_number += 1
        l2 = var_number
        then_stmt = '; <label>:{0}:\n'.format(l2) + self.then_stmt.code_generate(func)
        var_number += 1
        l3 = var_number
        else_stmt = ''
        if self.else_stmt:
            else_stmt = '; <label>:{0}:\n'.format(l3) + self.else_stmt.code_generate(func)
            var_number += 1
        l4 = var_number
        br_l = '  br label %{0}\n\n'.format(l4)
        code = cond + '  br i1 %{0}, label %{1}, label %{2}\n\n'.format(l1, l2, l3) + then_stmt + br_l + else_stmt + (br_l if self.else_stmt else '') + '; <label>:{0}:\n'.format(l4)
        return code


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

    def code_generate(self, func=None) -> str:
        global var_number

        init = self.init.code_generate(func)
        var_number += 1
        l1 = var_number
        init += '  br label %{0}\n\n'.format(l1)

        cond = '; <label>:{0}:\n'.format(l1) + self.cond.code_generate(func)
        var_number += 1
        l2 = var_number

        body = '; <label>:{0}:\n'.format(l2) + self.body.code_generate(func)
        var_number += 1
        l3 = var_number
        body += '  br label %{0}\n\n'.format(l3)

        step = '; <label>:{0}:\n'.format(l3) + self.step.code_generate(func)
        var_number += 1
        l4 = var_number
        step += '  br label %{0}\n\n'.format(l1)

        #br_l = '  br label %{0}\n\n'.format(l4)
        cond += '  br i1 %{0}, label %{1}, label %{2}\n\n'.format(l2-1, l2, l4)
        code = init + cond + body + step + '; <label>:{0}:\n'.format(l4)
        return code


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
        context = Context(context)
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
var_number = 0
