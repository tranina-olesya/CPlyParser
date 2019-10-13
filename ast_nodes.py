from base_nodes import *
from typing import Callable, Tuple, Optional, Union
from logger import *
from context import Context
from constants import *
import copy
from init import *


class LiteralNode(ExprNode):
    def __init__(self, literal: str,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        if literal in ('true', 'false'):
            self.const = eval(literal.capitalize())
        else:
            self.const = eval(literal.replace('\n', '\\n').replace('\r', '\\r'))
        self.literal = literal.replace('\n', '\\n').replace('\r', '\\r')
        if type(self.const) == str and literal[0] == '\"':  # type(self.const).__name__
            self.data_type = Type('string', row=self.row)
            # self.data_type = Type('char', rang=1, row=self.row)
        elif type(self.const) == str and literal[0] == '\'':
            self.const = ord(self.const.encode('cp866'))
            self.data_type = Type('char', row=self.row)
        elif type(self.const) == float:
            self.data_type = Type('double', row=self.row)
        elif type(self.const) == bool:
            self.const = int(self.const)
            self.data_type = Type('bool', row=self.row)
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
            logger.error(str(self.row) + ': ' + self.name + ': unknown identifier')

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        global VAR_NUMBER
        if CUR_LABEL is None:
            return ''
        VAR_NUMBER += 1
        if self.var_type in (VarType.Param, VarType.Local):
            t = '%' + str(self.index)  # + len(func.params))
        else:
            t = '@' + 'g.' + str(self.index)
        code = '  %{0} = load {1}, {1}* {2}, align {3}\n'.format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                                                                 t, LLVM_TYPES[str(self.data_type)][1])
        # self.index = VAR_NUMBER

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
                # self.const = bool(const)
                self.const = 1 if const else 0
            # elif self.data_type.name is BaseTypes.String:
            #   self.const = str(const)
            elif self.data_type.name is BaseTypes.Char:
                self.const = const
            else:
                t = self.data_type
                if t.name is BaseTypes.Double:
                    t = 'float'
                self.const = eval('{0}({1})'.format(t, self.var.const))

    @property
    def children(self) -> Tuple[ExprNode, Type]:
        return self.var, self.data_type

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        global VAR_NUMBER
        if CUR_LABEL is None:
            return ''
        code = self.var.code_generate(func, global_vars, global_arrays)
        n = VAR_NUMBER
        VAR_NUMBER += 1
        if self.data_type.name is BaseTypes.Bool:
            if self.var.data_type.name is BaseTypes.Double:
                t = LLVM_FLOAT_OPS['!=']
                code += '  %{0} = {1} {2} %{3}, {4}\n'.format(VAR_NUMBER, t,
                                                              LLVM_TYPES[self.var.data_type.name.value][0],
                                                              n, float(0))
            elif self.var.data_type.name is BaseTypes.Int:
                t = LLVM_INT_OPS['!=']
                code += '  %{0} = {1} {2} %{3}, {4}\n'.format(VAR_NUMBER, t,
                                                              LLVM_TYPES[self.var.data_type.name.value][0], n, 0)
        elif self.data_type.name is BaseTypes.Char:
            code += '  %{0} = trunc i32 %{1} to i8\n'.format(VAR_NUMBER, VAR_NUMBER - 1)
        elif self.data_type.name is BaseTypes.Double:
            code += '  %{0} = sitofp {1} %{2} to {3}\n'.format(VAR_NUMBER, LLVM_TYPES[self.var.data_type.name.value][0],
                                                               n,
                                                               LLVM_TYPES[self.data_type.name.value][0])
        elif self.data_type.name is BaseTypes.Int:
            code += '  %{0} = fptosi {1} %{2} to {3}\n'.format(VAR_NUMBER, LLVM_TYPES[self.var.data_type.name.value][0],
                                                               n,
                                                               LLVM_TYPES[self.data_type.name.value][0])
        else:
            code += '  %{0} = sext {1} %{2} to {3}\n'.format(VAR_NUMBER, LLVM_TYPES[self.var.data_type.name.value][0], n,
                                                             LLVM_TYPES[self.data_type.name.value][0])
        return code

    def __str__(self) -> str:
        if self.const is not None:
            return 'cast (const={0})'.format(self.const)
        return 'cast'


class BinOpNode(ExprNode):
    def __init__(self, op: BinOp, arg1: ExprNode, arg2: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg1 = arg1
        self.arg2 = arg2
        self.const = None

    def semantic_check(self, context: Context = None):
        for each in self.children:
            each.semantic_check(context)
        if not self.arg1.data_type or not self.arg2.data_type or not self.arg1.data_type.name or not self.arg2.data_type.name:
            return
        # if self.arg1.data_type.name is BaseTypes.String and self.arg2.data_type.name is BaseTypes.String and self.op.value not in ('+', '==', '!='):
        if self.arg1.data_type.rang and self.arg2.data_type.rang:
            logger.error(
                str(self.row) + ': \'{0}\' is not allowed for \'{1}\' and \'{2}\' types'.format(self.op.value, str(
                    self.arg1.data_type), str(self.arg2.data_type)))
            return
        if self.arg1.data_type.name is BaseTypes.Char and self.arg2.data_type.name is BaseTypes.Char:
            if self.op.value not in ('+', '==', '!='):
                logger.error(
                    str(self.row) + ': \'{0}\' is not allowed for \'{1}\' and \'{2}\' types'.format(self.op.value, str(
                        self.arg1.data_type), str(self.arg2.data_type)))
                return
        if self.op.value in ('&&', '||'):
            self.data_type = Type('bool')
            if self.arg1.data_type.name is not BaseTypes.Bool:
                self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
            if self.arg2.data_type.name is not BaseTypes.Bool:
                self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)
        elif self.arg1.data_type == self.arg2.data_type:
            if self.arg1.data_type.name is not BaseTypes.Char:
                self.data_type = self.arg1.data_type
            else:
                self.data_type = Type('int', row=self.row)
                # self.data_type = Type(BaseTypes.Char.value, rang=1, row=self.row)
        elif self.arg1.data_type.is_compatible_with(self.arg2.data_type) and self.arg1.data_type.is_castable_to(
                self.arg2.data_type):
            self.data_type = self.arg2.data_type
            self.arg1 = CastNode(self.arg1, self.data_type, self.arg1.const)
        elif self.arg2.data_type.is_compatible_with(self.arg1.data_type) and self.arg2.data_type.is_castable_to(
                self.arg1.data_type):
            self.data_type = self.arg1.data_type
            self.arg2 = CastNode(self.arg2, self.data_type, self.arg2.const)
        else:
            logger.error(str(self.row) + ': incompatible types: \'{0}\' and \'{1}\''.format(self.arg1.data_type,
                                                                                            self.arg2.data_type))
            return
        if self.op.value in ('>', '<', '>=', '<=', '==', '!='):
            self.data_type = Type('bool')

        if self.op.value in ('/', '%') and self.arg2.const is not None and int(self.arg2.const) == 0:
            logger.error(str(self.row) + ': division by zero')

        if self.arg1.const is not None and self.arg2.const is not None:
            if self.op.value == '&&':
                self.const = bool(self.arg1.const and self.arg2.const)
            elif self.op.value == '||':
                self.const = bool(self.arg1.const or self.arg2.const)
            # elif self.data_type.name is BaseTypes.Char:
            #    # if self.arg1.data_type.name is BaseTypes.Char and self.arg2.data_type.name is BaseTypes.Char:
            #    if self.arg1.data_type.rang == 0 and self.arg2.data_type.rang == 0:
            #        self.const = eval(
            #            '\"' + str(self.arg1.literal[1:-1]) + '\"' + self.op.value + '\"' + str(
            #                self.arg2.literal[1:-1]) + '\"')
            #    else:
            #        self.const = eval(
            #            '\"' + str(self.arg1.const) + '\"' + self.op.value + '\"' + str(self.arg2.const) + '\"')
            else:
                t = str(self.data_type)
                if t == 'double':
                    t = 'float'
                self.const = eval(
                    t + '(' + str(self.arg1.const) + self.op.value + str(self.arg2.const) + ')')
            if self.op.value in ('>', '<', '>=', '<=', '==', '!=', '&&', '||'):
                self.const = bool(self.const)

    @property
    def children(self) -> Tuple[ExprNode, ExprNode]:
        return self.arg1, self.arg2

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0, cur_path=[]) -> str:
        code = ''
        global VAR_NUMBER, CUR_LABEL
        if CUR_LABEL is None:
            return ''
        if self.op in (BinOp.LOGICAL_AND, BinOp.LOGICAL_OR):
            if level == 0:
                cur_path.append([])
            n1 = CUR_LABEL
            t = level + 1
            if self.op is BinOp.LOGICAL_OR and (
                    type(self.arg1) is BinOpNode and self.arg1.op is BinOp.LOGICAL_AND or type(
                self.arg2) is BinOpNode and self.arg2.op is BinOp.LOGICAL_AND):
                t = 0
            if self.arg1.const is None:
                first = self.arg1.code_generate(func, global_vars, global_arrays, t)
            else:
                VAR_NUMBER += 1
                first = '  %{0} = icmp eq i1 {1}, 1\n'.format(VAR_NUMBER, 1 if self.arg1.const else 0)
            if self.op is BinOp.LOGICAL_AND:
                first += '  br i1 %{0}, label %{1}, label %{{0}}\n\n'.format(VAR_NUMBER, VAR_NUMBER + 1)
            else:
                first += '  br i1 %{0}, label %{{0}}, label %{1}\n\n'.format(VAR_NUMBER, VAR_NUMBER + 1)

            if CUR_LABEL not in cur_path[-1]:
                cur_path[-1].append(CUR_LABEL)
            VAR_NUMBER += 1
            n2 = VAR_NUMBER
            CUR_LABEL = n2

            if self.arg2.const is None:
                second = self.arg2.code_generate(func, global_vars, global_arrays, t)
            else:
                VAR_NUMBER += 1
                second = '  %{0} = icmp eq i1 {1}, 1\n'.format(VAR_NUMBER, 1 if self.arg2.const else 0)

            second = '; <label>:{0}:\n'.format(n2) + second
            cur_path[-1].append(CUR_LABEL)
            code = first + second

            if level == 0:
                VAR_NUMBER += 1
                CUR_LABEL = VAR_NUMBER
                n = cur_path[-1][-1]
                code += '  br label %{0}\n\n'.format(VAR_NUMBER)
                code = code.format(CUR_LABEL)
                code += '; <label>:{0}:\n'.format(CUR_LABEL)
                t = ''
                tf = 'true' if self.op.value == '||' else 'false'
                for el in cur_path.pop()[:-1]:
                    t += (', ' if t else '') + '[ {0}, %{1} ]'.format(tf, el)
                t += (', ' if t else '') + '[ %{0}, %{1} ]'.format(VAR_NUMBER - 1, n)
                VAR_NUMBER += 1
                code += '  %{0} = phi i1 {1}\n'.format(VAR_NUMBER, t)
        else:
            if self.arg1.data_type.name in (BaseTypes.Int, BaseTypes.Bool):
                op = LLVM_INT_OPS[self.op.value]
            elif self.arg1.data_type.name is BaseTypes.Double:
                op = LLVM_FLOAT_OPS[self.op.value]
            if self.arg1.const is None and self.arg2.const is None:
                code += self.arg2.code_generate(func, global_vars, global_arrays)
                n2 = VAR_NUMBER
                code += self.arg1.code_generate(func, global_vars, global_arrays)
                n1 = VAR_NUMBER
                VAR_NUMBER += 1
                code += '  %{0} = {1} {2} %{3}, %{4}\n'.format(VAR_NUMBER, op,
                                                               LLVM_TYPES[self.arg1.data_type.name.value][0], n1, n2)
            elif self.arg1.const is not None:
                code += self.arg2.code_generate(func, global_vars, global_arrays)
                n2 = VAR_NUMBER
                VAR_NUMBER += 1
                code += '  %{0} = {1} {2} {3}, %{4}\n'.format(VAR_NUMBER, op,
                                                              LLVM_TYPES[self.arg1.data_type.name.value][0],
                                                              self.arg1.const, n2)
            elif self.arg2.const is not None:
                code += self.arg1.code_generate(func, global_vars, global_arrays)
                n1 = VAR_NUMBER
                VAR_NUMBER += 1
                code += '  %{0} = {1} {2} %{3}, {4}\n'.format(VAR_NUMBER, op,
                                                              LLVM_TYPES[self.arg1.data_type.name.value][0], n1,
                                                              self.arg2.const)
            # else:
            #   VAR_NUMBER += 1
            #  code = '  {3} = {0} {1} {2} \n'.format(self.arg1.const, self.op.value, self.arg2.const, VAR_NUMBER)

        return code

    def __str__(self) -> str:
        sb = ''
        if self.data_type:
            sb = 'dtype=' + str(self.data_type)
        if self.const is not None:
            sb += (', ' if sb else '') + 'const=' + str(self.const)
        return str(self.op.value) + (' (' + sb + ')' if sb else '')


class UnOpNode(ExprNode):
    def __init__(self, op: UnOp, arg: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.op = op
        self.arg = arg
        self.const = None

    @property
    def children(self) -> Tuple[ExprNode]:
        return (self.arg, )

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
            # self.arg = CastNode(self.arg, self.data_type, self.arg.const)
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

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0):
        global VAR_NUMBER
        if CUR_LABEL is None:
            return ''
        code = ''
        if self.const is None:
            code = self.arg.code_generate(func, global_vars, global_arrays)
        if self.op is UnOp.NOT:
            VAR_NUMBER += 1
            t = '0' if self.arg.data_type.name in (BaseTypes.Int, BaseTypes.Bool) else '0.0'
            op = LLVM_INT_OPS['!='] if self.arg.data_type.name in (BaseTypes.Int, BaseTypes.Bool) else LLVM_FLOAT_OPS['!=']
            t2 = 'i1' if type(self.arg) is BinOpNode and self.arg.op.value in ('>', '<', '==', '!=', '<=', '>=') else \
                LLVM_TYPES[self.arg.data_type.name.value][0]
            code += '  %{0} = {1} {2} %{3}, {4}\n'.format(VAR_NUMBER, op, t2, VAR_NUMBER - 1, t)
            VAR_NUMBER += 1
            code += '  %{0} = xor i1 %{1}, true\n'.format(VAR_NUMBER, VAR_NUMBER - 1)
            # VAR_NUMBER += 1
            # code += '  %{0} = zext i1 %{1} to i8\n'.format(VAR_NUMBER, VAR_NUMBER - 1)
        elif self.op is UnOp.SUB and self.const is None:
            VAR_NUMBER += 1
            code += '  %{0} = sub {1} 0, %{2}\n'.format(VAR_NUMBER, LLVM_TYPES[self.data_type.name.value][0],
                                                        VAR_NUMBER - 1)
        return code


class VarsDeclNode(StmtNode):
    def __init__(self, vars_type: Type, *vars_list: Tuple[AstNode, ...],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.vars_type = vars_type
        self.vars_list = vars_list

    @property
    def children(self) -> Tuple[Type, ...]:
        return self.vars_type, (*self.vars_list)

    def __str__(self) -> str:
        return 'var'

    def semantic_check(self, context: Context = None):
        for el in self.vars_list:
            arr = False
            if type(el) is AssignNode:
                el.val.semantic_check(context)
                if not context.parent:
                    if type(el.val) is IdentNode and el.const is None or type(el.val) is ArrayIdentNode and any(
                            elem.const is None for elem in el.val.elements):
                        logger.error(str(el.var.row) + ': ' + str(el.var.name) + ': global variables must be const')
                if type(el.val) is ArrayIdentNode:
                    # context.add_array(el.var, el.val)
                    arr = True if el.val.data_type.name is not BaseTypes.Char else False
                el = el.var

            if context.find_var(el.name) in context.vars:
                logger.error(str(self.row) + ': ' + str(el.name) + ': identifier was already declared')
            elif context.find_var(el.name) in context.get_function_context().params:
                logger.error(str(self.row) + ': ' + str(el.name) + ': identifier(param) was already declared')
            else:
                el.index = context.get_next_local_num()
                el.data_type = self.vars_type
                el.var_type = VarType.Local if context.parent else VarType.Global
                context.add_var(el)
        for each in self.children:
            each.semantic_check(context)
        if self.vars_type is None or self.vars_type.name == BaseTypes.Void:
            logger.error(str(self.row) + ': vars cannot be void')

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        if CUR_LABEL is None:
            return ''
        for each in self.vars_list:
            if type(each) is AssignNode:
                code += each.code_generate(func, global_vars, global_arrays)
        return code


class CallNode(StmtNode):
    def __init__(self, func: IdentNode, *params: [ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.func_name = func
        self.params = list(params)
        self.const = None
        self.func = None

    @property
    def children(self) -> Tuple[IdentNode, ...]:
        return self.func_name, (*self.params)

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

        if self.func_name.name == 'sizeof':
            if len(self.params) == 1:
                self.data_type = Type('int', row=self.row)
                if type(self.params[0]) and self.params[0].data_type.rang:
                    arr = {**context.get_function_context().arrays, **context.get_function_context().parent.arrays}
                    self.const = arr[list(filter(lambda k: k.index == self.params[0].index, arr))[0]].num.const * \
                                 LLVM_TYPES[self.params[0].data_type.name.value][1]
                else:
                    self.const = LLVM_TYPES[self.params[0].data_type.name.value][1]
            else:
                logger.error(str(self.row) + ': \'sizeof\' gets only one argument')
            return

        if all(param.data_type for param in self.params):
            v = context.find_functions(self.func_name.name, self.params)
            if len(v) == 1:
                self.func = v[0]
                self.data_type = self.func.data_type
                if len(self.params) < len(self.func.params):
                    logger.error(
                        str(self.row) + ': ' + str(self.func_name.name) + ': not enough arguments in the function call')
                elif len(self.params) > len(self.func.params):
                    logger.error(
                        str(self.row) + ': ' + str(self.func_name.name) + ': too many arguments in the function call')
                else:
                    for i, p in enumerate(self.params):
                        if str(self.func.params[i].data_type) != str(p.data_type):
                            if p.data_type.is_compatible_with(self.func.params[i].data_type):
                                self.params[i] = CastNode(self.params[i], self.func.params[i].data_type,
                                                          self.params[i].const)
                            else:
                                logger.error(
                                    str(self.row) + ': ' + str(self.func_name.name) + ': argument (' + str(i + 1) +
                                    ') of type \'' + str(p.data_type) + '\' is not compatible with type \'' +
                                    str(self.func.params[i].data_type) + '\'')
            elif len(v) == 0:
                p_str = ''
                for p in self.params:
                    p_str += (', ' if p_str else '') + str(p.data_type)
                logger.error(str(self.row) + ': {0}({1}) was not declared'.format(str(self.func_name.name), p_str))
            else:
                str_params = ''
                for param in self.params:
                    str_params += (', ' if str_params else '') + str(param.data_type)
                logger.error(str(self.row) + ': there is more than one function named \'{0}\' that match ({1})'.format(
                    self.func_name.name, str_params))

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0):
        code = ''
        if CUR_LABEL is None:
            return ''
        params_code = ''
        global VAR_NUMBER
        for arg in self.params[::-1]:
            if arg.const is None:
                code += arg.code_generate(func, global_vars, global_arrays)
            t = '%' + str(VAR_NUMBER) if arg.const is None else arg.const
            params_code = '{0} {1}'.format(LLVM_TYPES[str(arg.data_type)][0], t) + (
                ', ' if params_code else '') + params_code
        if self.data_type.name is not BaseTypes.Void:
            VAR_NUMBER += 1
            code += '  %{0} = call {1} {2}({3})\n'.format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                                                          self.func.LLVMName(), params_code)
        else:
            code += '  call {0} {1}({2})\n'.format(LLVM_TYPES[str(self.data_type)][0], self.func.LLVMName(), params_code)

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
        self.return_num = None
        self.arrays = None

    def check_params(self, params):
        return list(IdentNode(param.children[1], param.vars_type) for param in params if
                    param.vars_type.name)

    @property
    def children(self) -> Tuple[AstNode]:
        return (*self.body,)

    def LLVMName(self) -> str:
        if self.name == 'main':
            return '@main'
        types = {'int': 'H',
                 'void': 'X',
                 'double': 'N',
                 'char[]': 'PEAD',
                 'char': 'D',
                 'bool': 'D'}

        name = '@"?{0}@@YA{1}'.format(self.name, types[str(self.data_type)])
        args_str = ''
        for p in self.params:
            args_str += types[str(p.data_type)]
            if str(p.data_type) == 'char[]':
                types[str(p.data_type)] = '0'
        return name + ((args_str + '@') if args_str else 'X') + 'Z\"'

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
            str_params = ''
            for param in self.params:
                str_params += (', ' if str_params else '') + str(param.data_type)
            logger.error(str(self.row) + ': function \'{0}\'({1}) was already declared'.format(self.name, str_params))
            return
        context.add_function(self)
        context = Context(context)
        for el in self.params:
            el.index = context.get_next_param_num() + len(self.params) + 1
            if not context.is_param_added(el.name):
                context.add_param(el)
                el.var_type = VarType.Param
            else:
                logger.error(str(self.row) + ': param with name \'{0}\' already declared'.format(el.name))
        new_context = Context(context)

        ret = self.find_return(self.children, [])
        for each in self.children:
            each.semantic_check(new_context)

        if self.name == 'main' and len(self.params) > 0:
            if self.params[0].data_type.name is not BaseTypes.Int or self.params[0].data_type.rang:
                logger.error(str(self.row) + ' : first parameter of \'main\' must be of type \'int\'')

        self.args_list = context.vars
        self.arrays = context.arrays
        if self.data_type.name is not BaseTypes.Void:
            if len(ret) == 0:
                logger.error(str(self.row) + ': function must return some value')
            elif len(ret) == 1 and type(self.children[-1]) is ReturnNode:
                ret[0].ok = True
            else:
                self.return_num = context.get_next_local_num()
                for r in ret:
                    r.ok = False
                    r.index = self.return_num

    def find_return(self, childs, count):
        for each in childs:
            if type(each) is ReturnNode:
                count.append(each)
                each.data_type = self.data_type
            count.extend(self.find_return(each.children, []))
        return count

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        for k, v in self.arrays.items():
            if v.data_type.name is BaseTypes.Char:
                sym_str = ''.join(
                    '\\' + hex(x.const)[hex(x.const).find('x') + 1:].capitalize().rjust(2, '0') for x in v.elements)
                code += '@{0}.{1} = dso_local unnamed_addr constant [{2} x i8] c"{3}", align 1\n'.format(self.name,
                                                                                                         v.all_elements_hex(),
                                                                                                         v.num.const,
                                                                                                         sym_str)
        code += '\n'
        sb = body = ''
        for param in self.params:
            sb += (', ' if sb else '') + LLVM_TYPES[str(param.data_type)][0]
        code += 'define dso_local {0} {1}({2}) {{\n'.format(LLVM_TYPES[str(self.data_type)][0], self.LLVMName(), sb)

        global VAR_NUMBER, CUR_LABEL
        VAR_NUMBER = len(self.params)
        CUR_LABEL = VAR_NUMBER
        for elem in (*self.params, *self.args_list):
            VAR_NUMBER += 1
            if elem.data_type.rang == 0:
                body += '  %{0} = alloca {1}, align {2}\n'.format(VAR_NUMBER, *LLVM_TYPES[str(elem.data_type)])
            elif elem.data_type.name is BaseTypes.Char:
                body += '  %{0} = alloca i8*, align 8\n'.format(VAR_NUMBER)
            else:
                num = list(filter(lambda x: x.index == elem.index, self.arrays))
                num = self.arrays[num[0]].num.const if num else 0
                body += '  %{0} = alloca [{1} x {2}], align 16\n'.format(VAR_NUMBER, num,
                                                                         LLVM_TYPES[elem.data_type.name.value][0])
            elem.index = VAR_NUMBER
        if self.return_num is not None:
            VAR_NUMBER += 1
            body += '  %{0} = alloca {1}, align {2}\n'.format(VAR_NUMBER, *LLVM_TYPES[self.data_type.name.value])
        for i in range(len(self.params), 0, -1):
            body += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[str(self.params[i - 1].data_type)][0],
                                                                      i - 1, self.params[i - 1].index,
                                                                      LLVM_TYPES[str(self.params[i - 1].data_type)][1])
        for each in self.children:
            body += each.code_generate(self, global_vars, global_arrays)
        if self.data_type.name is BaseTypes.Void:
            body += '  ret void\n'
        elif self.return_num is not None:
            if CUR_LABEL != VAR_NUMBER:
                VAR_NUMBER += 1
                body += '; <label>:{_return}:\n'
            body = body.format(_return=VAR_NUMBER)
            VAR_NUMBER += 1
            body += '  %{0} = load {1}, {1}* %{2}, align {3}\n'.format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                                                                       self.return_num,
                                                                       LLVM_TYPES[str(self.data_type)][1])
            body += '  ret {0} %{1}\n'.format(LLVM_TYPES[str(self.data_type)][0], VAR_NUMBER)

        body = body.format(_return=VAR_NUMBER)
        code += body + '}\n\n'
        return code


class AssignNode(StmtNode):
    def __init__(self, var: IdentNode, val: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.var = var
        self.val = val
        self.const = None

    @property
    def children(self) -> Tuple[IdentNode, ExprNode]:
        return self.var, self.val

    def semantic_check(self, context: Context = None):
        if type(self.val) is ArrayIdentNode:
            # self.val.semantic_check(context)
            # self.var.data_type = self.val.data_type
            # if self.val.data_type.name is BaseTypes.Char:
            #  self.var.semantic_check(context)
            # else:
            self.var.semantic_check(context)
            if self.val.data_type.name is not BaseTypes.Char:
                if self.var.var_type is VarType.Local:
                    if not list(filter(lambda x: x == self.var, context.vars)):
                        self.var.index = context.get_next_local_num()
                        context.add_var(self.var)
                    context.add_array(self.var, self.val)
                elif self.var.var_type is VarType.Global:
                    p = context.get_function_context().parent if context.parent else context
                    # p.add_var(self.var)
                    if context.parent:
                        if not list(filter(lambda x: x == self.var, p.vars)):
                            self.var.index = p.get_next_local_num()
                            p.add_var(self.var)
                            val = copy.deepcopy(self.val)
                            val.elements = []
                            p.add_array(self.var, val)
                    else:
                        context.add_array(self.var, self.val)
        else:
            for each in self.children:
                each.semantic_check(context)
        if self.var.data_type.rang and self.val.data_type.rang and type(self.val) is IdentNode:
            # con = context.get_function_context()
            # v = list(filter(lambda k: k.index ==self.val.index, con.arrays.keys()))[0]
            self.var.index = self.val.index
            self.var.var_type = self.val.var_type
            context.add_var(self.var)
            # context.add_array(self.var, con.arrays[v])
        if not self.var.data_type or not self.var.data_type or not self.val.data_type or not self.val.data_type.name or not self.var.data_type.name:
            return
        self.data_type = self.var.data_type
        if self.var.data_type != self.val.data_type:
            if self.val.data_type.is_compatible_with(self.var.data_type):
                # c = self.val.const
                self.val = CastNode(self.val, self.data_type, self.val.const)
            else:
                logger.error(str(self.var.row) + ': ' + str(self.var.name) + ': value does not match \'' + str(
                    self.data_type) + '\' type')

        self.const = self.val.const

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        if CUR_LABEL is None:
            return ''
        global VAR_NUMBER
        if self.data_type.rang == 0 or (self.data_type.name is BaseTypes.Char and type(self.val) is not ArrayIdentNode):
            code += self.val.code_generate(func, global_vars, global_arrays) if self.val.const is None or type(
                self.val) is AssignNode else ''
            n1 = VAR_NUMBER
            if type(self.var) is ElementNode:
                code += self.var.get_ptr(func, global_vars, global_arrays)
                if self.val.const is None:
                    code += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.data_type.name.value][0],
                                                                              n1, VAR_NUMBER,
                                                                              LLVM_TYPES[self.data_type.name.value][1])
                else:
                    code += '  store {0} {1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.data_type.name.value][0],
                                                                             self.val.const, VAR_NUMBER,
                                                                             LLVM_TYPES[self.data_type.name.value][1])
            else:
                if self.var.var_type in (VarType.Param, VarType.Local):
                    t = '%' + str(self.var.index)  # + len(func.params))
                else:
                    t = '@' + 'g.' + str(self.var.index)
                # pass (вероятно это должно как-то быть в semantic_check)
                if type(self.val) is BinOpNode and self.val.op in (BinOp.LOGICAL_AND, BinOp.LOGICAL_OR):
                    VAR_NUMBER += 1
                    code += '  %{0} = zext i1 %{1} to i8\n'.format(VAR_NUMBER, VAR_NUMBER - 1)

                if self.const is None:
                    n = VAR_NUMBER
                    code += '  store {0} %{1}, {0}* {2}, align {3}\n'.format(LLVM_TYPES[str(self.data_type)][0], n, t,
                                                                             LLVM_TYPES[str(self.data_type)][1])
                else:
                    code += '  store {0} {1}, {0}* {2}, align {3}\n'.format(LLVM_TYPES[str(self.data_type)][0],
                                                                            self.val.const, t,
                                                                            LLVM_TYPES[str(self.data_type)][1])
        elif type(self.val) is ArrayIdentNode and self.data_type.name is BaseTypes.Char:
            if type(self.val) is ArrayIdentNode:
                code += '  store i8* getelementptr inbounds ([{0} x i8], [{0} x i8]* @{1}.{2}, i32 0, i32 0), i8** %{3}, align 8\n'.format(
                    self.val.num.const, func.name, self.val.all_elements_hex(), self.var.index)
            else:
                if self.var.var_type in (VarType.Param, VarType.Local):
                    t = '%' + str(self.var.index)  # + len(func.params))
                else:
                    t = '@' + 'g.' + str(self.var.index)
                code += self.val.code_generate(func, global_vars, global_arrays)
                code += '  store {0} %{1}, {0}* {2}, align {3}\n'.format(LLVM_TYPES[str(self.data_type)][0], VAR_NUMBER,
                                                                         t, LLVM_TYPES[str(self.data_type)][1])
        elif type(self.val) is ArrayIdentNode:
            num = None
            for el in self.val.elements:
                n1 = VAR_NUMBER
                VAR_NUMBER += 1
                if self.var.var_type == VarType.Global:
                    t = '@' + 'g.' + str(self.var.index)
                else:
                    t = '%' + str(self.var.index)
                if num is None:
                    code += '  %{0} = getelementptr inbounds [{1} x {2}], [{1} x {2}]* {3}, i64 0, i64 0\n'.format(
                        VAR_NUMBER, self.val.num.const, LLVM_TYPES[self.data_type.name.value][0], t)
                else:
                    code += '  %{0} = getelementptr inbounds {1}, {1}* %{2}, i64 1\n'.format(VAR_NUMBER, LLVM_TYPES[
                        self.data_type.name.value][0], num)
                num = VAR_NUMBER
                if el.const is None:
                    code += el.code_generate(func, global_vars, global_arrays)
                    code += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.data_type.name.value][0],
                                                                              VAR_NUMBER, num,
                                                                              LLVM_TYPES[self.data_type.name.value][1])
                else:
                    code += '  store {0} {1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.data_type.name.value][0],
                                                                             el.const, num,
                                                                             LLVM_TYPES[self.data_type.name.value][1])
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
        self.var_type = None

    @property
    def children(self) -> Tuple[IdentNode, ExprNode]:
        return self.name, self.num

    def __str__(self) -> str:
        if self.data_type:
            return '[] (dtype={0})'.format(self.data_type)
        return '[]'

    def semantic_check(self, context: Context = None):
        v: IdentNode = context.find_var(self.name.name)
        if not v:
            logger.error(str(self.row) + ': ' + str(self.name.name) + ': unknown identifier')
            return
        if v.data_type.rang == 0:  # and v.data_type.name is not BaseTypes.String:
            logger.error(str(self.row) + ': index can be used only with \'string\' type or array')
            return
        # if v.data_type.name is BaseTypes.String:
        #   self.data_type = Type('char', row=self.row)
        else:
            self.data_type = Type(v.data_type.name.value, row=self.row)
        self.var_type = v.var_type
        for each in self.children:
            each.semantic_check(context)
        if self.num.data_type.name is not BaseTypes.Int:
            logger.error(str(self.row) + ': array index must be int')

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0):
        global VAR_NUMBER
        code = self.get_ptr(func, global_vars, global_arrays)
        VAR_NUMBER += 1
        code += '  %{0} = load {1}, {1}* %{2}, align {3}\n'.format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                                                                   VAR_NUMBER - 1,
                                                                   LLVM_TYPES[str(self.data_type)][1])
        return code

    def get_ptr(self, func=None, global_vars=None, global_arrays=None, level=0):
        global VAR_NUMBER
        code = self.num.code_generate(func, global_vars, global_arrays)
        if code:
            VAR_NUMBER += 1
            code += '  %{0} = sext i32 %{1} to i64\n'.format(VAR_NUMBER, VAR_NUMBER - 1)

        VAR_NUMBER += 1
        if self.var_type == VarType.Global:
            t = '@' + 'g.' + str(self.name.index)
        else:
            t = '%' + str(self.name.index)
        if self.data_type.name is BaseTypes.Char:
            code += '  %{0} = load i8*, i8** {1}, align 8\n'.format(VAR_NUMBER, t)
            VAR_NUMBER += 1
            if self.num.const is not None:
                code += '  %{0} = getelementptr inbounds i8, i8* %{1}, i64 {2}\n'.format(
                    VAR_NUMBER, VAR_NUMBER - 1, self.num.const)
            else:
                code += '  %{0} = getelementptr inbounds i8, i8* %{1}, i64 %{2}\n'.format(
                    VAR_NUMBER, VAR_NUMBER - 1, VAR_NUMBER - 2)
        else:
            arr = {**func.arrays, **global_arrays}
            length = arr[list(filter(lambda x: x.index == self.name.index and x.var_type == self.name.var_type, arr))[
                0]].num.const
            if self.num.const is not None:
                code += '  %{0} = getelementptr inbounds [{1} x {2}], [{1} x {2}]* {3}, i64 0, i64 {4}\n'.format(
                    VAR_NUMBER, length, LLVM_TYPES[str(self.data_type)][0], t, self.num.const)
            else:
                code += '  %{0} = getelementptr inbounds [{1} x {2}], [{1} x {2}]* {3}, i64 0, i64 %{4}\n'.format(
                    VAR_NUMBER, length, LLVM_TYPES[str(self.data_type)][0], t, VAR_NUMBER - 1)
        return code


class ArrayIdentNode(ExprNode):
    def __init__(self, data_type: Type, num: ExprNode = None, *elements: [ExprNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        if self.data_type.name is BaseTypes.Char:
            self.elements = list(LiteralNode('\'' + ch + '\'') for ch in elements[1:-1])
            self.elements.append(LiteralNode('\'\\0\''))
        else:
            self.elements = list(elements) if elements else list()
        self.num = num if num else LiteralNode(str(len(self.elements)), row=self.row)

    @property
    def children(self) -> Tuple[AstNode]:
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
                if el.data_type.is_compatible_with(Type(self.data_type.name.value)):
                    self.elements[i] = CastNode(el, Type(self.data_type.name.value), el.const)
                else:
                    logger.error(str(self.row) + ': type of element ' + str(i) + ' does not match type of array')
        if self.data_type.name is BaseTypes.Char:
            context.add_array(IdentNode('none'), self)

    def all_elements_hex(self):
        return ''.join(hex(x.const)[hex(x.const).find('x') + 1:].capitalize().rjust(2, '0') for x in self.elements)

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0):
        global VAR_NUMBER
        VAR_NUMBER += 1
        code = '  %{0} = getelementptr inbounds [{1} x {2}], [{1} x {2}]* @{3}.{4}, i64 0, i64 0\n'.format(VAR_NUMBER,
                                                                                                           self.num.const,
                                                                                                           LLVM_TYPES[
                                                                                                               self.data_type.name.value][
                                                                                                               0],
                                                                                                           func.name,
                                                                                                           self.all_elements_hex())
        return code


class ReturnNode(StmtNode):
    def __init__(self, val: ExprNode = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.val = val
        self.data_type = None
        self.ok = None
        self.index = None

    @property
    def children(self) -> Tuple[ExprNode]:
        if self.val:
            return self.val,
        else:
            return tuple()

    def semantic_check(self, context: 'Context' = None):
        if self.val:
            self.val.semantic_check(context)
            self.const = self.val.const
            if self.data_type.name == BaseTypes.Void:
                logger.error(str(self.row) + ': void function must not return any value')
            elif self.val.data_type.is_compatible_with(self.data_type):
                if str(self.val.data_type) != str(self.data_type):
                    self.val = CastNode(self.val, self.data_type, self.val.const, row=self.row)
                    if self.const is not None:
                        t = self.data_type.name.value if self.data_type.name is not BaseTypes.Double else 'float'
                        self.const = eval('{0}({1})'.format(t, self.const))
            else:
                logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')
        elif self.data_type.name is not BaseTypes.Void:
            logger.error(str(self.row) + ': function must return \'' + str(self.data_type) + '\' value')

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        global CUR_LABEL
        if CUR_LABEL is None:
            return ''
        code = ''
        if self.ok:
            if self.const is not None:
                code = '  ret {0} {1}\n'.format(LLVM_TYPES[self.data_type.name.value][0], str(self.const))
            else:
                code += self.val.code_generate(func, global_vars, global_arrays)
                code += '  ret {0} %{1}\n'.format(LLVM_TYPES[self.val.data_type.name.value][0], VAR_NUMBER)
        else:
            if self.const is not None:
                code = '  store {0} {1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.data_type.name.value][0],
                                                                        str(self.const), self.index,
                                                                        LLVM_TYPES[self.data_type.name.value][1])
            elif self.val:
                code += self.val.code_generate(func, global_vars, global_arrays)
                code += '  store {0} %{1}, {0}* %{2}, align {3}\n'.format(LLVM_TYPES[self.val.data_type.name.value][0],
                                                                          VAR_NUMBER, self.index,
                                                                          LLVM_TYPES[self.data_type.name.value][1])
            code += '  br label %{_return}\n\n'
            CUR_LABEL = None
        return code

    def __str__(self) -> str:
        if self.data_type:
            return 'return (dtype={0})'.format(self.data_type)
        return 'return'


class IfNode(StmtNode):
    def __init__(self, cond: ExprNode, then_stmt: StmtNode, else_stmt: Optional[StmtNode] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.then_stmt = then_stmt if then_stmt else _empty
        self.else_stmt = else_stmt

    @property
    def children(self) -> Tuple[Union[ExprNode, StmtNode, 'StatementListNode'], ...]:
        return (self.cond, self.then_stmt) + ((self.else_stmt,) if self.else_stmt else tuple())

    def __str__(self) -> str:
        return 'if'

    def semantic_check(self, context: 'Context' = None):
        self.cond.semantic_check(context)
        if not self.cond.data_type:
            return
        if self.cond.data_type.name is not BaseTypes.Bool:
            self.cond = CastNode(self.cond, Type('bool', row=self.row), self.cond.const, row=self.row)
        context = Context(context)
        self.then_stmt.semantic_check(context)
        if self.else_stmt:
            context = Context(context)
            self.else_stmt.semantic_check(context)

    def code_generate(self, func=None, global_vars=None, global_arrays=None, level=0) -> str:
        global VAR_NUMBER, CUR_LABEL
        if CUR_LABEL is None:
            return ''
        cond = ''
        if self.cond.const is None:
            cond = self.cond.code_generate(func, global_vars, global_arrays)
            if type(self.cond) is IdentNode:
                VAR_NUMBER += 1
                cond += '  %{0} = trunc i8 %{1} to i1\n'.format(VAR_NUMBER, VAR_NUMBER - 1)
            cond += '  br i1 %{_cond_value}, label %{_label_then}, label %{_label_else}\n\n'
        else:
            if self.cond.const:
                cond += '  br label %{_label_then}\n\n'
            elif self.else_stmt:
                cond += '  br label %{_label_else}\n\n'
            else:
                cond += '  br label %{_label_end}\n\n'
        # start = CUR_LABEL
        cond_value = VAR_NUMBER
        VAR_NUMBER += 1
        label_then = VAR_NUMBER
        CUR_LABEL = VAR_NUMBER
        then_stmt = '; <label>:{_label_then}:\n' + self.then_stmt.code_generate(func, global_vars, global_arrays)
        VAR_NUMBER += 1
        n1 = n2 = CUR_LABEL
        if n1:
            then_stmt += '  br label %{_label_end}\n\n'
        label_else = VAR_NUMBER
        else_stmt = ''
        if self.else_stmt:
            # el = VAR_NUMBER
            CUR_LABEL = VAR_NUMBER
            else_stmt = '; <label>:{_label_else}:\n' + self.else_stmt.code_generate(func, global_vars, global_arrays)
            n2 = CUR_LABEL
            if CUR_LABEL:
                else_stmt += '  br label %{_label_end}\n\n'
            VAR_NUMBER += 1
        label_end = VAR_NUMBER
        CUR_LABEL = label_end

        code = cond + then_stmt + else_stmt + '; <label>:{_label_end}:\n'
        code = code.format(_cond_value=cond_value, _label_then=label_then, _label_else=label_else, _label_end=label_end,
                           _break='{_break}', _continue='{_continue}', _return='{_return}')
        CUR_LABEL = VAR_NUMBER
        return code


class StatementListNode(StmtNode):
    def __init__(self, *exprs: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.exprs = exprs

    def add_child(self, *ch):
        self.exprs += ch

    @property
    def children(self) -> Tuple[StmtNode, ...]:
        return self.exprs

    def semantic_check(self, context: Context = None):
        # context = Context(context)
        for each in self.children:
            each.semantic_check(context)

    def __str__(self) -> str:
        return '...'


class BlockNode(StatementListNode):
    def __init__(self, *exprs: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(exprs=exprs, row=row, line=line, **props)
        self.exprs = exprs

    def semantic_check(self, context: Context = None):
        context = Context(context)
        for each in self.children:
            each.semantic_check(context)


_empty = StatementListNode()
