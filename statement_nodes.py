"""Statement nodes"""

import copy
from typing import Optional, Tuple

from ast_nodes import ArrayIdentNode, ExprNode, IdentNode, CastNode
from ast_nodes import ReturnNode, StatementListNode, BinOpNode, ElementNode
from base_nodes import StmtNode, Type, AstNode
from constants import LLVM_TYPES, VarType, BaseTypes, BinOp
from context import Context
from logger import logger


class VarsDeclNode(StmtNode):
    """Variable declaration node"""

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
            # arr = False
            if isinstance(el, AssignNode):
                el.val.semantic_check(context)
                if not context.parent:
                    if isinstance(el.val, IdentNode) and el.const is None \
                            or isinstance(el.val, ArrayIdentNode) \
                            and any(elem.const is None for elem in el.val.elements):
                        logger.error(str(el.var.row) + ': ' + str(el.var.name)
                                     + ': global variables must be const')
                # if isinstance(el.val, ArrayIdentNode):
                    # context.add_array(el.var, el.val)
                    # arr = True if el.val.data_type.name is not BaseTypes.Char else False
                el = el.var

            if context.find_var(el.name) in context.vars:
                logger.error(str(self.row) + ': ' + str(el.name)
                             + ': identifier was already declared')
            elif context.find_var(el.name) in context.get_function_context().params:
                logger.error(str(self.row) + ': ' + str(el.name)
                             + ': identifier(param) was already declared')
            else:
                el.index = context.get_next_local_num()
                el.data_type = self.vars_type
                el.var_type = VarType.Local if context.parent else VarType.Global
                context.add_var(el)
        for each in self.children:
            each.semantic_check(context)
        if self.vars_type is None or self.vars_type.name == BaseTypes.Void:
            logger.error(str(self.row) + ': vars cannot be void')

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        if CUR_LABEL is None:
            return ''
        for each in self.vars_list:
            if isinstance(each, AssignNode):
                code += each.code_generate(funcs, global_vars, global_arrays)
        return code


class CallNode(StmtNode):
    """Function call node"""

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
                    arr = {**context.get_function_context().arrays,
                           **context.get_function_context().parent.arrays}
                    self.const = arr[list(filter(lambda k: k.index == self.params[0].index,
                                                 arr))[0]].num.const \
                                 * LLVM_TYPES[self.params[0].data_type.name.value][1]
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
                        str(self.row) + ': ' + str(self.func_name.name)
                        + ': not enough arguments in the function call')
                elif len(self.params) > len(self.func.params):
                    logger.error(
                        str(self.row) + ': ' + str(self.func_name.name)
                        + ': too many arguments in the function call')
                else:
                    for i, p in enumerate(self.params):
                        if str(self.func.params[i].data_type) != str(p.data_type):
                            if p.data_type.is_compatible_with(self.func.params[i].data_type):
                                self.params[i] = CastNode(self.params[i],
                                                          self.func.params[i].data_type,
                                                          self.params[i].const)
                            else:
                                logger.error(str(self.row) + ': '
                                             + str(self.func_name.name) + ': argument ('
                                             + str(i + 1) + ') of type \''
                                             + str(p.data_type)
                                             + '\' is not compatible with type \'' +
                                             str(self.func.params[i].data_type) + '\'')
            elif len(v) == 0:
                p_str = ''
                for p in self.params:
                    p_str += (', ' if p_str else '') + str(p.data_type)
                logger.error(str(self.row) + ': {0}({1}) was not declared'
                             .format(str(self.func_name.name),
                                     p_str))
            else:
                str_params = ''
                for param in self.params:
                    str_params += (', ' if str_params else '') + str(param.data_type)
                logger.error(str(self.row)
                             + ': there is more than one function named \'{0}\' that match ({1})'
                             .format(self.func_name.name, str_params))

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0):
        code = ''
        if CUR_LABEL is None:
            return ''
        params_code = ''
        global VAR_NUMBER
        for arg in self.params[::-1]:
            if arg.const is None:
                code += arg.code_generate(funcs, global_vars, global_arrays)
            t = '%' + str(VAR_NUMBER) if arg.const is None else arg.const
            params_code = '{0} {1}'.format(LLVM_TYPES[str(arg.data_type)][0], t) + (
                ', ' if params_code else '') + params_code
        if self.data_type.name is not BaseTypes.Void:
            VAR_NUMBER += 1
            code += '  %{0} = call {1} {2}({3})\n' \
                .format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                        self.func.LLVM_name(),
                        params_code)
        else:
            code += '  call {0} {1}({2})\n' \
                .format(LLVM_TYPES[str(self.data_type)][0],
                        self.func.LLVM_name(),
                        params_code)

        return code


class FunctionNode(StmtNode):
    """Function declaration node"""

    def __init__(self, data_type: Type, name: str, params: Tuple = None, *body: Tuple[StmtNode],
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.name = name
        self.params = self.check_params(params) if params else list()
        self.body = body if body else (StatementListNode(),)
        self.args_list = None
        self.return_num = None
        self.arrays = None

    def check_params(self, params):
        """Checks function parameters"""
        return list(IdentNode(param.children[1], param.vars_type) for param in params if
                    param.vars_type.name)

    @property
    def children(self) -> Tuple[AstNode]:
        return (*self.body,)

    def LLVM_name(self) -> str:
        """Provides LLVM name"""
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
        """Finds variable by name"""
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
                s += '(dtype={0}, vtype={1}, index={2})' \
                    .format(e.data_type, e.var_type.value, e.index)
        if s:
            return '{0} {1} ({2}) ({3})'.format(self.data_type, self.name, params, s)
        else:
            return '{0} {1} ({2})'.format(self.data_type, self.name, params)

    def semantic_check(self, context: Context = None):
        if context.check_if_was_declared(self.name, self.params):
            str_params = ''
            for param in self.params:
                str_params += (', ' if str_params else '') + str(param.data_type)
            logger.error(str(self.row) + ': function \'{0}\'({1}) was already declared'
                         .format(self.name, str_params))
            return
        context.add_function(self)
        context = Context(context)
        for el in self.params:
            el.index = context.get_next_param_num() + len(self.params) + 1
            if not context.is_param_added(el.name):
                context.add_param(el)
                el.var_type = VarType.Param
            else:
                logger.error(str(self.row) + ': param with name \'{0}\' already declared'
                             .format(el.name))
        new_context = Context(context)

        ret = self.find_return(self.children, [])
        for each in self.children:
            each.semantic_check(new_context)

        if self.name == 'main' and len(self.params) > 0:
            if self.params[0].data_type.name is not BaseTypes.Int or self.params[0].data_type.rang:
                logger.error(str(self.row)
                             + ' : first parameter of \'main\' must be of type \'int\'')

        self.args_list = context.vars
        self.arrays = context.arrays
        if self.data_type.name is not BaseTypes.Void:
            if len(ret) == 0:
                logger.error(str(self.row) + ': function must return some value')
            elif len(ret) == 1 and isinstance(self.children[-1], ReturnNode):
                ret[0].ok = True
            else:
                self.return_num = context.get_next_local_num()
                for r in ret:
                    r.ok = False
                    r.index = self.return_num

    def find_return(self, children, count):
        """Finds return statement"""
        for each in children:
            if isinstance(each, ReturnNode):
                count.append(each)
                each.data_type = self.data_type
            count.extend(self.find_return(each.children, []))
        return count

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        for _, v in self.arrays.items():
            if v.data_type.name is BaseTypes.Char:
                sym_str = ''.join(
                    '\\' + hex(x.const)[hex(x.const).find('x') + 1:].capitalize().rjust(2, '0')
                    for x in v.elements)
                code += '@{0}.{1} = dso_local unnamed_addr constant [{2} x i8] c"{3}", align 1\n' \
                    .format(self.name,
                            v.all_elements_hex(),
                            v.num.const,
                            sym_str)
        code += '\n'
        sb = body = ''
        for param in self.params:
            sb += (', ' if sb else '') + LLVM_TYPES[str(param.data_type)][0]
        code += 'define dso_local {0} {1}({2}) {{\n' \
            .format(LLVM_TYPES[str(self.data_type)][0], self.LLVM_name(), sb)

        global VAR_NUMBER, CUR_LABEL
        VAR_NUMBER = len(self.params)
        CUR_LABEL = VAR_NUMBER
        for elem in (*self.params, *self.args_list):
            VAR_NUMBER += 1
            if elem.data_type.rang == 0:
                body += '  %{0} = alloca {1}, align {2}\n' \
                    .format(VAR_NUMBER, *LLVM_TYPES[str(elem.data_type)])
            elif elem.data_type.name is BaseTypes.Char:
                body += '  %{0} = alloca i8*, align 8\n'.format(VAR_NUMBER)
            else:
                num = list(filter(lambda x: x.index == elem.index, self.arrays))
                num = self.arrays[num[0]].num.const if num else 0
                body += '  %{0} = alloca [{1} x {2}], align 16\n' \
                    .format(VAR_NUMBER, num,
                            LLVM_TYPES[elem.data_type.name.value][0])
            elem.index = VAR_NUMBER
        if self.return_num is not None:
            VAR_NUMBER += 1
            body += '  %{0} = alloca {1}, align {2}\n' \
                .format(VAR_NUMBER, *LLVM_TYPES[self.data_type.name.value])
        for i in range(len(self.params), 0, -1):
            body += '  store {0} %{1}, {0}* %{2}, align {3}\n' \
                .format(LLVM_TYPES[str(self.params[i - 1].data_type)][0],
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
            body += '  %{0} = load {1}, {1}* %{2}, align {3}\n' \
                .format(VAR_NUMBER, LLVM_TYPES[str(self.data_type)][0],
                        self.return_num,
                        LLVM_TYPES[str(self.data_type)][1])
            body += '  ret {0} %{1}\n'.format(LLVM_TYPES[str(self.data_type)][0], VAR_NUMBER)

        body = body.format(_return=VAR_NUMBER)
        code += body + '}\n\n'
        return code


class AssignNode(StmtNode):
    """Assignment node"""

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
        if isinstance(self.val, ArrayIdentNode):
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
        if self.var.data_type.rang and self.val.data_type.rang and isinstance(self.val, IdentNode):
            # con = context.get_function_context()
            # v = list(filter(lambda k: k.index ==self.val.index, con.arrays.keys()))[0]
            self.var.index = self.val.index
            self.var.var_type = self.val.var_type
            context.add_var(self.var)
            # context.add_array(self.var, con.arrays[v])
        if not self.var.data_type or not self.var.data_type \
                or not self.val.data_type or not self.val.data_type.name \
                or not self.var.data_type.name:
            return
        self.data_type = self.var.data_type
        if self.var.data_type != self.val.data_type:
            if self.val.data_type.is_compatible_with(self.var.data_type):
                # c = self.val.const
                self.val = CastNode(self.val, self.data_type, self.val.const)
            else:
                logger.error(str(self.var.row) + ': '
                             + str(self.var.name) + ': value does not match \''
                             + str(self.data_type) + '\' type')

        self.const = self.val.const

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        if CUR_LABEL is None:
            return ''
        global VAR_NUMBER
        if self.data_type.rang == 0 or \
                (self.data_type.name is BaseTypes.Char
                 and not isinstance(self.val, ArrayIdentNode)):
            code += self.val.code_generate(funcs, global_vars, global_arrays) \
                if self.val.const is None or isinstance(self.val, AssignNode) else ''
            n1 = VAR_NUMBER
            if isinstance(self.var, ElementNode):
                code += self.var.get_ptr(funcs, global_vars, global_arrays)
                if self.val.const is None:
                    code += '  store {0} %{1}, {0}* %{2}, align {3}\n' \
                        .format(LLVM_TYPES[self.data_type.name.value][0],
                                n1,
                                VAR_NUMBER,
                                LLVM_TYPES[self.data_type.name.value][1])
                else:
                    code += '  store {0} {1}, {0}* %{2}, align {3}\n' \
                        .format(LLVM_TYPES[self.data_type.name.value][0],
                                self.val.const,
                                VAR_NUMBER,
                                LLVM_TYPES[self.data_type.name.value][1])
            else:
                if self.var.var_type in (VarType.Param, VarType.Local):
                    t = '%' + str(self.var.index)  # + len(func.params))
                else:
                    t = '@' + 'g.' + str(self.var.index)
                # pass (вероятно это должно как-то быть в semantic_check)
                if isinstance(self.val, BinOpNode) \
                        and self.val.op in (BinOp.LOGICAL_AND, BinOp.LOGICAL_OR):
                    VAR_NUMBER += 1
                    code += '  %{0} = zext i1 %{1} to i8\n'.format(VAR_NUMBER, VAR_NUMBER - 1)

                if self.const is None:
                    n = VAR_NUMBER
                    code += '  store {0} %{1}, {0}* {2}, align {3}\n' \
                        .format(LLVM_TYPES[str(self.data_type)][0],
                                n,
                                t,
                                LLVM_TYPES[str(self.data_type)][1])
                else:
                    code += '  store {0} {1}, {0}* {2}, align {3}\n' \
                        .format(LLVM_TYPES[str(self.data_type)][0],
                                self.val.const,
                                t,
                                LLVM_TYPES[str(self.data_type)][1])
        elif isinstance(self.val, ArrayIdentNode) and self.data_type.name is BaseTypes.Char:
            if isinstance(self.val, ArrayIdentNode):
                code += '  store i8* getelementptr inbounds ' \
                        '([{0} x i8], [{0} x i8]* @{1}.{2}, i32 0, i32 0), i8** %{3}, align 8\n' \
                    .format(self.val.num.const,
                            funcs.name,
                            self.val.all_elements_hex(),
                            self.var.index)
            else:
                if self.var.var_type in (VarType.Param, VarType.Local):
                    t = '%' + str(self.var.index)  # + len(func.params))
                else:
                    t = '@' + 'g.' + str(self.var.index)
                code += self.val.code_generate(funcs, global_vars, global_arrays)
                code += '  store {0} %{1}, {0}* {2}, align {3}\n' \
                    .format(LLVM_TYPES[str(self.data_type)][0],
                            VAR_NUMBER,
                            t,
                            LLVM_TYPES[str(self.data_type)][1])
        elif isinstance(self.val, ArrayIdentNode):
            num = None
            for el in self.val.elements:
                n1 = VAR_NUMBER
                VAR_NUMBER += 1
                if self.var.var_type == VarType.Global:
                    t = '@' + 'g.' + str(self.var.index)
                else:
                    t = '%' + str(self.var.index)
                if num is None:
                    code += '  %{0} = getelementptr inbounds ' \
                            '[{1} x {2}], [{1} x {2}]* {3}, i64 0, i64 0\n' \
                        .format(VAR_NUMBER,
                                self.val.num.const,
                                LLVM_TYPES[self.data_type.name.value][0],
                                t)
                else:
                    code += '  %{0} = getelementptr inbounds {1}, {1}* %{2}, i64 1\n' \
                        .format(VAR_NUMBER, LLVM_TYPES[self.data_type.name.value][0], num)
                num = VAR_NUMBER
                if el.const is None:
                    code += el.code_generate(funcs, global_vars, global_arrays)
                    code += '  store {0} %{1}, {0}* %{2}, align {3}\n' \
                        .format(LLVM_TYPES[self.data_type.name.value][0],
                                VAR_NUMBER,
                                num,
                                LLVM_TYPES[self.data_type.name.value][1])
                else:
                    code += '  store {0} {1}, {0}* %{2}, align {3}\n' \
                        .format(LLVM_TYPES[self.data_type.name.value][0],
                                el.const,
                                num,
                                LLVM_TYPES[self.data_type.name.value][1])
        return code

    def __str__(self) -> str:
        c = d = sb = ''
        if self.data_type:
            d = 'dtype=' + str(self.data_type)
        if self.const is not None:
            c = 'const=' + str(self.const)
        if d:
            sb = d
        if c:
            sb = sb + (', ' if sb else '') + c
        return '=' + (' (' + sb + ')' if sb else '')
