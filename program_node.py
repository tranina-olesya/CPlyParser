"""Program node"""

from typing import Optional

from ast_nodes import IdentNode, ArrayIdentNode
from statement_nodes import AssignNode, FunctionNode, VarsDeclNode
from ast_nodes import StmtNode, StatementListNode, Type
from constants import LLVM_TYPES, DEFAULT_VALUES
from context import Context
from logger import logger


class Program(StatementListNode):
    """Program node"""

    def __init__(self, *exprs: StmtNode, row: Optional[int] = None,
                 line: Optional[int] = None, **props):
        super().__init__(exprs=exprs, row=row, line=line, **props)
        self.exprs = exprs
        self.global_vars = []
        self.global_arrays = []
        self.standard_functions = [
            FunctionNode(Type('void'), 'output', (VarsDeclNode(Type('int'), (IdentNode('a'),)),)),
            FunctionNode(Type('void'), 'output',
                         (VarsDeclNode(Type('double'), (IdentNode('a'),)),)),
            FunctionNode(Type('void'), 'output', (VarsDeclNode(Type('char'), (IdentNode('a'),)),)),
            FunctionNode(Type('void'), 'output',
                         (VarsDeclNode(Type('string'), (IdentNode('a'),)),)),
            FunctionNode(Type('char'), 'to_char', (VarsDeclNode(Type('int'), (IdentNode('a'),)),)),
            FunctionNode(Type('string'), 'to_str',
                         (VarsDeclNode(Type('int'), (IdentNode('a'),)),)),
            FunctionNode(Type('string'), 'to_str',
                         (VarsDeclNode(Type('double'), (IdentNode('a'),)),)),
            FunctionNode(Type('string'), 'to_str',
                         (VarsDeclNode(Type('char'), (IdentNode('a'),)),)),
            FunctionNode(Type('int'), 'input_int'),
            FunctionNode(Type('char'), 'input_char'),
            FunctionNode(Type('string'), 'input_str'),
            FunctionNode(Type('double'), 'input_double'),
            FunctionNode(Type('string'), 'concat', (
                VarsDeclNode(Type('string'), (IdentNode('s1'),)),
                VarsDeclNode(Type('string'), (IdentNode('s2'),))))
        ]

    def semantic_check(self, context: Context = None):
        context = Context(functions=self.standard_functions[::])
        for each in self.children:
            if each:
                each.semantic_check(context)
                if isinstance(each, VarsDeclNode):
                    each: VarsDeclNode
                    for x in each.vars_list:
                        if isinstance(x, IdentNode):
                            self.global_vars.append(x)
                        elif isinstance(x, AssignNode) and not isinstance(x.val, ArrayIdentNode):
                            self.global_vars.append(x)

        for k, v in context.arrays.items():
            an = AssignNode(k, v)
            an.data_type = an.var.data_type
            self.global_vars.append(an)
        self.global_arrays = context.arrays
        if not any(isinstance(x, FunctionNode) and x.name == 'main' for x in self.children):
            logger.error('1 : function \'main\' must be declared')

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        code = ''
        for func in self.standard_functions:
            args_str = ''
            for arg in func.params:
                args_str += (', ' if args_str else '') + LLVM_TYPES[str(arg.data_type)][0]
            code += 'declare dso_local {0} {1}({2})\n\n' \
                .format(LLVM_TYPES[str(func.data_type)][0],
                        func.LLVM_name(),
                        args_str)
        for var in self.global_vars:
            if isinstance(var, IdentNode) and var.data_type.rang == 0:
                code += '@{0} = common dso_local global {1} {2}, align {3}\n' \
                    .format('g.' + str(var.index), LLVM_TYPES[var.data_type.name.value][0],
                            DEFAULT_VALUES[var.data_type.name.value],
                            LLVM_TYPES[var.data_type.name.value][1])
            elif isinstance(var, IdentNode) and var.data_type.rang == 1:
                code += '@{0} = common dso_local global [1 x i32] ' \
                        'zeroinitializer, align {1}\n' \
                    .format('g.' + str(var.index)
                            , LLVM_TYPES[var.data_type.name.value][1])
            elif isinstance(var, AssignNode) and var.var.data_type.rang == 0:
                code += '@{0} = dso_local global {1} {2}, align {3}\n' \
                    .format('g.' + str(var.var.index),
                            LLVM_TYPES[var.data_type.name.value][0],
                            var.const,
                            LLVM_TYPES[var.data_type.name.value][1])
            elif isinstance(var, AssignNode) and var.var.data_type.rang == 1:
                if len(var.val.elements) > 0:
                    el_str = ''
                    for el in var.val.elements:
                        el_str += (', ' if el_str else '') + \
                                  '{0} {1}'.format(LLVM_TYPES[var.data_type.name.value][0],
                                                   el.const)
                    code += '@{0} = dso_local global [{1} x {2}] [{3}], align 16\n' \
                        .format('g.' + str(var.var.index),
                                var.val.num.const,
                                LLVM_TYPES[var.data_type.name.value][0],
                                el_str)
                else:
                    code += '@{0} = common dso_local global [{1} x i32] ' \
                            'zeroinitializer, align {2}\n' \
                        .format('g.' + str(var.var.index), var.val.num.const,
                                LLVM_TYPES[var.data_type.name.value][1])

        code += '\n'
        for each in self.children:
            if isinstance(each, FunctionNode):
                code += each.code_generate(funcs, self.global_vars, self.global_arrays)
        return code
