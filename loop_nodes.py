"""Loop nodes"""
from abc import ABC
from typing import Tuple, Optional, Union

from ast_nodes import CastNode
from ast_nodes import StatementListNode
from base_nodes import StmtNode, Type, ExprNode, AstNode
from constants import BaseTypes
from context import Context
from logger import logger
from init import CUR_LABEL, VAR_NUMBER


class LoopStmtNode(StmtNode, ABC):
    """Loop statement node"""

    def __init__(self, data_type: Type = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.is_inside_loop = False


class ContinueNode(LoopStmtNode):
    """'Continue' node"""

    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

    @property
    def children(self) -> Tuple[ExprNode]:
        return tuple()

    def __str__(self) -> str:
        return 'continue'

    def semantic_check(self, context: 'Context' = None):
        if not self.is_inside_loop:
            logger.error(str(self.row) + ': \'continue\' statement not in loop')

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0):
        global CUR_LABEL
        code = '  br label %{_continue}\n\n'
        CUR_LABEL = None
        return code


class BreakNode(LoopStmtNode):
    """'Break' node"""

    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

    @property
    def children(self) -> Tuple[ExprNode]:
        return tuple()

    def __str__(self) -> str:
        return 'break'

    def semantic_check(self, context: 'Context' = None):
        if not self.is_inside_loop:
            logger.error(str(self.row) + ': \'break\' statement not in loop')

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0):
        global CUR_LABEL
        code = '  br label %{_break}\n\n'
        CUR_LABEL = None
        return code


class LoopNode(StmtNode, ABC):
    """Loop node"""

    def check_break_stmt(self, childs):
        """Checks for break statement"""
        for el in childs:
            if isinstance(el, (BreakNode, ContinueNode)):
                el.is_inside_loop = True
            self.check_break_stmt(el.children)


class ForNode(LoopNode):
    """'for' cycle node"""

    def __init__(self, init: Union[StmtNode, None], cond: Union[ExprNode, StmtNode, None],
                 step: Union[StmtNode, None], body: Union[StmtNode, None] = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.init = init if init else StatementListNode()
        self.cond = cond if cond else StatementListNode()
        self.step = step if step else StatementListNode()
        self.body = body if body else StatementListNode()

    @property
    def children(self) -> Tuple[AstNode, ...]:
        return self.init, self.cond, self.step, self.body

    def __str__(self) -> str:
        return 'for'

    def semantic_check(self, context: 'Context' = None):
        self.check_break_stmt((self.body,))
        context = Context(context)
        for each in self.children:
            each.semantic_check(context)

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        global VAR_NUMBER, CUR_LABEL
        if CUR_LABEL is None:
            return ''
        init = self.init.code_generate(funcs, global_vars, global_arrays)
        init += '  br label %{_label_cond}\n\n'

        VAR_NUMBER += 1
        label_cond = VAR_NUMBER
        CUR_LABEL = label_cond
        cond = self.cond.code_generate(funcs, global_vars, global_arrays)
        cond_value = VAR_NUMBER
        if cond:
            cond += '  br i1 %{_cond_value}, label %{_label_body}, label %{_label_end}\n\n'
        else:
            cond += '  br label %{_label_body}\n\n'
        cond = '; <label>:{_label_cond}:\n' + cond

        VAR_NUMBER += 1
        label_body = VAR_NUMBER

        CUR_LABEL = label_body
        body = '; <label>:{_label_body}:\n' + \
               self.body.code_generate(funcs, global_vars, global_arrays)
        VAR_NUMBER += 1
        label_step = VAR_NUMBER
        if CUR_LABEL is not None:
            body += '  br label %{_label_step}\n\n'

        CUR_LABEL = label_step
        step = '; <label>:{_label_step}:\n' \
               + self.step.code_generate(funcs, global_vars, global_arrays)
        VAR_NUMBER += 1
        label_end = VAR_NUMBER
        step += '  br label %{_label_cond}\n\n'

        CUR_LABEL = label_end
        code = init + cond + body + step + '; <label>:{_label_end}:\n'
        code = code.format(_cond_value=cond_value,
                           _label_cond=label_cond,
                           _label_body=label_body,
                           _label_step=label_step,
                           _label_end=label_end,
                           _break=label_end,
                           _continue=label_step,
                           _return='{_return}')
        return code


class WhileNode(LoopNode):
    """'while' cycle node"""

    def __init__(self, cond: ExprNode, body: StmtNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.cond = cond
        self.body = body if body else StatementListNode()

    @property
    def children(self) -> Tuple[ExprNode, StmtNode]:
        return self.cond, self.body

    def __str__(self) -> str:
        return 'while'

    def semantic_check(self, context: 'Context' = None):
        self.check_break_stmt((self.body,))
        self.cond.semantic_check(context)
        if self.cond.data_type and self.cond.data_type.name is not BaseTypes.Bool:
            self.cond = CastNode(self.cond, Type('bool'), self.cond.const)
        context = Context(context)
        self.body.semantic_check(context)

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0):
        global CUR_LABEL, VAR_NUMBER
        if CUR_LABEL is None:
            return ''
        VAR_NUMBER += 1
        CUR_LABEL = VAR_NUMBER
        label_cond = CUR_LABEL
        if self.cond.const is None:
            cond = self.cond.code_generate(funcs, global_vars, global_arrays)
            cond += '  br i1 %{_cond_value}, label %{_label_body}, label %{_label_end}\n\n'
        elif self.cond.const:
            cond = '  br label %{_label_body}\n\n'
        else:
            cond = '  br label %{_label_end}\n\n'
        cond = ' <label>:{_label_cond}:\n' + cond
        cond_value = VAR_NUMBER
        VAR_NUMBER += 1

        label_body = VAR_NUMBER
        CUR_LABEL = VAR_NUMBER
        body = '; <label>:{_label_body}:\n' \
               + self.body.code_generate(funcs, global_vars, global_arrays) \
               + ('  br label %{_label_cond}\n\n' if CUR_LABEL else '')
        VAR_NUMBER += 1

        label_end = VAR_NUMBER
        code = '  br label %{_label_cond}\n\n;' + cond + body + '; <label>:{_label_end}:\n'
        CUR_LABEL = label_end
        code = code.format(_cond_value=cond_value,
                           _label_cond=label_cond,
                           _label_body=label_body,
                           _label_end=label_end,
                           _break=label_end,
                           _continue=label_cond,
                           _return='{_return}')
        return code


class DoWhileNode(LoopNode):
    """'do while' cycle node"""

    def __init__(self, body: StmtNode, cond: ExprNode,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.body = body
        self.cond = cond

    @property
    def children(self) -> Tuple[StmtNode, ExprNode]:
        return self.body, self.cond

    def __str__(self) -> str:
        return 'do while'

    def semantic_check(self, context: 'Context' = None):
        self.check_break_stmt((self.body,))
        context = Context(context)
        for each in self.children:
            each.semantic_check(context)
        if self.cond.data_type.name is not BaseTypes.Bool:
            self.cond = CastNode(self.cond, Type('bool'), self.cond.const)

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0):
        global CUR_LABEL, VAR_NUMBER
        if CUR_LABEL is None:
            return ''
        VAR_NUMBER += 1
        CUR_LABEL = VAR_NUMBER
        label_body = CUR_LABEL
        body = '; <label>:{_label_body}:\n' \
               + self.body.code_generate(funcs, global_vars, global_arrays)\
               + ('  br label %{_label_cond}\n\n' if CUR_LABEL else '')
        VAR_NUMBER += 1

        label_cond = VAR_NUMBER
        CUR_LABEL = VAR_NUMBER

        if self.cond.const is None:
            cond = self.cond.code_generate(funcs, global_vars, global_arrays)
            cond += '  br i1 %{_cond_value}, label %{_label_body}, label %{_label_end}\n\n'
        elif self.cond.const:
            cond = '  br label %{_label_body}\n\n'
        else:
            cond = '  br label %{_label_end}\n\n'
        cond_value = VAR_NUMBER

        VAR_NUMBER += 1
        label_end = VAR_NUMBER
        cond = '; <label>:{_label_cond}:\n' + cond
        code = '  br label %{_label_body}\n\n' + body + cond + '; <label>:{_label_end}:\n'
        code = code.format(_cond_value=cond_value,
                           _label_cond=label_cond,
                           _label_body=label_body,
                           _label_end=label_end,
                           _break=label_end,
                           _continue=label_cond,
                           _return='{_return}')
        CUR_LABEL = label_end
        return code
