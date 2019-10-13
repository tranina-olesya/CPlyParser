"""Base nodes"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

from constants import BaseTypes
from logger import logger


class AstNode(ABC):
    """Base Node class"""

    def __init__(self, row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__()
        self.row = row
        self.line = line
        for k, v in props.items():
            setattr(self, k, v)

    @property
    def children(self) -> Tuple['AstNode', ...]:
        """Node's children"""
        return ()

    @abstractmethod
    def __str__(self) -> str:
        pass

    def semantic_check(self, context: 'Context' = None):
        """Provides semantic check"""
        for each in self.children:
            each.semantic_check(context)

    def code_generate(self, funcs=None, global_vars=None, global_arrays=None, level=0) -> str:
        """Code generation"""
        code = ''
        for each in self.children:
            code += each.code_generate(funcs, global_vars, global_arrays)
        return code

    @property
    def tree(self) -> [str, ...]:
        """Print AST string"""
        res = [str(self)]
        childs_temp = self.children
        for i, child in enumerate(childs_temp):
            ch0, ch = '├', '│'
            if i == len(childs_temp) - 1:
                ch0, ch = '└', ' '
            res.extend(((ch0 if j == 0 else ch) + ' ' + s for j, s in enumerate(child.tree)))
        return res

    def __getitem__(self, index):
        return self.children[index] if index < len(self.children) else None


class Type(AstNode):
    """Variable type"""

    def __init__(self, data_type: str, rang: int = 0,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)

        if data_type == 'string':
            self.name = BaseTypes.Char
            self.rang = 1
        else:
            if data_type in (e.value for e in BaseTypes):
                self.name = BaseTypes(data_type)
            else:
                self.name = None
                logger.error(str(self.row) + ': ' + data_type + ': unknown identifier')
            self.rang = rang

    def is_castable_to(self, cast_type) -> bool:
        """Check for type cast"""
        if cast_type.rang == self.rang == 0:
            if cast_type.name == self.name:
                return True
            elif cast_type.name == BaseTypes.Bool:
                return True
            elif cast_type.name == BaseTypes.Double:
                return self.name == BaseTypes.Int
            elif cast_type.name == BaseTypes.Int:
                return self.name == BaseTypes.Char
            # elif cast_type.name == BaseTypes.Char:
            #    return self.name == BaseTypes.Int
            return False
        return False

    def is_compatible_with(self, cast_type) -> bool:
        """Check if compatible with cast type"""
        if cast_type.rang == self.rang == 0:
            if cast_type.name == self.name:
                return True
            elif cast_type.name == BaseTypes.Char and self.name == BaseTypes.Double or \
                    cast_type.name == BaseTypes.Double and self.name == BaseTypes.Char:
                return False
            return True
        return False

    def __eq__(self, other):
        return self.name == other.name and self.rang == other.rang

    def __str__(self) -> str:
        return str(self.name.value) + ('[]' if self.rang else '')


class ExprNode(AstNode, ABC):
    """Expression node"""

    def __init__(self, data_type: Type = None,
                 row: Optional[int] = None, line: Optional[int] = None, **props):
        super().__init__(row=row, line=line, **props)
        self.data_type = data_type
        self.const = None

    @abstractmethod
    def __str__(self) -> str:
        pass


class StmtNode(ExprNode, ABC):
    """Statement node"""

    @abstractmethod
    def __str__(self) -> str:
        pass
