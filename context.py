"""Context"""

from constants import BaseTypes


class Context:
    """Context"""

    def __init__(self, parent: 'Context' = None, functions=None):
        self.vars = []
        self.functions = functions
        self.parent = parent
        self.params = []
        self.arrays = {}

    def add_var(self, var: 'IdentNode'):
        """Adds variable (IdentNode)"""
        self.vars.append(var)
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
            loc_context.vars.append(var)

    def add_array(self, name: 'IdentNode', val: 'ArrayIdentNode'):
        """Adds array (IdentNode) with value (ArrayIdentNode)"""
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent

        if not any((x.all_elements_hex() == val.all_elements_hex()
                    if x.data_type.name is BaseTypes.Char else 0)
                   for x in loc_context.arrays.values()):
            loc_context.arrays[name] = val

    def add_param(self, var: 'IdentNode'):
        """Adds parameter (IdentNode)"""
        self.params.append(var)

    def add_function(self, func: 'FunctionNode'):
        """Adds function (FunctionNode)"""
        self.functions.append(func)

    def find_var(self, var_name) -> 'IdentNode':
        """Finds variable by name"""
        loc_var = None
        loc_context = self
        while not loc_var and loc_context:
            loc_var = list(filter(lambda x: x.name == var_name,
                                  (*loc_context.vars[::-1],
                                   *loc_context.params)))
            if loc_var:
                loc_var = loc_var[0]
            loc_context = loc_context.parent
        return loc_var

    def get_function_context(self) -> 'Context':
        """Provides context"""
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
        return loc_context

    def is_param_added(self, param_name):
        """Checks if parameter was added"""
        return list(filter(lambda x: x.name == param_name, self.params))

    def find_functions(self, func_name, func_args):
        """Finds function by name and arguments"""
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        func = []
        for function in loc_context.functions:
            if function.name == func_name and len(function.params) == len(func_args):
                if all(func_args[i].data_type == function.params[i].data_type
                       for i in range(len(function.params))):
                    return (function, )
                elif all(func_args[i].data_type.is_compatible_with(function.params[i].data_type)
                         for i in range(len(function.params))):
                    func.append(function)
        return func

    def check_if_was_declared(self, func_name, func_params) -> bool:
        """Checks if function with given name and arguments was added"""
        loc_context = self
        while loc_context.parent:
            loc_context = loc_context.parent
        for function in loc_context.functions:
            if function.name == func_name and len(function.params) == len(func_params):
                if all(func_params[i].data_type == function.params[i].data_type
                       for i in range(len(function.params))):
                    return True
        return False

    def get_next_local_num(self):
        """Provides next variable number"""
        loc_context = self
        if loc_context.parent:
            while loc_context.parent.parent:
                loc_context = loc_context.parent
        return len(loc_context.vars) + 2 * len(loc_context.params) + 1

    def get_next_param_num(self):
        """Provides next parameter number"""
        return len(self.params)
