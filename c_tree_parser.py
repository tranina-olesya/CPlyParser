import os

import ply.lex as lex
import ply.yacc as yacc

from ast_nodes import *
from loop_nodes import *
from statement_nodes import *
from program_node import Program

tokens = [
    'NUMBER', 'IDENT', 'STRING', 'CHAR',
    'ADD', 'SUB', 'MUL', 'DIV', 'MOD',
    'INC_OP', 'DEC_OP',
    'ASSIGN', 'ADD_ASSIGN', 'SUB_ASSIGN', 'MUL_ASSIGN', 'DIV_ASSIGN', 'MOD_ASSIGN',
    'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'LBRACKET', 'RBRACKET', 'BRACKETS',
    'SEMICOLON', 'COMMA',
    'GT', 'LT', 'GE', 'LE',
    'EQUALS', 'NOTEQUALS',
    'OR', 'AND', 'NOT',
]

reserved = {
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'while': 'WHILE',
    'do': 'DO',
    'true': 'TRUE',
    'false': 'FALSE',
    'return': 'RETURN',
    'break': 'BREAK',
    'continue': 'CONTINUE',
    'new': 'NEW',
}

tokens += reserved.values()

t_ADD = r'\+'
t_SUB = r'-'
t_MUL = r'\*'
t_DIV = r'/'
t_MOD = r'%'
t_ASSIGN = r'='
t_ADD_ASSIGN = r'\+='
t_SUB_ASSIGN = r'-='
t_MUL_ASSIGN = r'\*='
t_DIV_ASSIGN = r'/='
t_MOD_ASSIGN = r'%='
t_INC_OP = r'\+\+'
t_DEC_OP = r'--'
t_LPAREN = r'\('
t_RPAREN = r'\)'
t_LBRACE = r'{'
t_RBRACE = r'}'
t_LBRACKET = r'\['
t_RBRACKET = r'\]'

t_SEMICOLON = r';'
t_COMMA = r','
t_GT = r'>'
t_LT = r'<'
t_EQUALS = r'=='
t_NOTEQUALS = r'!='
t_GE = r'>='
t_LE = r'<='
t_OR = r'\|\|'
t_AND = r'&&'
t_NOT = r'!'
t_NUMBER = r'\d+\.?\d*([eE][+-]?\d+)?'
t_BRACKETS = r'\[\s*\]'
t_ignore = ' \r\t'
t_STRING = r'"(.|\n)*?"'
t_CHAR = r"'(.|\\(r|n|t))'"


def t_IDENT(t):
    r'[a-zA-Z_][a-zA-Z0-9_]*'
    if t.value in reserved:
        t.type = reserved[t.value]
    return t


def t_ccode_comment(t):
    r'(/\*(.|\n)*?\*/)|//.*'
    t.lexer.lineno += t.value.count("\n")


def t_newline(t):
    r'\n+'
    t.lexer.lineno += t.value.count("\n")


def t_error(t):
    logger.error(str(t.lineno) + ": illegal character '%s'" % t.value[0])


lexer = lex.lex()


def p_translation_unit(t):
    '''translation_unit :
                        | external_declaration
                        | translation_unit external_declaration'''
    if len(t) > 2:
        if t[2]:
            t[1].add_child(t[2])
        t[0] = t[1]
    elif len(t) > 1:
        t[0] = Program(t[1])
    else:
        t[0] = Program()


def p_external_declaration(t):
    '''external_declaration : semicolons
                            | vars_declaration semicolons
                            | function_definition'''
    t[0] = t[1]


def p_statement_list(t):
    '''statement_list :
                      | statement_list statement'''
    if len(t) > 1:
        if t[2]:
            t[1].add_child(t[2])
        t[0] = t[1]
    else:
        t[0] = StatementListNode(row=t.lexer.lineno)


def p_statement(t):
    '''statement : simple_statement
                 | block
                 | selection_statement
                 | iteration_statement
                 | jump_statement semicolons'''
    t[0] = t[1]


def p_simple_statement(t):
    '''simple_statement : semicolons
                        | expression semicolons'''
    t[0] = t[1]


def p_block(t):
    'block : LBRACE statement_list RBRACE'
    t[0] = BlockNode(*t[2].children)


def p_selection_statement(t):
    'selection_statement : if'
    t[0] = t[1]


def p_jump_statement(t):
    '''jump_statement : return
                      | continue
                      | break'''
    t[0] = t[1]


def p_return(t):
    '''return : RETURN
              | RETURN logical_expression
              | RETURN assignment'''
    if len(t) > 2:
        t[0] = ReturnNode(t[2], row=t.lexer.lineno)
    else:
        t[0] = ReturnNode(row=t.lexer.lineno)


def p_continue(t):
    '''continue : CONTINUE'''
    t[0] = ContinueNode(row=t.lexer.lineno)


def p_break(t):
    '''break : BREAK'''
    t[0] = BreakNode(row=t.lexer.lineno)


def p_iteration_statement(t):
    '''iteration_statement : for
                           | while
                           | dowhile'''
    t[0] = t[1]


def p_expression(t):
    '''expression : logical_expression
                  | assignment
                  | vars_declaration'''
    t[0] = t[1]


def p_expression_list(t):
    '''expression_list :
                       | expression
                       | expression_list COMMA expression'''
    if len(t) > 2:
        t[1].add_child(t[3])
        t[0] = t[1]
    elif len(t) > 1:
        t[0] = StatementListNode(t[1], row=t.lexer.lineno)
    else:
        t[0] = StatementListNode(row=t.lexer.lineno)


def p_logical_expression(t):
    'logical_expression : logical_or_expression'
    t[0] = t[1]


def p_logical_or_expression(t):
    '''logical_or_expression : logical_and_expression
                             | logical_or_expression OR logical_and_expression'''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_logical_and_expression(t):
    '''logical_and_expression : equality_expression
                              | logical_and_expression AND equality_expression'''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_equality_expression(t):
    '''equality_expression : relational_expression
                           | equality_expression EQUALS relational_expression
                           | equality_expression NOTEQUALS relational_expression '''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_relational_expression(t):
    '''relational_expression : additive_expression
                             | relational_expression GT additive_expression
                             | relational_expression LT additive_expression
                             | relational_expression GE additive_expression
                             | relational_expression LE additive_expression'''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_additive_expression(t):
    '''additive_expression : multiplicative_expression
                           | additive_expression ADD multiplicative_expression
                           | additive_expression SUB multiplicative_expression'''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_multiplicative_expression(t):
    '''multiplicative_expression : unary_expression
                                 | multiplicative_expression MUL unary_expression
                                 | multiplicative_expression DIV unary_expression
                                 | multiplicative_expression MOD unary_expression'''
    if len(t) > 2:
        t[0] = BinOpNode(BinOp(t[2]), t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_unary_expression(t):
    '''unary_expression : postfix_expression
                        | NOT group
                        | SUB group
                        | ADD group'''
    if len(t) > 2:
        t[0] = UnOpNode(UnOp(t[1]), t[2], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_postfix_expression(t):
    '''postfix_expression : group
                          | lvalue INC_OP
                          | lvalue DEC_OP'''
    if len(t) > 2:
        t[0] = AssignNode(t[1], BinOpNode(BinOp(t[2][0]), t[1], LiteralNode(str(1)), row=t.lexer.lineno),
                          row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_group(t):
    '''group : call
             | lvalue
             | LPAREN logical_expression RPAREN
             | number
             | string
             | char
             | bool_value'''

    if len(t) > 2:
        t[0] = t[2]
    else:
        t[0] = t[1]


def p_if(t):
    '''if : IF LPAREN logical_expression RPAREN statement
          | IF LPAREN logical_expression RPAREN statement ELSE statement'''

    if len(t) > 6:
        t[0] = IfNode(t[3], t[5], t[7], row=t.lexer.lineno)
    else:
        t[0] = IfNode(t[3], t[5], row=t.lexer.lineno)


def p_assignment(t):
    '''assignment : lvalue assignment_operation rvalue'''
    if t[2][0] == '=':
        t[0] = AssignNode(t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = AssignNode(t[1], BinOpNode(BinOp(t[2][0]), t[1], t[3], row=t.lexer.lineno), row=t.lexer.lineno)


def p_rvalue(t):
    '''rvalue : logical_expression
              | array_value
              | lvalue assignment_operation rvalue'''
    if len(t) > 2:
        if t[2][0] == '=':
            t[0] = AssignNode(t[1], t[3], row=t.lexer.lineno)
        else:
            t[0] = AssignNode(t[1], BinOpNode(BinOp(t[2][0]), t[1], t[3], row=t.lexer.lineno), row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_assignment_operation(t):
    '''assignment_operation : ASSIGN
                            | ADD_ASSIGN
                            | SUB_ASSIGN
                            | MUL_ASSIGN
                            | DIV_ASSIGN
                            | MOD_ASSIGN'''
    t[0] = t[1]


def p_simple_rvalue(t):
    '''simple_rvalue : logical_expression
                     | lvalue ASSIGN simple_rvalue'''
    if len(t) > 2:
        t[0] = AssignNode(t[1], t[3], row=t.lexer.lineno)
    else:
        t[0] = t[1]


def p_string(t):
    '''string : STRING'''
    t[0] = ArrayIdentNode(Type('char', rang=1, row=t.lexer.lineno), None,
                          *t[1].replace(r'\n', '\n').replace(r'\r', '\r').replace(r'\t', '\t'), row=t.lexer.lineno)


def p_char(t):
    '''char : CHAR'''
    t[0] = LiteralNode(t[1].replace(r'\n', '\n').replace(r'\r', '\r').replace(r'\t', '\t'), row=t.lexer.lineno)


def p_call(t):
    '''call : ident LPAREN args_list RPAREN'''
    t[0] = CallNode(t[1], *t[3].children, row=t.lexer.lineno)


def p_function_definition(t):
    '''function_definition : type ident LPAREN arguments_declaration_list RPAREN block'''
    t[0] = FunctionNode(t[1], t[2].name, t[4].children, *t[6].children, row=t.lexer.lineno)


def p_arguments_declaration_list(t):
    '''arguments_declaration_list :
                                  | argument_declaration
                                  | arguments_declaration_list COMMA argument_declaration'''
    if len(t) > 2:
        t[1].add_child(t[3])
        t[0] = t[1]
    elif len(t) > 1:
        t[0] = StatementListNode(t[1], row=t.lexer.lineno)
    else:
        t[0] = StatementListNode(row=t.lexer.lineno)


def p_argument_declaration(t):
    '''argument_declaration : type ident'''
    t[0] = VarsDeclNode(t[1], t[2], row=t.lexer.lineno)


def p_args_list(t):
    '''args_list :
                 | logical_expression
                 | args_list COMMA logical_expression'''
    if len(t) > 2:
        t[1].add_child(t[3])
        t[0] = t[1]
    elif len(t) > 1:
        t[0] = StatementListNode(t[1], row=t.lexer.lineno)
    else:
        t[0] = StatementListNode(row=t.lexer.lineno)


def p_vars_declaration(t):
    '''vars_declaration : type init_declarator_list
                        | type_array init_array_declarator_list'''
    if len(t) > 3:
        t[0] = VarsDeclNode(t[1], *t[4].children, row=t.lexer.lineno)
    else:
        t[0] = VarsDeclNode(t[1], *t[2].children, row=t.lexer.lineno)


def p_ident(t):
    '''ident : IDENT'''
    t[0] = IdentNode(t[1], row=t.lexer.lineno)


def p_type(t):
    '''type : IDENT'''
    t[0] = Type(t[1], row=t.lexer.lineno)


def p_type_array(t):
    '''type_array : IDENT BRACKETS'''
    t[0] = Type(t[1], 1, row=t.lexer.lineno)


def p_init_declarator_list(t):
    '''init_declarator_list : init_declarator
                            | init_declarator_list COMMA init_declarator'''
    if len(t) > 2:
        t[1].add_child(t[3])
        t[0] = t[1]
    else:
        t[0] = StatementListNode(t[1], row=t.lexer.lineno)


def p_init_array_declarator_list(t):
    '''init_array_declarator_list : init_array_declarator
                                  | init_array_declarator_list COMMA init_array_declarator'''
    if len(t) > 2:
        t[1].add_child(t[3])
        t[0] = t[1]
    else:
        t[0] = StatementListNode(t[1], row=t.lexer.lineno)


def p_init_declarator(t):
    '''init_declarator : ident
                       | ident_initializer'''
    t[0] = t[1]


def p_init_array_declarator(t):
    '''init_array_declarator : array_ident
                             | array_initializer'''
    t[0] = t[1]


def p_ident_initializer(t):
    '''ident_initializer : ident ASSIGN simple_rvalue'''
    t[0] = AssignNode(t[1], t[3], row=t.lexer.lineno)


def p_array_initializer(t):
    '''array_initializer : ident ASSIGN array_value'''
    t[0] = AssignNode(t[1], t[3], row=t.lexer.lineno)


def p_array_value(t):
    '''array_value : NEW type LBRACKET logical_expression RBRACKET
                   | NEW type BRACKETS LBRACE args_list RBRACE
                   | NEW type LBRACKET logical_expression RBRACKET LBRACE args_list RBRACE
                   | string '''
    if len(t) == 2:
        t[0] = t[1]
    else:
        t[2].rang = 1
        if len(t) == 6:
            t[0] = ArrayIdentNode(t[2], t[4], row=t.lexer.lineno)
        elif len(t) == 7:
            t[0] = ArrayIdentNode(t[2], None, *t[5].children, row=t.lexer.lineno)
        else:
            t[0] = ArrayIdentNode(t[2], t[4], *t[7].children, row=t.lexer.lineno)


def p_array_ident(t):
    '''array_ident : ident'''
    t[0] = IdentNode(t[1], row=t.lexer.lineno)


def p_lvalue(t):
    '''lvalue : ident
              | get_element'''
    t[0] = t[1]


def p_get_element(t):
    '''get_element : ident LBRACKET logical_expression RBRACKET'''
    t[0] = ElementNode(t[1], t[3], row=t.lexer.lineno)


def p_for(t):
    '''for : FOR LPAREN expression_list SEMICOLON for_condition SEMICOLON expression_list RPAREN statement'''
    if type(t[9]) is BlockNode:
        t[9] = StatementListNode(*t[9].children)
    t[0] = ForNode(t[3], t[5], t[7], t[9], row=t.lexer.lineno)


def p_for_condition(t):
    '''for_condition :
                     | logical_expression'''
    if len(t) > 1:
        t[0] = t[1]


def p_dowhile(t):
    '''dowhile : DO statement WHILE LPAREN logical_expression RPAREN semicolons'''
    if type(t[2]) is BlockNode:
        t[2] = StatementListNode(*t[2].children)
    t[0] = DoWhileNode(t[2], t[5], row=t.lexer.lineno)


def p_while(t):
    '''while : WHILE LPAREN logical_expression RPAREN statement'''
    t[0] = WhileNode(t[3], t[5], row=t.lexer.lineno)


def p_bool_value(t):
    '''bool_value : TRUE
                  | FALSE'''
    t[0] = LiteralNode(t[1], row=t.lexer.lineno)


def p_expression_number(t):
    'number : NUMBER'
    t[0] = LiteralNode(t[1], row=t.lexer.lineno)


def p_semicolons(p):
    '''semicolons : SEMICOLON
                  | semicolons SEMICOLON'''


def p_error(t):
    logger.error(str(t.lineno) + ': \'' + t.value + '\'' + ': syntax error')


parser = yacc.yacc()


def build_tree(s):
    p = parser.parse(s)
    print(*p.tree, sep=os.linesep)

    p.semantic_check()

    if not logger.error.counter:
        print(*p.tree, sep=os.linesep)
        print(p.code_generate())


def code_generate(file_name):
    with open(file_name, 'r', encoding='cp866') as file:
        s = file.read()
    p = parser.parse(s)
    p.semantic_check()
    w_file_name = file_name + '.ll'
    if not logger.error.counter:
        with open(w_file_name, 'w') as file:
            file.write(p.code_generate())
