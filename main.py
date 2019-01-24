from cTreeParser import *
import os
import sys


test4 = '''
void main() {

int an, size = an = 7;
}
'''
#build_tree(test4)
code_generate(sys.argv[1])
