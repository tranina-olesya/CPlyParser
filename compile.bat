@echo off

set PYTHON="D:\python\компиляторы\CTreeLark\venv\Scripts\python.exe"

%PYTHON% "%~dp0main.py" %1
clang -o %1.exe hello.ll %1.ll 
