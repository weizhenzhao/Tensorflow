'''
Created on Apr 18, 2017

@author: P0079482
'''

#!/usr/bin/env python3
#Copyright (c) 2008 Qtrac Ltd .All rights reserved.
"""
This module provides a few string manipulation functions
>>>is_balanced("(Python (is not (lisp)))")
True
>>>shorten("The Crossing",10)
'The Cro...'
>>>simplify(" some text with spurious whitespace ")
'some text with spurious whitespace'
"""
import string
#一个可以化简字符串的函数
def simplify(text,whitespace=string.whitespace,delete=""):
    r"""Returns the text with multiple spaces reduced to single spaces

        The whitespace parameter is a string of characters,each of which
        is considered to be a space.
        if delete is not empty it should be a string,in which case any
        characters in the delete string are excluded from the resultant
        stirng.
    >>>simplify(" this     and\n that\t too ")
    'this and that too'
    >>>simplify("  Washington       D.C\n  ")
    'Washington D.C.'
    >>>simplify("  Washington        D.C.\n",delete=",;:.")
    'Washington DC'
    >>>simplify(" disemvoweled ",delete="aeiou")
    'dsmvwld'
    """
    result=[]
    word=""
    for char in text:
        if char in delete:
            continue
        elif char in whitespace:
            if word:
                result.append(char)
                word=""
        else:
            result.append(char)
            word+=char
    return "".join(result)

#该函数构造两个字典,count字典的键是开字符
def is_balanced(text,brackets="()[]{}<>"):
    counts={}
    left_for_right={}
    for left,right in zip(brackets[::2],brackets[1::2]):
        assert left!=right,"the bracket characters must differ"
        counts[left]=0
        left_for_right[right]=left
    for c in text:
        if c in counts:
            counts[c]+=1
        elif c in left_for_right:
            left=left_for_right[c]
            if counts[left]==0:
                return False
            counts[left]-=1
    return not any(counts.values())

#任何模块被导入后，Python都将为该模块创建一个名为__name__的变量，并将其设置为字符串"__main__"
#doctest.testmod()函数使用Python的内省功能来发现模块及其docstrings中的所有函数，并尝试执行其发现的所有
#docstring代码段
#doctest模块不仅可以发现模块docstring中的测试用例，还可以发现函数docstrings中的测试用例
if __name__=="__main__":
    import doctest
    doctest.testmod()
