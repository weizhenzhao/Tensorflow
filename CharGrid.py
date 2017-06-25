'''
Created on Apr 18, 2017

@author: P0079482
'''
class RangeError(Exception):pass
class RowRangeError(RangeError):pass
class ColumnRangeError(RangeError):pass

_CHAR_ASSERT_TEMPLATE=("char must be a single character:'{0}'"
                       "is too long")
_max_rows=25
_max_columns=80
_grid=[]
_background_char=" "
#用下划线开头的是私有数据。
#如果使用from CharGrid import *导入，则这些私有变量都不会被实际导入


#如果要清空控制台就需要调用系统底层的方法
#import sys
#if sys.platform.startswith("win"):
#    def clear_screen():
#        subprocess.call(["cmd.exe","/C","cls"])
#else:
#    def clear_screen():
#        subprocess.call(["clear"])
#clear_screen.__doc__="""Clears the screen using the underlying \
#window system's clear screen command"""

#但是使用下边这个方法就可以有效地处理平台之间的差别
#def clear_screen():
#    command=(["clear"] if not sys.platform.startswith("win") else ["cmd.exe","/C","cls"])
#    subprocess.call(command)


def resize(max_rows,max_columns,char=None):
       """Changes the size of the grid, wiping out the contents and changing the background if the
       background char is not None
       """
       assert max_rows > 0 and max_columns > 0, "too small"
       global _grid,_max_rows,_max_columns,_background_char
       if char is not None:
           #使用assert语句来指定一种策略，即将网格大小重定义为小于1*1应该被视为一种编码错误
           #必须使用global语句，因为我们需要在函数内部对大量全局变量进行更新。通过使用面向对象的方法，
           #可以避免这种做法。
           #_grid是通过在列表内涵内部使用列表内涵创建的，使用列表复制，
           assert len(char)==1,_CHAR_ASSERT_TEMPLATE.format(char)
           _background_char=char
           _max_rows=max_rows
           _max_columns=max_columns
           _grid=[[_background_char for column in range(_max_columns)] for row in range(_max_rows)]

#_grid=[[_background_char for column in range(_max_columns)] for row in range(_max_rows)]
#下边这段代码相当于上边那条注释了的代码
#_grid=[]
#for row in range(_max_rows):
#    _grid.append([])
#    for column in range(_max_columns):
#        _grid[-1].append(_background_char)

#查看其中的一个绘图函数
def add_horizontal_line(row,column0,column1,char="-"):
    """Adds a horizontal line to the grid using the given char
    
    >>>add_horizontal_line(8,20,25,"=")
    >>>char_at(8,20)==char_at(8,24)=="="
    True
    >>>add_horizontal_line(31,11,12)
    char_at()函数返回网格中指定行列处的字符
    """
    assert len(char)==1,_CHAR_ASSERT_TEMPLATE.format(char)
    try:
        for column in range(column0,column1):
            _grid[row][column]=char
    except IndexError:
            if not 0<=row<=_max_rows:
                raise RowRangeError()
            raise ColumnRangeError()
















