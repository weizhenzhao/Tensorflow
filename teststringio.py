#python提供了两种将文本写入到文件的不同方法，
#一种是使用文件对象的write()方法
#另一种是使用print()函数，并将其关键字参数file设置为打开并等待写入的文件对象
import sys
print("An error message",file=sys.stdout)
sys.stdout.write("Another error message\n")
#sys.stdout表示是控制台，不同与sys.stderr("错误输出流")，sys.stderr是非缓冲的

#将本将要写入到文件中去的输出信息捕获到字符串中是有用的可以使用io.StringIO类实现
#改类提供的对象可以像文件对象一样使用，但其中以字符串来存放写入其中的任何数据
import io
sys.stdout=io.StringIO()#发送给sys.stdout的任意文本实际上的都将发送给io.StringIO，
#这是该行代码创建的一个类似于文件的对象
#可以使用下面的语句来恢复原始的sys.stdout
#sys.stdout=sys.__stdout__
print(io.StringIO.getvalue())#来获取所有写入到io.StringIO对象的字符串


#对csv2html2_opt.py文件进行重写，使用optparse模块来出来命令行参数
import optparse
def main():
    parser=optparse.OptionParser();
    parser.add_option("-w","--maxwidth",dest="maxwidth",type="int",help=("the maximum number of characters that can be output to string fields [default:%default]"))
    parser.add_option("-f","--format",dest="format",help=("the format used for outputting numbers [default:%default]"))
    parser.set_defaults(maxwidth=100,format=".0f")
    opts,args=parser.parse_args()
#对命令行进行分析后这些选项名可以通过dest名称进行访问

#calendar,datetime,与time模块
#datetime.datetime类型的对象通常是由程序创建的，存放UTC日期/时间的对象通常是从外部接收的
import calendar,datetime,time
moon_datetime_a=datetime.datetime(1969,7,20,20,17,40)           #存放的是Apollo 11 的登月时间
moon_time=calendar.timegm(moon_datetime_a.utctimetuple())       #存放的是int变量，其中存放的是登月时间至今经过的秒数，改数值是由calendar.timegm()函数提供，改函数接受由
                                                                #datetime.datetime.utctimetuple()函数返回的time_struct对象作为参数，并返回time_struct表示的秒数
moon_datetime_b=datetime.datetime.utcfromtimestamp(moon_time)   #对象类型是datetime.datetime是根据moon_time整数创建的，以便展示从秒数(自初始时间至今)到datetime.datetime对象的转换
moon_datetime_a.isoformat()                                     #下边这三行返回的是等价的但以ISO 8601格式表示的日期/时间字符串
moon_datetime_b.isoformat()
time.strftime("%Y-%m-%dT%H:%M:%S",time.gmtime())

#算法与组合数据类型
#heapq模块提供的函数可以将序列（列表）转换为堆
#collections包提供了字典collections.defaultdict与组合数据类型collections.named-tuple
#该包还提供了collections.UserList与collections.UserDict等数据类型
#另外一种类型是collections.deque改类型与list类似，特点是在列表的开始或结尾处添加或移除项有很快的速度

#collections.OrderedDict具有正常dicts相同的API，总是以插入顺序返回，而且popitem()方法总是返回最近被添加的项目
#Counter类是dict的一个子类，它提供了一种保持各种计数的便捷而且快速的方法。
#Python的非数值类型抽象基类，也在collections包中提供
#array模块提供了序列类型array.array。可以以非常节省空间的方式存储数单词或字符。与列表类型相似，区别是其中可存放的对象类型是固定的（在创建时就已经指定）
#weakref模块提供了创建弱引用的功能：如果对某个对象仅有的引用是弱引用，那么该对象仍然可以被调度进垃圾收集，这可以防止某些对象仅仅因为存在对其的引用而保持在内存中

#heaq模块
import heapq
heap=[]
heapq.heappush(heap,(5,"rest"))
heapq.heappush(heap,(2,"work"))
heapq.heappush(heap,(4,"study"))
#如果已有某个列表就可以使用heapq.heapify(alist)将其转换为堆，改函数可以自动完成必要的重新排序
#使用heapq.heappop(heap)可以从堆中移除最小项
for x in heapq.merge([1,3,5,8],[2,4,5],[0,1,6,8,9]):
    print(x,end=" ")#prints: 0 1 1 2 3 4 5 6 7 8 8 9


#heapq.merge()函数以任意数量的排序后iterables作为参数，并返回一个迭代子，该迭代子对iterables依序指定的所有项进行迭代


#5.2.8文件格式，编码与数据持久性
#base64模块提供的函数可以读写RFC 3548*中指定的Base16 Base32 与Base64等编码格式
#quopri模块提供的函数可以读写 “quoted-printable”格式，改格式在RFC 1521中定义,用于MIME(多用途Internet邮件扩展)数据
#uu模块提供的函数可以读写uuencoded数据。
#RFC1832定义了外部数据表示标准。xdrlib模块可以读写这种格式

#bz2模块可以处理.bz2文件
#gzip模块可以处理.gz文件
#tarfile模块可以处理.tar、  .tar.gz(.tgz) 与.tar.bz2.文件
#zipfile模块可以处理.zip文件

#对音频格式数据的处理功能
#aifc模块可以处理AIFF（音频交换文件格式）
#wave模块可以处理(未压缩的).wav文件
#有些音频数据格式可与audioop模块进行操纵
#sndhdr模块提供了两个函数，可用于确定文件中存放的是哪种类型的音频数据，以及某些特征，比如采样率

#RFC 822 中定义了一种配置文件(类似于老格式的Windows.ini文件)格式
#configparser模块提供了用于读写这种文件格式的函数

#pickle模块用于向磁盘中存储或从磁盘中取回任意的Python对象(包括整个组合)


#ase64模块
#在对以ASCII文本嵌入在电子邮件中的二进制数据进行处理时，base64模块的使用最为广泛
#改模块也可以用于将二进制数据存储到.py文件中
#第一步是将二进制数据转换为Base64格式
import base64
def read_binary_saveasbase64(left_align_png):
    binary=open(left_align_png,"rb").read()
    ascii_text=""
    for i,c in enumerate(base64.b64encode(binary)):
        if i and i%68==0:
            ascii_text+="\\\n"
        ascii_text+=chr(c)
    return ascii_text
#二进制数据可以使用open(filename,"wb").write(binary)写入到文件中。

#tarfile模块
BZ2_AVAILABLE=True
try:
    import bz2
except ImportError:
    BZ2_AVAILABLE=False
import string
UNTRUSTED_PREFIXES=tuple(["/","\\"]+[c+":" for c in string.ascii_letters])
#创建了元组('/','\','A:','B:',...,'Z:','a:','b:',...,'Z:','a:','b:',...'z:')
#UNTRUSTED_PREFIXES以这些开头的文件的路径可能是绝对路径，如果解压可能会重写系统文件，因此作为一种预警，对任何以这些字符为前缀的文件名，我们将不对其进行解压
import tarfile
import 
def untar(archive):
    tar=None
    try:
        tar=tarfile.open(archive)
        for member in tar.getmembers():#tarball中的每个文件称为一个成员，tarfile.getmembers()函数可以返回一个tarfile.TarInfo对象列表，其中每个代表一个成员，成员的函数名存储在tarfile.TarInfo.name属性中
                                       #如果名称益某个不可信的前缀开始，或其路径中包含... ，就输出一条错误消息；否则调用tarfile.extract()函数将成员保存到磁盘
            if member.name.startswith(UNTRUSTED_PREFIXES):
                print("untrusted prefix, ignoring",member.name)
            elif  ".." in member.name:
                print("suspect path, ignoring",member.name)
            else:
                tar.extract(member)
                print("unpacked",member.name)
    except (tarfile.TarError,EnvironmentError) as err:
        error(err)
    finally:
        if tar is not None:
            tar.close()

def error(message,exit_status=1):
    























