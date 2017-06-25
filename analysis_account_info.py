'''
Created on 2017年4月18日

@author: weizhen
'''
import csv
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
import numpy as np
#load data file
def load_csv_file(filename):
    head_count=[]
    line_count=[]
    times_count=[]
    account_id=[]
    csv_reader=csv.reader(open(filename))
    print(type(csv_reader))
    for row in csv_reader:
        head_count.append(row[0])
        line_count.append(row[1])
        times_count.append(row[2])
        account_id.append(row[3])
    return np.array([head_count[1:],line_count[1:],times_count[1:]])
X=load_csv_file("account_info.csv").transpose()
X=X.astype("float32")
print(np.corrcoef(X,rowvar=0)*0.5+0.5)
#X=preprocessing.scale(X)
y_pred = KMeans(n_clusters=3).fit(X)
print(y_pred.labels_)
print(y_pred.cluster_centers_)
print(y_pred.inertia_)
ax = plt.figure().add_subplot(111, projection = '3d')
x1=X.transpose()[0]
y1=X.transpose()[1]
z1=X.transpose()[2]
ax.scatter(x1, y1,z1 , c=y_pred.labels_,s=20, marker = '*')
ax.set_xlabel('account_header_count')  
ax.set_ylabel('account_line_count')  
ax.set_zlabel('account_adjustment_count')
plt.show()
print(X)
temp=0
f3=open(u"highcharsresult1.txt","a")
f4=open(u"highcharsresult2.txt","a")
f5=open(u"highcharsresult3.txt","a")
for arr in X:
    if y_pred.labels_[temp]==0:
        f3.write("[")
        f3.write(str(arr[0]))
        f3.write(",")
        f3.write(str(arr[1]))
        f3.write(",")
        f3.write(str(arr[2]))
        f3.write("]")
        f3.write(",")
        f3.write("\r")
    if y_pred.labels_[temp]==1:
        f4.write("[")
        f4.write(str(arr[0]))
        f4.write(",")
        f4.write(str(arr[1]))
        f4.write(",")
        f4.write(str(arr[2]))
        f4.write("]")
        f4.write(",")
        f4.write("\r")
    if y_pred.labels_[temp]==2:
        f5.write("[")
        f5.write(str(arr[0]))
        f5.write(",")
        f5.write(str(arr[1]))
        f5.write(",")
        f5.write(str(arr[2]))
        f5.write("]")
        f5.write(",")
        f5.write("\r")
    temp=temp+1
#plt.figure(figsize=(12, 12))
#plt.scatter(X.transpose()[0], X.transpose()[1], c=y_pred.labels_)
#plt.title("sperate in five result")
#plt.show()