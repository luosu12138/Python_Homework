import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import font_manager

my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")
df=pandas.read_csv("./expert.csv")

data=df.loc[:,['expertId','articles','age','ssq']]

'''data['ssq']=data['ssq']*10
print(data)'''

plt.figure(figsize=(10, 6))
#气泡图
plt.scatter(x=data['articles'],y=data['age'],s=data['ssq']*10,c='red',alpha=0.5)
plt.xlabel('发文量',fontproperties=my_font)
plt.ylabel('彩龄',fontproperties=my_font)
plt.title('发文量和彩龄对专家中将数的关系',fontproperties=my_font)
plt.savefig("./第四问-expert.png")