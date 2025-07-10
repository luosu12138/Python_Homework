import pandas
from matplotlib import pyplot as plt
from matplotlib import font_manager
#获取原数据
source=pandas.read_csv("./richers.csv")
#删除有空数据的样本
source=source.dropna(subset=['hometown'])

#获取‘省’数据
province_list=source['hometown'].str.split('-')
province=province_list.str[1]

#添加‘省’列
source['province']=province
#根据‘省’列进行分组，获得各省的财富值
data=source.groupby('province')['wealth'].sum()

my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")

plt.figure(figsize=(15, 10))
indexs=data.index
patches, texts, autotexts=plt.pie(x=data,labels=indexs,autopct=lambda p: '{:.0f}'.format(p) if p > 5 else '',textprops={'fontproperties': my_font})
# 调整标签字体大小
plt.setp(texts, size=10)
plt.setp(autotexts, size=15)
plt.title('财富额在各省的分布情况',fontproperties=my_font)
plt.savefig("./第3问.png")