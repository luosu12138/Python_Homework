import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import font_manager
#设置中文格式
my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")

#导入数据
df=pandas.read_csv("./richers.csv")

#获取行业类型总数
temp_type_columns=df['type'].str.split('、').tolist()
type_columns=list(set([j for i in temp_type_columns for j in i]))

#分析每个富豪对所处行业类型的贡献
type_zeros=pandas.DataFrame(numpy.zeros((df.shape[0],len(type_columns))),columns=type_columns)
for i in range(df.shape[0]):
    type_zeros.loc[i,temp_type_columns[i]]=df.loc[i,'wealth']

#统计各行业的经济值
type_wealth=type_zeros.sum(axis=0)
type_wealth=type_wealth.sort_values(ascending=False)
#获取行业经济值
type_wealth=type_wealth[:100]#由于数据过多，只选取前一百种行业类型
#获取行业类型名
type_index=type_wealth.index

#设置图片大小
plt.figure(figsize=(20, 6))
#作条形图
plt.bar(range(len(type_index)), type_wealth)
#设置x轴刻度
plt.xticks(range(len(type_index)), type_index, rotation=90,fontproperties=my_font)
#设置标题
plt.title('各行业的发展态势（前100）',fontproperties=my_font)
#设置y轴标签
plt.ylabel('财富值（单位/亿元）', fontproperties=my_font)
#保存图片
plt.savefig("./第2问.png")