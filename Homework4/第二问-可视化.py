import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import font_manager

my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")

#加载模型
df=pandas.read_csv("./data.csv")
#号码区字符串转换成号码列表
temp_front_list=df['frontWinningNum'].str.split(' ').tolist()
#统计出现过的号码
column_front_list=list(set([j for i in temp_front_list for j in i]))
#创建二维数组用于统计号码出现次数
zero_front_list=pandas.DataFrame(numpy.zeros((df.shape[0],len(column_front_list))),columns=column_front_list)
#分析各期的号码
for i in range(df.shape[0]):
    zero_front_list.loc[i,temp_front_list[i]]=1

#统计号码出现次数
front_frequent=zero_front_list.sum(axis=0)
front_frequent=front_frequent.astype('int')

#统计近期号码出现次数
front_recent_frequent=zero_front_list[:30].sum(axis=0)
front_recent_frequent=front_recent_frequent.astype('int')

plt.figure(figsize=(15, 6))
_x=range(len(column_front_list))
x_front_frequent=[i-0.15 for i in range(len(column_front_list))]
x_front_recent_requent=[i+0.15 for i in range(len(column_front_list))]
bar_front_frequent=plt.bar(x_front_frequent, front_frequent,label='前区总频率',width=0.3)
bar_front_recent_frequent=plt.bar(x_front_recent_requent, front_recent_frequent,label='前区近期频率',width=0.3)
# 在条形上添加数字标签
plt.bar_label(bar_front_frequent, labels=front_frequent, padding=3)  # padding 控制标签位置
plt.bar_label(bar_front_recent_frequent, labels=front_recent_frequent, padding=3)

plt.title('大乐透前区号码出现频率',fontproperties=my_font)
plt.xlabel('号码',fontproperties=my_font)
plt.ylabel('出现次数',fontproperties=my_font)
plt.xticks(_x,column_front_list,rotation=45)
plt.legend(prop=my_font)
plt.savefig('第二问-frontNum.png')
plt.close()

#根据最近高频号码预测
print(front_recent_frequent.sort_values(ascending=False).head(5))
'''/////////////////////////////////////////////////////////////////////////'''
temp_back_list=df['backWinningNum'].str.split(' ').tolist()
print(temp_back_list)
column_back_list=list(set([j for i in temp_back_list for j in i]))
print(column_back_list)
zero_back_list=pandas.DataFrame(numpy.zeros((df.shape[0],len(column_back_list))),columns=column_back_list)

for i in range(df.shape[0]):
    zero_back_list.loc[i,temp_back_list[i]]=1
print(zero_back_list)

back_frequent=zero_back_list.sum(axis=0)
back_frequent=back_frequent.astype('int')


back_recent_frequent=zero_back_list[:30].sum(axis=0)
back_recent_frequent=back_recent_frequent.astype('int')

print(back_recent_frequent)
print(type(back_recent_frequent))

plt.figure(figsize=(15, 6))
_x=range(len(column_back_list))
x_back_frequent=[i-0.15 for i in range(len(column_back_list))]
x_back_recent_requent=[i+0.15 for i in range(len(column_back_list))]

bar_back_frequent=plt.bar(x_back_frequent, back_frequent,label='后区总频率',width=0.3)
bar_back_recent_frequent=plt.bar(x_back_recent_requent, back_recent_frequent,label='后区近期频率',width=0.3)
# 在条形上添加数字标签
plt.bar_label(bar_back_frequent, labels=back_frequent, padding=3)  # padding 控制标签位置
plt.bar_label(bar_back_recent_frequent, labels=back_recent_frequent, padding=3)

plt.title('大乐透后区号码出现频率',fontproperties=my_font)
plt.xlabel('号码',fontproperties=my_font)
plt.ylabel('出现次数',fontproperties=my_font)
plt.xticks(_x,column_back_list,rotation=45)
plt.legend(prop=my_font)
plt.savefig('第二问-backNum.png')
plt.close()

#根据最近高频号码预测
print(back_recent_frequent.sort_values(ascending=False).head(2))