import pandas
import numpy
from matplotlib import pyplot as plt
from matplotlib import font_manager

my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")
#加载数据
df=pandas.read_csv("./data.csv")
#截取所需数据
df_sale=df.loc[:,['saleMoney','week']]

#截取各星期种类的数据
mon_sale=df_sale.loc[df_sale['week']=='星期一'].loc[:,'saleMoney']
wed_sale=df_sale.loc[df_sale['week']=='星期三'].loc[:,'saleMoney']
sat_sale=df_sale.loc[df_sale['week']=='星期六'].loc[:,'saleMoney']



plt.figure(figsize=(15, 6))
plt.boxplot(mon_sale,positions=[1],showmeans=True,label='Monday')
plt.boxplot(wed_sale,positions=[2],showmeans=True,label='Wednesday')
plt.boxplot(sat_sale,positions=[3],showmeans=True,label='Saturday')
plt.xlabel('week')
plt.ylabel('saleMoney单位/亿元',fontproperties=my_font)
plt.xticks(range(1,4),['Monday','Wednesday','Saturday'])
plt.legend(loc='best')
plt.savefig("./第三问-week_sale.png")
plt.close()
'''//////////////////////////////////////////////////////////////////'''
temp_front_list=df['frontWinningNum'].str.split(' ').tolist()
column_front_list=list(set([j for i in temp_front_list for j in i]))
'''//////////////////////////////////////////////////////////////////'''
df_mon=df[df['week']=='星期一']

temp_front_mon=df_mon['frontWinningNum'].str.split(' ').tolist()

zero_front_mon=pandas.DataFrame(numpy.zeros((df_mon.shape[0],len(column_front_list))),columns=column_front_list)

for i in range(df_mon.shape[0]):
    zero_front_mon.loc[i,temp_front_mon[i]]=1


mon_frequent=zero_front_mon.sum(axis=0)
mon_frequent=mon_frequent.astype('int')

'''//////////////////////////////////////////////////////////////////'''
df_wed=df[df['week']=='星期三']

temp_front_wed=df_wed['frontWinningNum'].str.split(' ').tolist()

zero_front_wed=pandas.DataFrame(numpy.zeros((df_wed.shape[0],len(column_front_list))),columns=column_front_list)

for i in range(df_wed.shape[0]):
    zero_front_wed.loc[i,temp_front_wed[i]]=1


wed_frequent=zero_front_wed.sum(axis=0)
wed_frequent=wed_frequent.astype('int')

'''//////////////////////////////////////////////////////////////////'''
df_sat=df[df['week']=='星期六']

temp_front_sat=df_sat['frontWinningNum'].str.split(' ').tolist()

zero_front_sat=pandas.DataFrame(numpy.zeros((df_sat.shape[0],len(column_front_list))),columns=column_front_list)

for i in range(df_sat.shape[0]):
    zero_front_sat.loc[i,temp_front_sat[i]]=1


sat_frequent=zero_front_sat.sum(axis=0)
sat_frequent=sat_frequent.astype('int')

'''//////////////////////////////////////////////////////////////////'''
num_data=pandas.DataFrame([mon_frequent,wed_frequent,sat_frequent],index=['Monday','Wednesday','Saturday'])
print(num_data)
plt.figure(figsize=(15, 6))
plt.imshow(num_data,cmap='hot')
plt.xticks(range(len(column_front_list)),column_front_list)
plt.yticks(range(3),num_data.index.tolist())
plt.colorbar()
plt.savefig("./第三问-week_num.png")
