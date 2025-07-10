import pandas
from matplotlib import pyplot as plt
from matplotlib import font_manager
from statsmodels.tsa.arima.model import ARIMA
#加载数据
df=pandas.read_csv("./data.csv")
#排序
df.sort_values('issue',ascending=True,inplace=True)
#提取所要数据
x=df['openTime']
y=df['saleMoney']

'''data=pandas.DataFrame({'openTime':x,'saleMoney':y})
print(data)
print(type(data))'''
#创建模型，喂数据
model = ARIMA(df['saleMoney'], order=(5,1,0))
#训练模型
model_fit = model.fit()
#预测下一个y值
forecast = model_fit.forecast(steps=1)
print(forecast)

my_font=font_manager.FontProperties(fname="C:\Windows\Fonts\simsun.ttc")
plt.figure(figsize=(30,8),dpi=80)
plt.plot(x,y)
plt.xticks(rotation=90)
plt.grid(True)
plt.xlabel('time')
plt.ylabel('saleMoney单位：亿元',fontproperties=my_font)
plt.title('大乐透总销售额随开奖日期的变化趋势',fontproperties=my_font)
plt.savefig('第一问.png')
plt.close()
#plt.show()