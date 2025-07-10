
import requests
import re
import csv
import json
#设置富豪个数
count=1000

#爬取数据
url=f'https://www.hurun.net/zh-CN/Rank/HsRankDetailsList?num=ODBYW2BI&search=&offset=0&limit={count}'
response = requests.get(url)

#提取数据
content = response.text
obj=re.compile(r'"rows":(?P<data>.*?),"total"',re.S)
data=json.loads(obj.search(content).group('data'))

#选取数据摘要
richers=[]
for item in data:
    richer={}
    #姓名
    richer['name']=item['hs_Rank_Rich_ChaName_Cn'].split('、')[0]
    #财富值
    richer['wealth']=item['hs_Rank_Rich_Wealth_USD']
    #企业类型
    richer['type']=item['hs_Rank_Rich_Industry_Cn']
    #公司名
    richer['company']=item['hs_Rank_Rich_ComName_Cn']
    #公司所在地
    richer['workplace']=item['hs_Rank_Rich_ComHeadquarters_Cn']
    #if type(item['hs_Character'])
    #性别
    richer['gender']=item['hs_Character'][0]['hs_Character_Gender']
    #年龄
    richer['age'] = item['hs_Character'][0]['hs_Character_Age']
    #出生地
    richer['hometown']=item['hs_Character'][0]['hs_Character_BirthPlace_Cn']
    print(richer)
    richers.append(richer)

#保存数据
columns=['name','wealth','type','company','workplace','gender','age','hometown']
with open('richers.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()  # 写入标题行
    writer.writerows(richers)