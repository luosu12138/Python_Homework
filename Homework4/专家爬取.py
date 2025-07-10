import requests
import re
import csv
import json
import time
'''////////////////////////////////////////////////////////////////////'''
#爬取一位专家的信息
def GetOneExpert(expertId):
    #每爬取一次延时一秒，防止因爬取频率过高而被拦截
    time.sleep(1)

    url=f'https://i.cmzj.net/expert/queryExpertById?expertId={expertId}'
    headers={
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36 Core/1.116.520.400 QQBrowser/19.2.6473.400',
        'Referer': 'http://www.cmzj.net/',
        'Authorization_code': '00b884ef7f7b4c4daff99dfd2fb2bdb4',
    }
    #爬取数据
    response = requests.get(url,headers=headers)

    #数据清洗
    content=response.text
    obj = re.compile(r'''"data":(?P<data>.*?),"code"''', re.S)
    result = obj.search(content).group('data')
    data=json.loads(result)

    #截取重要的数据
    con={}
    con['expertId']=data['expertId']
    con['name']=data['name']
    con['articles']=data['articles']
    con['age']=data['age']
    con['ssq']=data['ssqOne']+data['ssqTwo']+data['ssqThree']
    #部分专家没有大乐透的奖项
    if data['dltOne'] is not None:
        con['dlt']=data['dltOne']+data['dltTwo']+data['dltThree']
    else:
        con['dlt']=0
    print(con)
    return con
'''////////////////////////////////////////////////////////////////////'''
count=30
url=f'https://i.cmzj.net/expert/queryExpert?limit={count}&page=1&sort=0'
headers={
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36 Core/1.116.520.400 QQBrowser/19.2.6473.400',
    'Referer':'http://www.cmzj.net/',
    'Authorization_code':'00b884ef7f7b4c4daff99dfd2fb2bdb4',
}
response = requests.get(url,headers=headers)
content=response.text
obj=re.compile(r'''"data":(?P<data>.*?),"count"''',re.S)
result=obj.search(content).group('data')
data=json.loads(result)
expertIds=[]
for item in data:
    expertIds.append(item['expertId'])
expertCount=len(expertIds)

expertData=[]
for id in expertIds:
    expert=GetOneExpert(id)
    expertData.append(expert)
print(expertData)
columns=['expertId','name','articles','age','ssq','dlt']
with open('expert.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()  # 写入标题行
    writer.writerows(expertData)
