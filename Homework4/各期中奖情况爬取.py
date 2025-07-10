import requests
import re
import csv
import json
#设置期数
count=100

#用于暂存数据
data=[]

#一页显示30期，计算总共有多少页
page=count//30+1

#设置爬虫时所需的各种参数
header = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.5845.97 Safari/537.36 Core/1.116.508.400 QQBrowser/19.1.6429.400',
    'Referer': 'https://www.zhcw.com/',

    'Cookie': '6333762c95037d16=C1nj9UJXariX2WSqQKQKqEoMCAPah9txJmPcRuc6HN97I%2B4N1NP8DbJ4cs1PS8MTESlN3x%2FHYRgc4Xso2J8Q9YtAZVWR5DL5u%2BsgvXc4sl55%2FK%2FThK9DfL7EyuMyZdKCxPsP5bcZxPsYdG6MpnzMS%2BkgilOal7c%2B4ACVhKclls9gDp52JBwZrOVTDaVFIrATmwUvt%2FfdBeMXDkfmRuzZ2h1nZPSMquiN9%2BceDKTQD%2Bna%2BbSsUIsj5P0WtgsiYtgo2UtPLUwNE3sWDm18h0dxPqJLJHjx6934YvfQ3yX8j2XAA04JO4pwQg%3D%3D; _TDID_CK=1750507299284; ll="118124"; bid=C_28miqXSuY; _pk_id.100001.4cf6=052c9ac4be8f8721.1750408908.; _vwo_uuid_v2=DA2FFD3CD5F1170312241AA01BED1E153|a9384a9724ad87ce292235dd0a0acc5f; __yadk_uid=BDOaXPp6r9SMeMqVuF6kmzlFtksZ6DoU; __utmz=30149280.1750409045.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmz=223695111.1750409045.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); ct=y; __utmc=30149280; __utmc=223695111; ap_v=0,6.0; __utma=30149280.2071020673.1750409045.1750485357.1750503866.4; __utma=223695111.318573330.1750409045.1750485357.1750503867.4; dbcl2="289485326:mUSxHqeAznI"; ck=Xrkh; _pk_ref.100001.4cf6=%5B%22%22%2C%22%22%2C1750507298%2C%22https%3A%2F%2Faccounts.douban.com%2F%22%5D; _pk_ses.100001.4cf6=1; frodotk_db="82baa1c011548cdf882fe54033fda69d"; push_noty_num=0; push_doumail_num=0'
}

#按页爬取各期数据
for pageNum in range(1,page+1):
    #爬取数据
    url=f'https://jc.zhcw.com/port/client_json.php?callback=jQuery112200843625236438994_1751272753898&transactionType=10001001&lotteryId=281&issueCount={count}&startIssue=&endIssue=&startDate=&endDate=&type=0&pageNum={pageNum}&pageSize=30&tt=0.0769601146386103&_=1751272753899'
    response = requests.get(url,headers=header)

    #数据清洗
    content=response.text
    obj=re.compile(r'''"data":(?P<data>.*?),"resCode"''',re.S)
    result=obj.search(content)
    #暂存
    data+=json.loads(result.group('data'))

#获取数据标签
columns=[]
for item in data[0].keys():
        columns.append(item)

#删除不需要的数据
columns.pop(len(columns)-1)
for item in data:
    item.pop('winnerDetails')

#存储数据
with open('data.csv', 'w', encoding='utf-8', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writeheader()  # 写入标题行
    writer.writerows(data)