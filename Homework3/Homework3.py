import requests
from bs4 import BeautifulSoup
import time
from urllib.parse import urljoin
import pandas as pd
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from wordcloud import WordCloud
import nltk
from statsmodels.tsa.arima.model import ARIMA
import string
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import re

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ====================================(1) 爬取​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 论文信息 ========================================
def get_paper_data(conference, year):
    base_url = f"https://dblp.org/db/conf/{conference.lower()}/{conference.lower()}{year}.html"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    }
    
    try:
        time.sleep(2)
        
        print(f"正在爬取 {conference} {year} 论文数据...")
        response = requests.get(base_url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        paper = []    
        # 解析论文条目
        for entry in soup.find_all('li', class_='entry inproceedings'): 
            # 提取标题
            title = entry.find('span', class_='title').text.strip()
            
            # 提取作者列表
            authors = [a.text.strip() for a in entry.find_all('span', itemprop='author')]
            
            # 提取URL
            url_tag = entry.find('nav', class_='publ').find('a') if entry.find('nav', class_='publ') else None
            url = urljoin(base_url, url_tag['href']) if url_tag else ""
            
            paper.append({
                'title': title,
                'authors': ", ".join(authors), 
                'year': str(year),
                'conference': conference,
                'url': url
            })

        return paper    
    
    except requests.exceptions.RequestException as e:
        print(f"网络请求失败: {e}")
    except Exception as e:
        print(f"发生错误: {e}")

def crawl_weather_data(conference):
    year = 2020
    all_data = []
    
    while year <= 2025:
        paper_data = get_paper_data(conference, year)
        if paper_data is None:
            break
        all_data.extend(paper_data)
        year = year + 1
    
    # 保存数据
    pd.DataFrame(all_data).to_csv(f'{conference}_PaperData_2020-2025.csv', index=False)

# ====================================(2) 绘制​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 每届论文数量变化趋势图 ========================================
def plot_PaperCount():
    # 数据处理
    df_AAAI = pd.read_csv('AAAI_PaperData_2020-2025.csv')
    df_CVPR = pd.read_csv('CVPR_PaperData_2020-2025.csv')
    df_ICML = pd.read_csv('ICML_PaperData_2020-2025.csv')
    df_ICLR = pd.read_csv('ICLR_PaperData_2020-2025.csv')

    count_AAAI = []
    count_CVPR = []
    count_ICML = []
    count_ICLR = []

    for y in range(2020,2026):
        count_AAAI.append({
            'year': y,
            'count': (df_AAAI['year'] == y).sum()
        })

        count_ICLR.append({
            'year': y,
            'count': (df_ICLR['year'] == y).sum()
        })
    
    for y in range(2020,2025):
        count_CVPR.append({
            'year': y,
            'count': (df_CVPR['year'] == y).sum()
        })

        count_ICML.append({
            'year': y,
            'count': (df_ICML['year'] == y).sum()
        })
  
    df_AAAI_counts = pd.DataFrame(count_AAAI)
    df_CVPR_counts = pd.DataFrame(count_CVPR)
    df_ICML_counts = pd.DataFrame(count_ICML)
    df_ICLR_counts = pd.DataFrame(count_ICLR)

    # 绘制趋势图
    plt.figure(figsize=(15, 10))

    # AAAI 趋势图
    plt.subplot(2, 2, 1)
    aaai_plot = plt.plot(df_AAAI_counts['year'], df_AAAI_counts['count'], marker='o', color='blue')
    plt.title('AAAI 论文数量 (2020-2025)')
    plt.xlabel('年份')
    plt.ylabel('论文数量')
    plt.grid(True)
    # 添加数据标签
    for x, y in zip(df_AAAI_counts['year'], df_AAAI_counts['count']):
        plt.text(x, y+10, f'{y}', ha='center', va='bottom')

    # CVPR 趋势图
    plt.subplot(2, 2, 2)
    cvpr_plot = plt.plot(df_CVPR_counts['year'], df_CVPR_counts['count'], marker='o', color='red')
    plt.title('CVPR 论文数量 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('论文数量')
    plt.grid(True)
    # 添加数据标签
    plt.xticks(df_CVPR_counts['year'].astype(int)) 
    for x, y in zip(df_CVPR_counts['year'], df_CVPR_counts['count']):
        plt.text(x, y+10, f'{y}', ha='center', va='bottom')

    # ICML 趋势图
    plt.subplot(2, 2, 3)
    icml_plot = plt.plot(df_ICML_counts['year'], df_ICML_counts['count'], marker='o', color='green')
    plt.title('ICML 论文数量 (2020-2024)')
    plt.xlabel('年份')
    plt.ylabel('论文数量')
    plt.grid(True)
    # 添加数据标签
    plt.xticks(df_ICML_counts['year'].astype(int))
    for x, y in zip(df_ICML_counts['year'], df_ICML_counts['count']):
        plt.text(x, y+10, f'{y}', ha='center', va='bottom')

    # ICLR 趋势图
    plt.subplot(2, 2, 4)
    iclr_plot = plt.plot(df_ICLR_counts['year'], df_ICLR_counts['count'], marker='o', color='purple')
    plt.title('ICLR 论文数量 (2020-2025)')
    plt.xlabel('年份')
    plt.ylabel('论文数量')
    plt.grid(True)
    # 添加数据标签
    for x, y in zip(df_ICLR_counts['year'], df_ICLR_counts['count']):
        plt.text(x, y+10, f'{y}', ha='center', va='bottom')

    plt.tight_layout()
    # 保存图片
    plt.savefig('paper_count_trend.png', dpi=300, bbox_inches='tight')
    plt.show()

# ====================================(3) 绘制​ 2020-2025年 研究热点词云图 ========================================''''''
# 下载NLTK数据
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 文本预处理
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    
    # 转换为小写
    text = text.lower()
    # 移除标点符号
    text = text.translate(str.maketrans('', '', string.punctuation))
    # 使用正则表达式分词
    tokens = re.findall(r'\b[a-z]{3,}\b', text)  # 只保留3个字母以上的单词
    
    # 自定义停用词列表
    custom_stopwords = {'learning', 'based', 'using', 'approach', 'method', 
                       'problem', 'results', 'show', 'propose', 'paper',
                       'via', 'new', 'task', 'model', 'network', 'deep', 'models'}
    stop_words = set(ENGLISH_STOP_WORDS).union(custom_stopwords)
    
    # 过滤停用词
    tokens = [word for word in tokens if word not in stop_words]
    
    return tokens


def generate_wordclouds():
    # 读取数据
    df_AAAI = pd.read_csv('AAAI_PaperData_2020-2025.csv')
    df_CVPR = pd.read_csv('CVPR_PaperData_2020-2025.csv')
    df_ICML = pd.read_csv('ICML_PaperData_2020-2025.csv')
    df_ICLR = pd.read_csv('ICLR_PaperData_2020-2025.csv')

    # 合并所有会议数据
    df_all = pd.concat([df_AAAI, df_CVPR, df_ICML, df_ICLR], ignore_index=True)

    # 提取所有标题的关键词
    df_all['keywords'] = df_all['title'].apply(preprocess_text)

    # 按年份分组
    yearly_keywords = {}
    for year in range(2020, 2026):
        keywords = df_all[df_all['year'] == year]['keywords'].sum()
        yearly_keywords[year] = keywords

    # 统计每年的高频关键词
    yearly_top_keywords = {}
    for year, keywords in yearly_keywords.items():
        counter = Counter(keywords)
        yearly_top_keywords[year] = counter.most_common(50)  # 取前50个高频词

    # 生成并保存每年的词云图
    for year in range(2020, 2026):
        if year in df_all['year'].unique():
            # 获取该年份的所有关键词
            keywords = df_all[df_all['year'] == year]['keywords'].sum()
            word_freq = Counter(keywords)
            
            # 生成词云
            wordcloud = WordCloud(width=800, height=400,
                                background_color='white',
                                max_words=100).generate_from_frequencies(word_freq)
            
            # 创建子图
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title(f'{year} 研究热点', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(f'{year} 研究热点', dpi=300, bbox_inches='tight', facecolor='white')
            plt.show()
            plt.close()

# ====================================(4) 预测下一届会议的论文数量 ========================================
# 转换为Pandas Series(时间序列格式)
def prepare_data(data):
    df = pd.DataFrame(data)
    return df.set_index('year')['count']

# 使用ARIMA进行预测
def arima_predict(series, steps=1, order=(1,1,1)):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast

def paper_count_forecast():
    # 数据处理
    df_AAAI = pd.read_csv('AAAI_PaperData_2020-2025.csv')
    df_CVPR = pd.read_csv('CVPR_PaperData_2020-2025.csv')
    df_ICML = pd.read_csv('ICML_PaperData_2020-2025.csv')
    df_ICLR = pd.read_csv('ICLR_PaperData_2020-2025.csv')

    count_AAAI = []
    count_CVPR = []
    count_ICML = []
    count_ICLR = []

    for y in range(2020,2026):
        count_AAAI.append({
            'year': y,
            'count': (df_AAAI['year'] == y).sum()
        })

        count_ICLR.append({
            'year': y,
            'count': (df_ICLR['year'] == y).sum()
        })
    
    for y in range(2020,2025):
        count_CVPR.append({
            'year': y,
            'count': (df_CVPR['year'] == y).sum()
        })

        count_ICML.append({
            'year': y,
            'count': (df_ICML['year'] == y).sum()
        })

    aaai_series = prepare_data(count_AAAI)
    cvpr_series = prepare_data(count_CVPR)
    iclr_series = prepare_data(count_ICLR)
    icml_series = prepare_data(count_ICML)

    # ARIMA预测
    forecast_aaai = arima_predict(aaai_series)
    forecast_cvpr = arima_predict(cvpr_series)
    forecast_iclr = arima_predict(iclr_series)
    forecast_icml = arima_predict(icml_series)

    predicted_count_aaai = int(round(forecast_aaai.iloc[0]))
    predicted_count_cvpr = int(round(forecast_cvpr.iloc[0]))
    predicted_count_iclr = int(round(forecast_iclr.iloc[0]))
    predicted_count_icml = int(round(forecast_icml.iloc[0]))

    # 打印结果
    print(f"AAAI 2026年预测论文数量: {predicted_count_aaai}")
    print(f"CVPR 2026年预测论文数量: {predicted_count_cvpr}")
    print(f"ICML 2026年预测论文数量: {predicted_count_icml}")
    print(f"ICLR 2026年预测论文数量: {predicted_count_iclr}")

# 测试用例
if __name__ == "__main__":  
    # 爬取​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 论文信息
    print('爬取​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 论文信息:')
    crawl_weather_data('AAAI')
    crawl_weather_data('CVPR')
    crawl_weather_data('ICML')
    crawl_weather_data('ICLR')
    # 绘制​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 每届论文数量变化趋势图
    print('绘制​ ​AAAI、CVPR、ICML、ICLR​​ 2020-2025 每届论文数量变化趋势图:')
    plot_PaperCount()
    # 绘制​ 2020-2025年 研究热点词云图
    print('绘制​ 2020-2025年 研究热点词云图')
    generate_wordclouds()
    # 预测下一届会议的论文数量
    print('预测下一届会议的论文数量')
    paper_count_forecast()