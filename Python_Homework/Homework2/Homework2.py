import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
import re
from sklearn.linear_model import LinearRegression

plt.style.use('seaborn-v0_8')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 使用微软雅黑
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ====================================(1) 爬取数据（数据获取）========================================
def get_weather_data(year, month):
    url = f"http://www.tianqihoubao.com/lishi/dalian/month/{year}{month:02d}.html"
    
    headers = {
        "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36 Edg/138.0.0.0"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.encoding = "utf-8" 
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 解析表格数据
        table = soup.find('table', {'class': 'weather-table'})
        rows = table.find_all('tr')[1:]  # 跳过表头

        data = []
        for row in rows:
            if row.get_text(strip=True):
                cols = row.find_all('td')
                date = cols[0].text.strip()
                date_obj = datetime.strptime(date, "%Y年%m月%d日")
                new_date = date_obj.strftime("%Y-%m-%d")
                weather = cols[1].text.strip().split('/')  # 白天和夜晚天气
                temp = cols[2].text.strip().split('/')      # 最高最低温度
                wind = cols[3].text.strip().split('/')      # 白天夜晚风力
                
                data.append({
                    'date': new_date,
                    'day_weather': weather[0],
                    'night_weather': weather[1],
                    'max_temp': temp[0],
                    'min_temp': temp[1],
                    'day_wind': wind[0],
                    'night_wind': wind[1]
                })
        
        return data
    
    except Exception as e:
        print(f"获取{year}年{month}月数据失败: {e}")
        return []

def crawl_weather_data():
    year = 2022
    month = 1
    all_data = []
    
    while year <= 2024:
        print(f"正在获取{year}年{month}月数据...")
        monthly_data = get_weather_data(year, month)
        all_data.extend(monthly_data)
        
        # 随机延迟防止被封
        time.sleep(random.uniform(3, 5))
        
        if month == 12:
            year = year + 1
            month = 1
        else:
            month = month + 1
    
    # 保存数据
    pd.DataFrame(all_data).to_csv('dalian_weather_2022-2024.csv', index=False)

# ====================================(2) 绘制近三年月平均气温变化图 ========================================
def plot_monthly_temperature():
    df = pd.read_csv('dalian_weather_2022-2024.csv')
    # 提取年份和月份
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # 计算每月平均最高温和最低温
    df['max_temp'] = df['max_temp'].str.replace('℃', '').astype(float)
    df['min_temp'] = df['min_temp'].str.replace('℃', '').astype(float)


    monthly_avg = df.groupby(['year', 'month']).agg({
        'max_temp': 'mean',
        'min_temp': 'mean'
    }).reset_index()

    # 保存为.csv 文件
    pd.DataFrame(monthly_avg).to_csv('dalian_avgtemp_2022-2024.csv',index=False)

    # 计算三年同月的平均气温
    three_year_avg = monthly_avg.groupby('month').agg({
        'max_temp': 'mean',
        'min_temp': 'mean'
    }).reset_index()

     # 创建图表
    fig, ax = plt.subplots(figsize=(14, 7), dpi=100)
    
    # 绘制折线图
    sns.lineplot(data=three_year_avg, x='month', y='max_temp', 
                 color='#E74C3C', linewidth=3, marker='o', 
                 markersize=10, label='平均最高温度', ax=ax)
    
    sns.lineplot(data=three_year_avg, x='month', y='min_temp', 
                 color='#3498DB', linewidth=3, marker='s', 
                 markersize=10, label='平均最低温度', ax=ax)
    
    # 填充区域
    plt.fill_between(three_year_avg['month'], 
                     three_year_avg['min_temp'], 
                     three_year_avg['max_temp'], 
                     color='#AED6F1', alpha=0.3)
    
    # 设置标题和标签
    plt.title('大连近三年月平均气温变化趋势 (2022-2024)\n', 
              fontsize=18, pad=20, fontweight='bold')
    plt.xlabel('月份', fontsize=14, labelpad=10)
    plt.ylabel('温度 (℃)', fontsize=14, labelpad=10)
    
    # 设置x轴刻度
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                   '7月', '8月', '9月', '10月', '11月', '12月']
    plt.xticks(three_year_avg['month'], month_names, fontsize=12)
    
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    
    # 添加图例
    legend = plt.legend(fontsize=12, framealpha=1, 
                        shadow=True, borderpad=1, 
                        loc='upper right')
    legend.get_frame().set_facecolor('white')
    
    # 添加数据标签
    for i, row in three_year_avg.iterrows():
        ax.text(row['month'], row['max_temp']+0.8, 
                f"{row['max_temp']:.1f}℃", 
                ha='center', fontsize=10, color='#E74C3C',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
        ax.text(row['month'], row['min_temp']-0.8, 
                f"{row['min_temp']:.1f}℃", 
                ha='center', fontsize=10, color='#3498DB',
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # 添加边框美化
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.5)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('dalian_temperature_trend.png', dpi=300, bbox_inches='tight')
    
    # 显示图表
    plt.show()

# ====================================(3) 绘制近三年风力情况分布图  ========================================
# 获得所有实际出现的风力等级
def extract_all_wind_levels(wind_series):
    levels = set()
    for wind_str in wind_series:
        if pd.isna(wind_str):
            continue
        matches = re.findall(r'(\d+-\d+级)', wind_str)
        for match in matches:
            levels.add(match)
        
    return sorted(levels)

# 提取风力等级
def extract_wind_level(wind_str):
    if pd.isna(wind_str):
        pass
    match = re.search(r'(\d+-\d+级)', wind_str)
    return match.group(1)

# 创建颜色映射
def generate_colors(levels):
    base_colors = ['#9EDE73', '#FFE162', '#FF9F45', '#FF6363', '#A85CF9', '#6A0572', '#1A936F', '#114B5F']
    return {level: base_colors[i % len(base_colors)] for i, level in enumerate(levels)}


def plot_wind_distribution():
    df = pd.read_csv('dalian_weather_2022-2024.csv')

    # 提取年份和月份
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month

    # 获取所有实际出现的风力等级
    all_day_levels = extract_all_wind_levels(df['day_wind'])
    all_night_levels = extract_all_wind_levels(df['night_wind'])
    all_levels = sorted(set(all_day_levels + all_night_levels))

    print("所有出现的风力等级:", all_levels)

    # 创建扩展数据框
    day_data = df[['date', 'year', 'month']].copy()
    day_data['wind_level'] = df['day_wind'].apply(lambda x: re.search(r'(\d+-\d+级)', str(x)).group(1) if pd.notna(x) else None)
    
    night_data = df[['date', 'year', 'month']].copy()
    night_data['wind_level'] = df['night_wind'].apply(lambda x: re.search(r'(\d+-\d+级|\d+级)', str(x)).group(1) if pd.notna(x) else None)
    
    df_extended = pd.concat([day_data, night_data]).dropna()

    # 统计每月各风力等级天数
    result = []
    for (year, month), group in df_extended.groupby(['year', 'month']):
        for level in all_levels:
            count = (group['wind_level'] == level).sum()
            result.append({
                'year': year,
                'month': month,
                'wind_level': level,
                'days': count
            })
            
    wind_df = pd.DataFrame(result)

    # 统计各月份各风力等级的总天数
    monthly_total = wind_df.groupby(['month', 'wind_level'])['days'].sum().reset_index()

    # 创建图表
    plt.figure(figsize=(14, 7))

    # 准备x轴位置（12个月）
    months = range(1, 13)
    month_names = ['1月', '2月', '3月', '4月', '5月', '6月', 
                '7月', '8月', '9月', '10月', '11月', '12月']
    x = np.arange(len(months))
    width = 0.8 / len(all_levels)  # 根据等级数量调整宽度

    # 创建颜色映射
    colors = generate_colors(all_levels)

    # 绘制并列柱状图
    for i, level in enumerate(all_levels):
        level_data = monthly_total[monthly_total['wind_level'] == level].sort_values('month')
        
        days = np.zeros(len(months))
        for _, row in level_data.iterrows():
            days[row['month']-1] = row['days']
        
        x_pos = x + i * width
        
        # 绘制柱状图
        plt.bar(x_pos, days, width, 
                label=level,
                color=colors[level],
                edgecolor='white', linewidth=0.5)
        
        # 添加数据标签
        for j in range(len(days)):
            if days[j] > 0:
                plt.text(x_pos[j], days[j] + 0.1, f'{int(days[j])}', 
                        ha='center', va='bottom', fontsize=8)

    # 设置图表标题和标签
    plt.title('大连近三年风力情况分布（2022-2024）', fontsize=15, pad=15)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('总天数', fontsize=12)

    plt.xticks(x + width*(len(all_levels)-1)/2, month_names, fontsize=10)

    max_days = monthly_total['days'].max()
    plt.ylim(0, max_days + 3)

    plt.grid(axis='y', linestyle='--', alpha=0.6)

    # 添加图例
    plt.legend(title='风力等级', bbox_to_anchor=(1.05, 1), loc='upper left')

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('dalian_wind_total_days.png', dpi=300, bbox_inches='tight')

    plt.show()

# ====================================(4) 绘制近三年天气状况分布图   ========================================
def plot_monthly_weather_distribution():
    # 1. 读取和处理数据
    df = pd.read_csv("dalian_weather_2022-2024.csv")
    df['date'] = pd.to_datetime(df['date'])
    df['month'] = df['date'].dt.month
    df['quarter'] = pd.cut(df['month'], 
                         bins=[0,3,6,9,12],
                         labels=['第一季度', '第二季度', '第三季度', '第四季度'])

    # 2. 合并白天和夜间天气数据（保持不变）
    weather_data = pd.concat([
        df[['quarter', 'month', 'day_weather']].rename(columns={'day_weather': 'weather'}),
        df[['quarter', 'month', 'night_weather']].rename(columns={'night_weather': 'weather'})
    ])

    # 3. 获取所有天气类型（保持不变）
    all_weathers = weather_data['weather'].unique()

    # 4. 为每个季度创建独立图表
    for quarter in ['第一季度', '第二季度', '第三季度', '第四季度']:        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_axes([0.1, 0.2, 0.75, 0.6])
        
        # 筛选当前季度数据
        q_data = weather_data[weather_data['quarter'] == quarter]
        counts = q_data.groupby(['month', 'weather']).size().unstack(fill_value=0)
        counts = counts.reindex(range(1,13), fill_value=0)
        
        # 设置当前季度的月份范围
        months_in_q = range(1,4) if quarter == '第一季度' else \
                     range(4,7) if quarter == '第二季度' else \
                     range(7,10) if quarter == '第三季度' else range(10,13)
        
        n_weather = len(counts.columns)
        bar_width = 0.8 / n_weather
        x = np.arange(len(months_in_q))
        
        # 绘制柱状图
        for j, weather in enumerate(counts.columns):
            bars = ax.bar(
                x + j * bar_width,
                counts.loc[months_in_q, weather],
                width=bar_width,
                label=weather,
                color=plt.cm.Paired(j/len(all_weathers)),
                edgecolor='white',
                linewidth=0.5
            )
            
            # 添加数据标签
            for bar, month in zip(bars, months_in_q):
                height = counts.at[month, weather]
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width()/2,
                        height + 0.5,
                        f'{int(height)}',
                        ha='center',
                        va='bottom',
                        fontsize=10
                    )
        
        # 设置图表元素
        ax.set_title(f'大连近三年{quarter}天气分布（2022-2024）', fontsize=14, pad=12)
        ax.set_xlabel('月份', fontsize=12)
        ax.set_ylabel('出现天数', fontsize=12)
        ax.set_xticks(x + bar_width*(n_weather-1)/2)
        ax.set_xticklabels([f'{m}月' for m in months_in_q])
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        
        # 添加图例
        ax.legend(
            title='天气类型',
            bbox_to_anchor=(1.02, 1),
            loc='upper left',
            fontsize=10,
            title_fontsize=12
        )
        
        plt.subplots_adjust(
            top=0.85,    
            bottom=0.15, 
            left=0.1,   
            right=0.8   
        )
        
        # 保存当前季度的图表
        filename = f"{quarter}_weather_distribution.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()  

# ====================================(5) 气温预测及对比图绘制   ========================================
# 预测函数，使用线性回归模型
def predict_temperature(df, target_year, months_to_predict):
    predictions = []
    df = df[['year', 'month', 'max_temp']]
    for month in months_to_predict:
        # 提取该月的历史数据
        month_data = df[df['month'] == month]
        X = month_data['year'].values.reshape(-1, 1)
        y = month_data['max_temp'].values
        
        # 训练线性回归模型
        model = LinearRegression()
        model.fit(X, y)
        
        # 预测目标年份的温度
        pred = model.predict(np.array([[target_year]]))[0]
        predictions.append(pred)
    
    return predictions

def temperature_prediction():
    # 爬取2025年1-6月气温数据
    year = 2025
    month = 1
    data_2025 = []
    while month <= 6:
        print(f"正在获取{year}年{month}月数据...")
        monthly_data = get_weather_data(year, month)
        data_2025.extend(monthly_data)
        
        # 随机延迟防止被封
        time.sleep(random.uniform(3, 5))
        
        month = month + 1

    # 求取2025年1-6月的平均最高气温
    df_2025 = pd.DataFrame(data_2025)
    df_2025['date'] = pd.to_datetime(df_2025['date'])
    df_2025['month'] = df_2025['date'].dt.month

    df_2025['max_temp'] = df_2025['max_temp'].str.replace('℃', '').astype(float)

    monthly_avg_2025 = df_2025.groupby(['month']).agg({
        'max_temp': 'mean'
    }).reset_index()

    df = pd.read_csv('dalian_avgtemp_2022-2024.csv')

    # 预测2025年1-6月的最高气温
    months = range(1, 7)
    predicted_2025 = predict_temperature(df, 2025, months)

    # 绘制比较图
    plt.figure(figsize=(12, 7), dpi=100)
    ax = plt.subplot(111)

    # 设置背景色
    ax.set_facecolor('#f5f5f5')
    plt.gcf().set_facecolor('white')

    line_pred, = ax.plot(months, predicted_2025, 'o-', linewidth=3, markersize=10, 
                        label='预测值', color='#3498db')
    line_actual, = ax.plot(months, monthly_avg_2025['max_temp'], 's-', linewidth=3, markersize=10, 
                        label='实际值', color='#e74c3c')

    for i, (pred, actual) in enumerate(zip(predicted_2025, monthly_avg_2025['max_temp'])):
        ax.text(i+1, pred+0.5, f'{pred:.1f}°C', ha='center', va='bottom', fontsize=10, color='#3498db')
        ax.text(i+1, actual-0.8, f'{actual:.1f}°C', ha='center', va='top', fontsize=10, color='#e74c3c')

    plt.title('大连2025年1-6月平均最高气温预测 vs 实际值', fontsize=16, pad=20, fontweight='bold')
    plt.xlabel('月份', fontsize=14, labelpad=10)
    plt.ylabel('平均最高气温(℃)', fontsize=14, labelpad=10)

    plt.xticks(months, ['1月', '2月', '3月', '4月', '5月', '6月'], fontsize=12)
    plt.yticks(fontsize=12)

    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7, color='#aaaaaa')
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = ax.legend(loc='upper left', fontsize=12, framealpha=1, 
                    shadow=True, fancybox=True)
    legend.get_frame().set_facecolor('white')

    # 调整布局
    plt.tight_layout()
    # 保存为高清图片
    plt.savefig('dalian_temperature_prediction_2025.png', dpi=300, bbox_inches='tight', facecolor='white')
    # 显示图表
    plt.show()

def main():
    # 爬取数据
    print('数据爬取中：')
    crawl_weather_data()
    # 绘制近三年月平均气温变化图
    print('绘制近三年月平均气温变化图：')
    plot_monthly_temperature()
    # 绘制近三年风力情况分布图
    print('绘制近三年风力情况分布图图：')
    plot_wind_distribution() 
    # 绘制近三年天气状况分布图
    print('绘制近三年天气状况分布图')
    plot_monthly_weather_distribution()
    # 2025年气温预测及对比图绘制
    print('2025年气温预测及对比图绘制')
    temperature_prediction()


if __name__ == "__main__":
    main()