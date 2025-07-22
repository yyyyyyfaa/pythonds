"""
Exercises for Introduction to Python for Data Science
Week 08 - Pandas

Matthias Feurer and Andreas Bender
2026-11-06
"""

import pandas as pd
import numpy as np

print("="*80)
print("Week 08 - Pandas - 练习题及答案")
print("="*80)

# ====================================================================
# Exercise 1
# ====================================================================

print("\n# Exercise 1")
print("""
Create a pandas DataFrame that represents rapper information. The
DataFrame should contain the following data:

| Name           | Age | Genre   | Net_Worth_Millions | Albums_Released |
|----------------|-----|---------|--------------------|-----------------| 
| Drake          | 37  | Hip-Hop | 250                | 8               |
| Kendrick Lamar | 36  | Hip-Hop | 75                 | 5               |
| J. Cole        | 39  | Hip-Hop | 60                 | 7               |
| Travis Scott   | 32  | Trap    | 80                 | 4               |
| Post Malone    | 28  | Hip-Hop | 45                 | 5               |

You should create this DataFrame using a dictionary where the keys are
the column names and the values are lists containing the data for each
column.
""")

print("解答:")

# 创建rapper DataFrame
rapper_data = {
    'Name': ['Drake', 'Kendrick Lamar', 'J. Cole', 'Travis Scott', 'Post Malone'],
    'Age': [37, 36, 39, 32, 28],
    'Genre': ['Hip-Hop', 'Hip-Hop', 'Hip-Hop', 'Trap', 'Hip-Hop'],
    'Net_Worth_Millions': [250, 75, 60, 80, 45],
    'Albums_Released': [8, 5, 7, 4, 5]
}

rapper_df = pd.DataFrame(rapper_data)
print("✓ Rapper DataFrame创建完成:")
print(rapper_df)

# ====================================================================
# Exercise 2
# ====================================================================

print("\n# Exercise 2")
print("""
Using the rapper DataFrame from Exercise 1, perform the following
operations:

1. Select all rappers who are Hip-Hop artists
2. Find the average Net_Worth_Millions of all rappers
3. Select rappers who have released more than 5 albums
4. Sort the DataFrame by Net_Worth_Millions in descending order

Show the results of each operation.
""")

print("解答:")

# 1. 选择所有Hip-Hop艺术家
print("1. Hip-Hop艺术家:")
hiphop_rappers = rapper_df[rapper_df['Genre'] == 'Hip-Hop']
print(hiphop_rappers)

# 2. 计算平均净资产
print("\n2. 平均净资产:")
avg_net_worth = rapper_df['Net_Worth_Millions'].mean()
print(f"平均净资产: ${avg_net_worth:.2f} 百万")

# 3. 选择发行专辑超过5张的rapper
print("\n3. 发行专辑超过5张的rapper:")
prolific_rappers = rapper_df[rapper_df['Albums_Released'] > 5]
print(prolific_rappers)

# 4. 按净资产降序排列
print("\n4. 按净资产降序排列:")
sorted_by_wealth = rapper_df.sort_values('Net_Worth_Millions', ascending=False)
print(sorted_by_wealth)

# ====================================================================
# Exercise 3
# ====================================================================

print("\n# Exercise 3")
print("""
Create a pandas Series that represents daily temperatures for a week.
The Series should have:
- Index: Dates from January 1, 2023 to January 7, 2023
- Values: Temperatures [7, 6, 9, 10, 8, 7, 8] (in Celsius)

Then perform the following operations:
1. Calculate the mean temperature
2. Find the maximum and minimum temperatures
3. Calculate the temperature range (max - min)
4. Convert temperatures to Fahrenheit using the formula: (C * 9/5) + 32
""")

print("解答:")

# 创建日期索引
dates = pd.date_range(start='2023-01-01', end='2023-01-07', freq='D')
temperatures_celsius = [7, 6, 9, 10, 8, 7, 8]

# 创建温度Series
temp_series = pd.Series(temperatures_celsius, index=dates, name='Temperature_Celsius')
print("✓ 温度Series创建完成:")
print(temp_series)

# 1. 计算平均温度
print(f"\n1. 平均温度: {temp_series.mean():.2f}°C")

# 2. 最高和最低温度
print(f"2. 最高温度: {temp_series.max()}°C")
print(f"   最低温度: {temp_series.min()}°C")

# 3. 温度范围
temp_range = temp_series.max() - temp_series.min()
print(f"3. 温度范围: {temp_range}°C")

# 4. 转换为华氏温度
temp_fahrenheit = (temp_series * 9/5) + 32
print("4. 华氏温度:")
print(temp_fahrenheit)

# ====================================================================
# Exercise 4
# ====================================================================

print("\n# Exercise 4")
print("""
Create a DataFrame with missing values and demonstrate different methods
for handling them. Create the following DataFrame:

| City     | Jan_Temp | Feb_Temp | Mar_Temp |
|----------|----------|----------|----------|
| New York | 2        | NaN      | 8        |
| London   | 7        | 9        | NaN      |
| Berlin   | 3        | 5        | 10       |
| Paris    | 6        | 8        | 12       |

Then perform the following operations:
1. Check which values are missing using `isnull()`
2. Count the number of missing values in each column
3. Fill missing values with the mean of each column
4. Drop rows that contain any missing values
""")

print("解答:")

# 创建包含缺失值的DataFrame
city_temp_data = {
    'City': ['New York', 'London', 'Berlin', 'Paris'],
    'Jan_Temp': [2, 7, 3, 6],
    'Feb_Temp': [np.nan, 9, 5, 8],
    'Mar_Temp': [8, np.nan, 10, 12]
}

city_temp_df = pd.DataFrame(city_temp_data)
print("✓ 包含缺失值的DataFrame:")
print(city_temp_df)

# 1. 检查缺失值
print("\n1. 缺失值检查 (isnull()):")
print(city_temp_df.isnull())

# 2. 计算每列缺失值数量
print("\n2. 每列缺失值