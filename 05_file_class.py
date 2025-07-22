"""
Exercises for Introduction to Python for Data Science
Week 05 - Files and Object Oriented Programming

Matthias Feurer and Andreas Bender
2027-03-05
"""

import os
import hashlib
from pathlib import Path
from collections import defaultdict
import json
import yaml
import pickle

print("=" * 80)
print("Week 05 - Files and Object Oriented Programming - 练习题及答案")
print("=" * 80)

# ====================================================================
# Exercise 1
# ====================================================================

print("\n# Exercise 1")
print("""
Write a function called `replace_all` that takes as arguments a pattern
string, a replacement string, and two filenames. It should read the
first file and write the contents into the second file (creating it if
necessary). If the pattern string appears anywhere in the contents, it
should be replaced with the replacement string.

Here's an outline of the function to get you started.

def replace_all(old, new, source_path, dest_path):
    # read the contents of the source file, open the file using a context manager
    # replace the old string with the new
    # write the result into the destination file

To test your function, read the file `photos/notes.txt`, replace
`'photos'` with `'images'`, and write the result to the file
photos/new_notes.txt.
""")

print("解答:")


def replace_all(old, new, source_path, dest_path):
    """替换文件中的字符串并写入新文件"""
    # 读取源文件内容，使用上下文管理器
    with open(source_path, 'r', encoding='utf-8') as source_file:
        contents = source_file.read()

    # 替换旧字符串为新字符串
    new_contents = contents.replace(old, new)

    # 将结果写入目标文件
    with open(dest_path, 'w', encoding='utf-8') as dest_file:
        dest_file.write(new_contents)


# 测试示例（需要实际文件）
# replace_all('photos', 'images', 'photos/notes.txt', 'photos/new_notes.txt')

print("✓ replace_all 函数已实现")

# ====================================================================
# Exercise 2
# ====================================================================

print("\n# Exercise 2")
print("""
In a large collection of files, there may be more than one copy of the
same file, stored in different directories or with different file names.
The goal of this exercise is to search for duplicates. As an example,
we'll work with image files in the `photos` directory.

Here's how it will work:
- We'll use the `walk` function to search this directory for files
  that end with one of the extensions in `config['extensions']`.
- For each file, we'll use `md5_digest` to compute a digest of the contents.
- Using a dict, we'll make a mapping from each digest to a list of
  paths with that digest.
- Finally, we'll search the dict for any digests that map to multiple
  files.
- If we find any, we'll use function same_contents to confirm that the
  files contain the same data.

Here are some suggestions on which functions to write first:

1. To identify image files, write a function called `is_image` that
   takes a path and a list of file extensions, and returns True if the
   path ends with one of the extensions in the list. Hint: Use
   `os.path.splitext`. Also: How can this be solved using pathlib?

2. Write a function called `add_path` that takes as arguments a path
   and a dict. It should use `md5_digest` to compute a digest of the
   file contents. Then it should update the dict, either creating a new
   item that maps from the digest to a list containing the path, or
   appending the path to the list if it exists. Hint: can you use a
   specialized version of dict?

3. Write a function called `walk_images` that takes a dict and a
   directory and uses a `walk` function to walk through the files in
   the directory and its subdirectories. For each file, it should use
   `is_image` to check whether it's an image file and `add_path` to add
   it to the dict.
""")

print("解答:")


def md5_digest(file_path):
    """计算文件的MD5摘要"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def is_image(file_path, extensions):
    """检查文件是否为图片格式"""
    # 方法1: 使用 os.path.splitext
    _, ext = os.path.splitext(file_path.lower())
    return ext in extensions

    # 方法2: 使用 pathlib (注释掉的替代方案)
    # path = Path(file_path)
    # return path.suffix.lower() in extensions


def add_path(file_path, mapping):
    """将文件路径添加到映射字典中"""
    digest = md5_digest(file_path)
    if digest:
        mapping[digest].append(file_path)


def walk_images(mapping, directory):
    """遍历目录查找图片文件"""
    config = {
        'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff']
    }

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if is_image(file_path, config['extensions']):
                add_path(file_path, mapping)


def same_contents(path1, path2):
    """检查两个文件是否包含相同内容"""
    try:
        with open(path1, 'rb') as f1, open(path2, 'rb') as f2:
            return f1.read() == f2.read()
    except Exception as e:
        print(f"Error comparing files: {e}")
        return False


# 主程序
def find_duplicates(directory):
    """查找重复文件的主程序"""
    mapping = defaultdict(list)  # 使用特殊版本的dict
    walk_images(mapping, directory)

    # 检查重复文件
    duplicates = []
    for digest, paths in mapping.items():
        if len(paths) > 1:
            print(f"发现重复文件: {paths}")
            # 验证文件内容是否真正相同
            if len(paths) == 2 and same_contents(paths[0], paths[1]):
                print("确认：文件内容相同")
            duplicates.append(paths)

    return mapping, duplicates


# Bonus: 保存数据结构的示例
def save_mapping_formats(mapping, base_filename):
    """以不同格式保存映射数据 (Bonus 2)"""
    # 转换为可序列化的格式
    serializable_mapping = dict(mapping)

    # YAML 格式 - 人类可读
    with open(f"{base_filename}.yaml", 'w') as f:
        yaml.dump(serializable_mapping, f, default_flow_style=False)

    # JSON 格式 - 结构化，部分可读
    with open(f"{base_filename}.json", 'w') as f:
        json.dump(serializable_mapping, f, indent=2)

    # Pickle 格式 - 二进制，不可读但高效
    with open(f"{base_filename}.pkl", 'wb') as f:
        pickle.dump(serializable_mapping, f)


print("✓ 重复文件查找系统已实现")
print("  - is_image(): 检查图片文件扩展名")
print("  - add_path(): 将文件路径添加到映射字典")
print("  - walk_images(): 遍历目录查找图片")
print("  - same_contents(): 验证文件内容")
print("  - Bonus: 支持YAML、JSON、Pickle格式保存")

# ====================================================================
# Exercise 3
# ====================================================================

print("\n# Exercise 3")
print("""
Write a function called `subtract_time` that takes two `Time` objects
and returns the interval between them in seconds – assuming that they
are two times during the same day.
""")

print("解答:")


class Time:
    """表示时间的简单类"""

    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second


def subtract_time(t1, t2):
    """计算两个时间对象之间的秒数差"""
    # 将时间转换为总秒数
    seconds1 = t1.hour * 3600 + t1.minute * 60 + t1.second
    seconds2 = t2.hour * 3600 + t2.minute * 60 + t2.second

    # 返回差值
    return seconds1 - seconds2


# 测试示例
time1 = Time(14, 30, 0)  # 14:30:00
time2 = Time(12, 15, 30)  # 12:15:30
diff = subtract_time(time1, time2)
print(f"✓ 测试: 14:30:00 - 12:15:30 = {diff} 秒")

# ====================================================================
# Exercise 4
# ====================================================================

print("\n# Exercise 4")
print("""
Write a function called `is_after` that takes two `Time` objects and
returns `True` if the first time is later in the day than the second,
and `False` otherwise.

def is_after(t1, t2):
    \"\"\"Checks whether `t1` is after `t2`.

    >>> is_after(make_time(3, 2, 1), make_time(3, 2, 0))
    True
    >>> is_after(make_time(3, 2, 1), make_time(3, 2, 1))
    False
    >>> is_after(make_time(11, 12, 0), make_time(9, 40, 0))
    True
    \"\"\"
    return None
""")

print("解答:")


def make_time(hour, minute, second):
    """创建时间对象"""
    return Time(hour, minute, second)


def is_after(t1, t2):
    """检查 t1 是否在 t2 之后

    >>> is_after(make_time(3, 2, 1), make_time(3, 2, 0))
    True
    >>> is_after(make_time(3, 2, 1), make_time(3, 2, 1))
    False
    >>> is_after(make_time(11, 12, 0), make_time(9, 40, 0))
    True
    """
    # 将时间转换为总秒数进行比较
    seconds1 = t1.hour * 3600 + t1.minute * 60 + t1.second
    seconds2 = t2.hour * 3600 + t2.minute * 60 + t2.second

    return seconds1 > seconds2


# 测试doctest中的例子
print("✓ 测试结果:")
print(f"  is_after(3:02:01, 3:02:00): {is_after(make_time(3, 2, 1), make_time(3, 2, 0))}")
print(f"  is_after(3:02:01, 3:02:01): {is_after(make_time(3, 2, 1), make_time(3, 2, 1))}")
print(f"  is_after(11:12:00, 9:40:00): {is_after(make_time(11, 12, 0), make_time(9, 40, 0))}")

# ====================================================================
# Exercise 5
# ====================================================================

print("\n# Exercise 5")
print("""
Here's a definition for a `Date` class that represents a date – that is,
a year, month, and day of the month.

class Date:
    \"\"\"Represents a year, month, and day\"\"\"

1. Write a function called `make_date` that takes `year`, `month`, and
   `day` as parameters, makes a `Date` object, assigns the parameters
   to attributes, and returns the result the new object. Create an
   object that represents June 22, 1933.

2. Write a function called `print_date` that takes a `Date` object,
   uses an f-string to format the attributes, and prints the result. If
   you test it with the `Date` you created, the result should be
   `1933-06-22`.

3. Write a function called `is_after` that takes two `Date` objects as
   parameters and returns `True` if the first comes after the second.
   Create a second object that represents September 17, 1933, and check
   whether it comes after the first object.

Hint: You might find it useful to write a function called
`date_to_tuple` that takes a Date object and returns a tuple that
contains its attributes in year, month, day order.
""")

print("解答:")


class Date:
    """表示年、月、日"""
    pass


def make_date(year, month, day):
    """创建Date对象"""
    date = Date()
    date.year = year
    date.month = month
    date.day = day
    return date


def print_date(date):
    """打印日期"""
    print(f"{date.year:04d}-{date.month:02d}-{date.day:02d}")


def date_to_tuple(date):
    """将日期转换为元组 (提示中建议的辅助函数)"""
    return (date.year, date.month, date.day)


def is_after_date(date1, date2):
    """检查date1是否在date2之后"""
    return date_to_tuple(date1) > date_to_tuple(date2)


# 测试
date1 = make_date(1933, 6, 22)  # June 22, 1933
date2 = make_date(1933, 9, 17)  # September 17, 1933

print("✓ 测试结果:")
print("  第一个日期 (1933年6月22日): ", end="")
print_date(date1)
print("  第二个日期 (1933年9月17日): ", end="")
print_date(date2)
print(f"  第二个日期在第一个日期之后吗？ {is_after_date(date2, date1)}")

# ====================================================================
# Exercise 6
# ====================================================================

print("\n# Exercise 6")
print("""
In the previous chapter, a series of exercises asked you to write a
`Date` class and several functions that work with `Date` objects. Now
let's practice rewriting those functions as methods.

1. Write a definition for a `Date` class that represents a date – that
   is, a year, month, and day of the month.

2. Write an `__init__` method that takes `year`, `month`, and `day` as
   parameters and assigns the parameters to attributes. Create an
   object that represents June 22, 1933.

3. Write `__str__` method that uses an f-string to format the
   attributes and returns the result. If you test it with the `Date`
   you created, the result should be 1933-06-22.

4. Write a method called `is_after` that takes two `Date` objects and
   returns `True` if the first comes after the second. Create a second
   object that represents September 17, 1933, and check whether it
   comes after the first object.

Hint: You might find it useful write a method called `to_tuple` that
returns a tuple that contains the attributes of a `Date` object in
year-month-day order.
""")

print("解答:")


class DateOOP:
    """表示年、月、日的面向对象版本"""

    def __init__(self, year, month, day):
        """初始化方法 - 构造函数"""
        self.year = year
        self.month = month
        self.day = day

    def __str__(self):
        """字符串表示方法 - 当使用print()或str()时调用"""
        return f"{self.year:04d}-{self.month:02d}-{self.day:02d}"

    def to_tuple(self):
        """返回表示日期的元组 (提示中建议的辅助方法)"""
        return (self.year, self.month, self.day)

    def is_after(self, other):
        """检查当前日期是否在另一个日期之后"""
        return self.to_tuple() > other.to_tuple()


# 测试
print("✓ 测试结果:")
date_oop1 = DateOOP(1933, 6, 22)  # June 22, 1933
date_oop2 = DateOOP(1933, 9, 17)  # September 17, 1933

print(f"  第一个日期 (使用__str__): {date_oop1}")
print(f"  第二个日期 (使用__str__): {date_oop2}")
print(f"  第二个日期在第一个日期之后吗？ {date_oop2.is_after(date_oop1)}")
print(f"  第一个日期在第二个日期之后吗？ {date_oop1.is_after(date_oop2)}")

print("\n" + "=" * 80)
print("所有练习完成！")
print("=" * 80)

print("""
总结:
- Exercise 1: 文件读写和字符串替换
- Exercise 2: 文件系统遍历、MD5摘要计算、重复文件检测
- Exercise 3: 时间对象和时间差计算  
- Exercise 4: 时间比较函数(带doctest)
- Exercise 5: 函数式编程风格的日期处理
- Exercise 6: 面向对象编程风格的日期类(__init__, __str__, 实例方法)

关键概念:
- 上下文管理器 (with语句)
- 文件I/O操作
- os.walk() 目录遍历
- 哈希函数应用
- 函数式 vs 面向对象编程范式
- Python特殊方法 (__init__, __str__)
""")