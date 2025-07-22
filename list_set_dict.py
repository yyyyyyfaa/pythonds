# ====================================================================
# LIST 列表的经典方法
# ====================================================================

# 创建列表
my_list = [1, 2, 3, 4, 5]
print("原始列表:", my_list)

# 1. 添加元素
my_list.append(6)           # 在末尾添加单个元素
print("append(6):", my_list)

my_list.insert(0, 0)        # 在指定位置插入元素
print("insert(0, 0):", my_list)

my_list.extend([7, 8, 9])   # 扩展列表，添加多个元素
print("extend([7,8,9]):", my_list)

# 2. 删除元素
my_list.remove(0)           # 删除第一个匹配的元素
print("remove(0):", my_list)

popped = my_list.pop()      # 删除并返回最后一个元素
print("pop():", my_list, "删除的元素:", popped)

popped_index = my_list.pop(2)  # 删除并返回指定索引的元素
print("pop(2):", my_list, "删除的元素:", popped_index)

del my_list[0]              # 删除指定索引的元素
print("del my_list[0]:", my_list)

my_list.clear()             # 清空列表
print("clear():", my_list)

# 重新创建列表用于演示其他方法
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5]
print("\n新列表:", my_list)

# 3. 查找和计数
index_of_4 = my_list.index(4)      # 查找元素的索引
print("index(4):", index_of_4)

count_of_1 = my_list.count(1)      # 计算元素出现次数
print("count(1):", count_of_1)

# 4. 排序和反转
my_list.sort()                     # 原地排序
print("sort():", my_list)

my_list.reverse()                  # 原地反转
print("reverse():", my_list)

# 5. 复制
copied_list = my_list.copy()       # 浅拷贝
print("copy():", copied_list)

# 6. 列表推导式（不是方法，但很重要）
squares = [x**2 for x in range(5)]
print("列表推导式:", squares)

print("\n" + "="*60)

# ====================================================================
# SET 集合的经典方法
# ====================================================================

# 创建集合
my_set = {1, 2, 3, 4, 5}
other_set = {4, 5, 6, 7, 8}
print("原始集合 my_set:", my_set)
print("另一个集合 other_set:", other_set)

# 1. 添加元素
my_set.add(6)                      # 添加单个元素
print("add(6):", my_set)

my_set.update([7, 8, 9])           # 添加多个元素
print("update([7,8,9]):", my_set)

# 2. 删除元素
my_set.remove(9)                   # 删除指定元素（不存在会报错）
print("remove(9):", my_set)

my_set.discard(10)                 # 删除指定元素（不存在不报错）
print("discard(10):", my_set)

popped_item = my_set.pop()         # 随机删除并返回一个元素
print("pop():", my_set, "删除的元素:", popped_item)

# 3. 集合运算
print("\n集合运算:")
print("并集 union:", my_set.union(other_set))
print("交集 intersection:", my_set.intersection(other_set))
print("差集 difference:", my_set.difference(other_set))
print("对称差集 symmetric_difference:", my_set.symmetric_difference(other_set))

# 使用运算符
print("并集 |:", my_set | other_set)
print("交集 &:", my_set & other_set)
print("差集 -:", my_set - other_set)
print("对称差集 ^:", my_set ^ other_set)

# 4. 子集和超集判断
small_set = {1, 2}
print(f"\n{small_set} 是否为 {my_set} 的子集:", small_set.issubset(my_set))
print(f"{my_set} 是否为 {small_set} 的超集:", my_set.issuperset(small_set))
print(f"{my_set} 和 {other_set} 是否无交集:", my_set.isdisjoint(other_set))

# 5. 复制和清空
copied_set = my_set.copy()
print("copy():", copied_set)

my_set.clear()
print("clear():", my_set)

print("\n" + "="*60)

# ====================================================================
# DICT 字典的经典方法
# ====================================================================

# 创建字典
my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
print("原始字典:", my_dict)

# 1. 访问元素
print("get('a'):", my_dict.get('a'))           # 安全获取值
print("get('z', 0):", my_dict.get('z', 0))     # 不存在时返回默认值

# 2. 添加和更新
my_dict['e'] = 5                               # 添加新键值对
print("添加 'e': 5:", my_dict)

my_dict.update({'f': 6, 'g': 7})               # 更新多个键值对
print("update({'f': 6, 'g': 7}):", my_dict)

# 3. 删除元素
removed_value = my_dict.pop('g')               # 删除并返回指定键的值
print("pop('g'):", my_dict, "删除的值:", removed_value)

removed_item = my_dict.popitem()               # 删除并返回最后一个键值对
print("popitem():", my_dict, "删除的项:", removed_item)

# 4. 获取视图对象
print("\n字典视图:")
print("keys():", list(my_dict.keys()))         # 所有键
print("values():", list(my_dict.values()))     # 所有值
print("items():", list(my_dict.items()))       # 所有键值对

# 5. setdefault 方法
my_dict.setdefault('h', 8)                     # 如果键不存在则设置
print("setdefault('h', 8):", my_dict)

my_dict.setdefault('a', 100)                   # 如果键存在则不改变
print("setdefault('a', 100):", my_dict)

# 6. 复制和清空
copied_dict = my_dict.copy()
print("copy():", copied_dict)

# 7. 字典推导式（不是方法，但很重要）
squared_dict = {k: v**2 for k, v in my_dict.items() if isinstance(v, int)}
print("字典推导式:", squared_dict)

# 8. fromkeys 类方法
new_dict = dict.fromkeys(['x', 'y', 'z'], 0)
print("fromkeys(['x','y','z'], 0):", new_dict)

# 清空演示
my_dict.clear()
print("clear():", my_dict)

print("\n" + "="*60)

# ====================================================================
# 总结常用场景
# ====================================================================

print("常用场景总结:")
print("\nLIST - 有序、可重复、可变:")
print("- 适用于需要保持顺序的数据")
print("- 支持索引访问")
print("- 常用方法: append, insert, remove, pop, sort, reverse")

print("\nSET - 无序、不可重复、可变:")
print("- 适用于去重和集合运算")
print("- 查找速度快")
print("- 常用方法: add, remove, union, intersection, difference")

print("\nDICT - 键值对、无序（Python 3.7+保持插入顺序）、可变:")
print("- 适用于映射关系")
print("- 通过键快速访问值")
print("- 常用方法: get, update, pop, keys, values, items")