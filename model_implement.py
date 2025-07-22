"""
经典机器学习模型从零实现
包含：线性回归、逻辑回归、K-均值聚类、朴素贝叶斯、K近邻
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

print("=" * 80)
print("经典机器学习模型从零实现")
print("=" * 80)


# =============================================================================
# 1. 逻辑回归 (Logistic Regression)
# =============================================================================

class LogisticRegression:
    """二分类逻辑回归，使用梯度下降优化"""

    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol

    def _sigmoid(self, z):
        """Sigmoid激活函数，防止数值溢出"""
        # 限制z的范围避免溢出
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """
        训练逻辑回归模型

        数学原理：
        - 假设函数: h(x) = sigmoid(θ^T x)
        - 损失函数: J(θ) = -(1/m) Σ[y*log(h) + (1-y)*log(1-h)]
        - 梯度: ∂J/∂θ = (1/m) X^T (h - y)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # 标准化特征
        self.x_mean_ = X.mean(axis=0, keepdims=True)
        self.x_std_ = X.std(axis=0, keepdims=True)
        self.x_std_[self.x_std_ == 0] = 1.0

        X_scaled = (X - self.x_mean_) / self.x_std_

        # 添加截距项
        X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

        # 初始化参数
        n_features = X_scaled.shape[1]
        self.weights_ = np.random.normal(0, 0.01, (n_features, 1))

        # 梯度下降
        m = X_scaled.shape[0]
        prev_cost = float('inf')

        for i in range(self.max_iter):
            # 前向传播
            z = X_scaled @ self.weights_
            h = self._sigmoid(z)

            # 计算代价函数
            cost = -(1 / m) * np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15))

            # 计算梯度
            gradient = (1 / m) * X_scaled.T @ (h - y)

            # 更新参数
            self.weights_ -= self.learning_rate * gradient

            # 检查收敛
            if abs(prev_cost - cost) < self.tol:
                print(f"逻辑回归在第{i + 1}次迭代收敛")
                break
            prev_cost = cost

        return self

    def predict_proba(self, X):
        """预测概率"""
        if not hasattr(self, 'weights_'):
            raise AttributeError("模型尚未训练")

        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.x_mean_) / self.x_std_
        X_scaled = np.hstack([np.ones((X_scaled.shape[0], 1)), X_scaled])

        z = X_scaled @ self.weights_
        return self._sigmoid(z).ravel()

    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return (probas >= 0.5).astype(int)


# =============================================================================
# 2. K-均值聚类 (K-Means Clustering)
# =============================================================================

class KMeans:
    """K-均值聚类算法"""

    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        """
        训练K-均值模型

        算法步骤：
        1. 随机初始化k个聚类中心
        2. 分配每个点到最近的聚类中心
        3. 更新聚类中心为分配点的均值
        4. 重复2-3直到收敛
        """
        if self.random_state:
            np.random.seed(self.random_state)

        X = np.asarray(X, dtype=float)
        n_samples, n_features = X.shape

        # 随机初始化聚类中心
        self.cluster_centers_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        for iteration in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))

            # 分配到最近的聚类中心
            labels = np.argmin(distances, axis=0)

            # 更新聚类中心
            new_centers = np.array([X[labels == k].mean(axis=0)
                                    for k in range(self.n_clusters)])

            # 检查收敛
            if np.allclose(self.cluster_centers_, new_centers, rtol=self.tol):
                print(f"K-均值在第{iteration + 1}次迭代收敛")
                break

            self.cluster_centers_ = new_centers

        self.labels_ = labels
        return self

    def predict(self, X):
        """预测聚类标签"""
        if not hasattr(self, 'cluster_centers_'):
            raise AttributeError("模型尚未训练")

        X = np.asarray(X, dtype=float)
        distances = np.sqrt(((X - self.cluster_centers_[:, np.newaxis]) ** 2).sum(axis=2))
        return np.argmin(distances, axis=0)


# =============================================================================
# 3. 朴素贝叶斯 (Naive Bayes) - 高斯版本
# =============================================================================

class GaussianNaiveBayes:
    """高斯朴素贝叶斯分类器"""

    def fit(self, X, y):
        """
        训练朴素贝叶斯模型

        数学原理：
        - 贝叶斯定理: P(y|X) = P(X|y) * P(y) / P(X)
        - 朴素假设: 特征条件独立
        - 高斯假设: P(xi|y) ~ N(μ, σ²)
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_features = X.shape[1]

        # 存储每个类别的统计信息
        self.class_priors_ = np.zeros(n_classes)
        self.feature_means_ = np.zeros((n_classes, n_features))
        self.feature_vars_ = np.zeros((n_classes, n_features))

        for idx, class_label in enumerate(self.classes_):
            # 该类别的数据
            X_class = X[y == class_label]

            # 先验概率
            self.class_priors_[idx] = len(X_class) / len(X)

            # 特征的均值和方差
            self.feature_means_[idx] = X_class.mean(axis=0)
            self.feature_vars_[idx] = X_class.var(axis=0) + 1e-9  # 添加小值避免除零

        return self

    def _gaussian_pdf(self, x, mean, var):
        """计算高斯概率密度"""
        coeff = 1 / np.sqrt(2 * np.pi * var)
        exponent = np.exp(-(x - mean) ** 2 / (2 * var))
        return coeff * exponent

    def predict_proba(self, X):
        """预测概率"""
        if not hasattr(self, 'classes_'):
            raise AttributeError("模型尚未训练")

        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # 计算后验概率
        posteriors = np.zeros((n_samples, n_classes))

        for idx in range(n_classes):
            # 先验概率
            prior = self.class_priors_[idx]

            # 似然概率（假设特征独立）
            likelihood = np.prod(
                self._gaussian_pdf(X, self.feature_means_[idx], self.feature_vars_[idx]),
                axis=1
            )

            # 后验概率（未归一化）
            posteriors[:, idx] = prior * likelihood

        # 归一化
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        return posteriors

    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return self.classes_[np.argmax(probas, axis=1)]


# =============================================================================
# 4. K近邻 (K-Nearest Neighbors)
# =============================================================================

class KNeighborsClassifier:
    """K近邻分类器"""

    def __init__(self, n_neighbors=5, metric='euclidean'):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        """
        训练K近邻模型（实际上只是存储数据）

        算法原理：
        - 懒惰学习：训练时不学习参数，只存储数据
        - 预测时找k个最近邻，投票决定类别
        """
        self.X_train_ = np.asarray(X, dtype=float)
        self.y_train_ = np.asarray(y, dtype=int)
        return self

    def _euclidean_distance(self, x1, x2):
        """计算欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))

    def predict(self, X):
        """预测类别"""
        if not hasattr(self, 'X_train_'):
            raise AttributeError("模型尚未训练")

        X = np.asarray(X, dtype=float)
        predictions = []

        for x in X:
            # 计算到所有训练样本的距离
            distances = self._euclidean_distance(self.X_train_, x)

            # 找到k个最近邻的索引
            k_nearest_indices = np.argsort(distances)[:self.n_neighbors]

            # 获取k个最近邻的标签
            k_nearest_labels = self.y_train_[k_nearest_indices]

            # 投票决定类别
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)


# =============================================================================
# 5. 线性回归（回顾，简化版）
# =============================================================================

class SimpleLinearRegression:
    """简化的线性回归实现"""

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        # 添加截距项
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # 正规方程解
        self.coef_ = np.linalg.inv(X.T @ X) @ X.T @ y
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return (X @ self.coef_).ravel()


# =============================================================================
# 测试所有模型
# =============================================================================

"""
经典模型测试练习 - 类似Exercise 2
为每个实现的模型生成相应的测试和验证
"""

"""
经典模型测试练习 - 类似Exercise 2
简洁的测试代码，直接使用已实现的模型
"""

import numpy as np

# 假设我们已经有了之前实现的模型类
# LinearRegression, LogisticRegression, KMeans, GaussianNaiveBayes, KNeighborsClassifier

print("=" * 80)
print("经典模型测试练习 - 简洁版")
print("=" * 80)

# =============================================================================
# Exercise 2.1 - 线性回归测试
# =============================================================================

print("\nExercise 2.1 - 线性回归测试")
print("=" * 50)

import numpy as np

rng = np.random.default_rng(42)

# --------------------------------------------------------------------------- #
# 1) 模拟数据
# --------------------------------------------------------------------------- #
n_samples = 1000
n_features = 3
true_beta = rng.normal(size=(n_features + 1, 1))  # 随机系数

X = rng.normal(size=(n_samples, n_features))  # 输入矩阵
noise = rng.normal(scale=0.05, size=(n_samples, 1))  # 噪声
y = true_beta[0] + X @ true_beta[1:] + noise  # 目标值

# --------------------------------------------------------------------------- #
# 2) 拟合模型
# --------------------------------------------------------------------------- #
model = LinearRegression().fit(X, y)

# --------------------------------------------------------------------------- #
# 3) 比较系数
# --------------------------------------------------------------------------- #
np.set_printoptions(precision=3, suppress=True)
print("True β*:      ", true_beta.ravel())
print("Estimated β̂: ", model.coef_.ravel())

# --------------------------------------------------------------------------- #
# 4) 计算RMSE
# --------------------------------------------------------------------------- #
pred = model.predict(X)
rmse = np.sqrt(((pred - y.ravel()) ** 2).mean())
print("RMSE:", rmse)

# =============================================================================
# Exercise 2.2 - 逻辑回归测试
# =============================================================================

print("\nExercise 2.2 - 逻辑回归测试")
print("=" * 50)

rng = np.random.default_rng(123)

# --------------------------------------------------------------------------- #
# 1) 模拟二分类数据
# --------------------------------------------------------------------------- #
n_samples = 1000
n_features = 2
true_weights = np.array([0.5, -1.2, 0.8])  # [截距, w1, w2]

X = rng.normal(size=(n_samples, n_features))
linear_comb = true_weights[0] + X @ true_weights[1:]
proba = 1 / (1 + np.exp(-linear_comb))
y = (proba > 0.5).astype(int)

# --------------------------------------------------------------------------- #
# 2) 拟合模型
# --------------------------------------------------------------------------- #
model = LogisticRegression(learning_rate=0.1, max_iter=1000).fit(X, y)

# --------------------------------------------------------------------------- #
# 3) 比较权重和性能
# --------------------------------------------------------------------------- #
pred = model.predict(X)
pred_proba = model.predict_proba(X)

print("True weights: ", true_weights)
print("Learned weights:", model.weights_.ravel())
print("Accuracy:     ", np.mean(pred == y))
print("Log Loss:     ", -np.mean(y * np.log(pred_proba + 1e-15) +
                                 (1 - y) * np.log(1 - pred_proba + 1e-15)))

# =============================================================================
# Exercise 2.3 - K-均值聚类测试
# =============================================================================

print("\nExercise 2.3 - K-均值聚类测试")
print("=" * 50)

rng = np.random.default_rng(456)

# --------------------------------------------------------------------------- #
# 1) 模拟聚类数据
# --------------------------------------------------------------------------- #
true_centers = np.array([[2, 2], [-2, -2], [2, -2]])
n_samples_per_cluster = 100

X_list = []
for center in true_centers:
    cluster_data = rng.normal(loc=center, scale=0.5, size=(n_samples_per_cluster, 2))
    X_list.append(cluster_data)

X = np.vstack(X_list)

# --------------------------------------------------------------------------- #
# 2) 拟合模型
# --------------------------------------------------------------------------- #
model = KMeans(n_clusters=3, random_state=42).fit(X)

# --------------------------------------------------------------------------- #
# 3) 比较聚类中心
# --------------------------------------------------------------------------- #
learned_centers = model.cluster_centers_

print("True centers:")
print(true_centers)
print("Learned centers:")
print(learned_centers)

# 计算WCSS (类内距离平方和)
wcss = 0
for k in range(3):
    cluster_points = X[model.labels_ == k]
    center = learned_centers[k]
    wcss += np.sum((cluster_points - center) ** 2)

print("WCSS:", wcss)

# =============================================================================
# Exercise 2.4 - 朴素贝叶斯测试
# =============================================================================

print("\nExercise 2.4 - 朴素贝叶斯测试")
print("=" * 50)

rng = np.random.default_rng(789)

# --------------------------------------------------------------------------- #
# 1) 模拟高斯分布数据
# --------------------------------------------------------------------------- #
n_samples = 600
n_features = 2

# 为每个类生成不同的均值和方差
class_0_mean = np.array([0, 0])
class_1_mean = np.array([2, 2])
class_2_mean = np.array([-1, 3])

X_0 = rng.normal(loc=class_0_mean, scale=0.8, size=(200, n_features))
X_1 = rng.normal(loc=class_1_mean, scale=1.0, size=(200, n_features))
X_2 = rng.normal(loc=class_2_mean, scale=0.6, size=(200, n_features))

X = np.vstack([X_0, X_1, X_2])
y = np.hstack([np.zeros(200), np.ones(200), np.full(200, 2)])

# --------------------------------------------------------------------------- #
# 2) 拟合模型
# --------------------------------------------------------------------------- #
model = GaussianNaiveBayes().fit(X, y)

# --------------------------------------------------------------------------- #
# 3) 评估性能
# --------------------------------------------------------------------------- #
pred = model.predict(X)
pred_proba = model.predict_proba(X)

print("True class means:")
print("Class 0:", class_0_mean)
print("Class 1:", class_1_mean)
print("Class 2:", class_2_mean)

print("Learned class means:")
print("Class 0:", model.feature_means_[0])
print("Class 1:", model.feature_means_[1])
print("Class 2:", model.feature_means_[2])

print("Accuracy:", np.mean(pred == y))
print("Class priors:", model.class_priors_)

# =============================================================================
# Exercise 2.5 - K近邻测试
# =============================================================================

print("\nExercise 2.5 - K近邻测试")
print("=" * 50)

rng = np.random.default_rng(101)

# --------------------------------------------------------------------------- #
# 1) 模拟非线性分类数据
# --------------------------------------------------------------------------- #
n_samples = 400
X = rng.uniform(-3, 3, size=(n_samples, 2))

# 创建环形决策边界
distances = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
y = ((distances > 1) & (distances < 2.5)).astype(int)

# --------------------------------------------------------------------------- #
# 2) 分割训练/测试集
# --------------------------------------------------------------------------- #
train_size = 300
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --------------------------------------------------------------------------- #
# 3) 测试不同的k值
# --------------------------------------------------------------------------- #
k_values = [1, 3, 5, 7, 9]

print("K-value | Train Acc | Test Acc")
print("--------|-----------|----------")

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"   {k}    |   {train_acc:.3f}   |  {test_acc:.3f}")

# =============================================================================
# Exercise 2.6 - 模型比较
# =============================================================================

print("\nExercise 2.6 - 模型比较")
print("=" * 50)

rng = np.random.default_rng(202)

# --------------------------------------------------------------------------- #
# 1) 创建统一的分类数据集
# --------------------------------------------------------------------------- #
n_samples = 1000
n_features = 4

X = rng.normal(size=(n_samples, n_features))
# 线性分类边界
linear_comb = 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.8 * X[:, 2] - 0.2 * X[:, 3]
y = (linear_comb > 0).astype(int)

# 训练/测试分割
split_idx = 800
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# --------------------------------------------------------------------------- #
# 2) 训练多个模型
# --------------------------------------------------------------------------- #
models = {
    'Logistic Regression': LogisticRegression(learning_rate=0.1, max_iter=1000),
    'Naive Bayes': GaussianNaiveBayes(),
    'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5)
}

print("Model               | Train Acc | Test Acc")
print("--------------------|-----------|----------")

for name, model in models.items():
    # 训练模型
    model.fit(X_train, y_train)

    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # 计算准确率
    train_acc = np.mean(train_pred == y_train)
    test_acc = np.mean(test_pred == y_test)

    print(f"{name:19s} |   {train_acc:.3f}   |  {test_acc:.3f}")

print("\n" + "=" * 80)
print("所有测试完成！")
print("=" * 80)