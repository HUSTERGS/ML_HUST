from KNNClassifier import KNNClassifier
import matplotlib.pyplot  as plt
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784')
X = mnist.data
y = mnist.target
X_train, X_test, y_train, y_test = X[:6000], X[6000:7000], y[:6000], y[6000:7000]
data_test = []
data_train = []
for k in range(1, 21):
    # 分别测试k从1到20
    KNN = KNNClassifier(n_neighbors=k)
    KNN.fit(X_train, y_train)
    mis_test = KNN.score(X_test, y_test)
    mis_train = KNN.score(X_train, y_train)
    print(mis_test)
    print(mis_train)
    data_test.append(mis_test)
    data_train.append(mis_train)
# 进行绘制
fig, ax = plt.subplots()
ax.plot([x + 1 for x in range(20)], data_test, label='Test', marker='.', markersize=8)
ax.plot([x + 1 for x in range(20)], data_train, label='Train', marker='*', markersize=8)
ax.set_ylabel('misclassfication rate', fontsize='medium')
ax.set_xlabel('k', fontsize='medium')
plt.xticks([x for x in range(1, 21)], [x for x in range(1, 21)])
plt.legend()
plt.show()


KNN = KNNClassifier(n_neighbors=5, n_similar=10)
KNN.fit(X_train, y_train)
# 绘制距离最近的几个图片
KNN.draw_closest_with_data(X[10])
KNN.draw_closest_with_data(X[101])
KNN.draw_closest_with_data(X[4000])
KNN.draw_closest_with_data(X[202])
