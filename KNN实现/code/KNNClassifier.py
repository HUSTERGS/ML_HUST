import matplotlib.pyplot as plt
import numpy as np
from heapq import nsmallest


class KNNClassifier:
    def __init__(self, n_neighbors=5, n_similar=5):
        self.n_neighbors = n_neighbors
        self.n_similar = min(n_similar, n_neighbors)  # 展示的最近的几张图片的数目,不能超过k
        self.n_similar_index = []

    def fit(self, X, y):
        # fit数据，其中X，y均为numpy数组，
        self._X = X.copy()
        self._y = y.copy()
        pass

    def score(self, X, y):
        # 输出误判率
        return sum(self.predict(X) != y) / y.size

    def predict(self, X):
        self.n_similar_index = []
        return np.array(list(map(lambda x: self._predict_one(x), X)))

    def _predict_one(self, x):
        k = self.n_neighbors
        # 只判断一个,x为728维的数组
        dist_list = list(np.sqrt(((self._X - x) ** 2).sum(axis=1)))
        # (距离，标签，index)三元组，其中index用于定位原来
        dist_list = list(zip(dist_list, self._y, list(x for x in range(0, len(self._X)))))
        # 暂时不考虑第k和第k+1个元素的距离相等的情况
        # 首先通过nsmallest函数获得最近的k个元素
        dist_list = nsmallest(k, dist_list)
        # 再通过np.unique函数返回前k个数据点中每一种标签的数目
        label, counts = np.unique([x[1] for x in dist_list[:k]], return_counts=True)
        lable_counts = list(zip(counts, label))
        # 找到数目最多的标签
        lable_counts.sort(reverse=True)
        result = lable_counts[0][1]
        while k > 1 and len(lable_counts) > 1 and lable_counts[0][0] == lable_counts[1][0]:
            # 当最大和第二大相等的时候，需要缩小k，重新进行计算
            if dist_list[k - 1] == lable_counts[0][1]:
                result = lable_counts[1][1]
                break
            elif dist_list[k - 1] == lable_counts[1][1]:
                result = lable_counts[0][1]
                break
            k -= 1
        self.n_similar_index = [x[2] for x in dist_list[:self.n_similar]]  # 选出最近的n的点的index
        return result

    def draw_closest_with_data(self, x):
        """
        x为数据
        """
        result = self._predict_one(x)
        self.plot_pixels(self.n_similar_index, x)

    def plot_pixels(self, indexes, x):
        """
        x 目标图片的点位数据
        indexes 为最近的n个图片的index
        """
        pixels = np.reshape(x, (28, 28))
        plt.figure(figsize=(2, 2))
        plt.axis('off')
        plt.title(f"Target Image")
        plt.imshow(pixels, cmap='gray')

        count = len(indexes)
        rows = int(count ** 0.5)
        columns = int(count / rows + 1)
        fig = plt.figure(figsize=(columns, rows))
        for i in range(count):
            pixels = np.reshape(self._X[[indexes[i]]], (28, 28))
            ax = fig.add_subplot(rows, columns, i + 1)
            ax.title.set_text(f"No.{indexes[i]}")
            plt.axis('off')
            plt.imshow(pixels, cmap='gray')
        fig.tight_layout()
        plt.show()
