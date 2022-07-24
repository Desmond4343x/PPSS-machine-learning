import numpy as np
import sklearn
from sklearn.cluster import KMeans
import matplotlib as plt
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.metrics import ConfusionMatrixDisplay

np.random.seed(10)

import numpy as np
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.cluster import KMeans
import sklearn.metrics
from scipy.ndimage import gaussian_filter

"""Pseudocode. 
def x_extremeties
	returns file with dimensions (10000, 4, 1). [x_min, x_left_center, x_right_center, x_max] where 0 <= x_i <= 28
def cluster
	returns cluster score from KMeans and plots confusion matrix

def main
	load data with eventual filter and call every function"""


def x_values(training_data, label_data):
	outer_array = []
	picture_index = 0
	labels = []
	xmin = []
	xmax = []
	while picture_index <= (training_data.shape[0] - 1):
		sum = 0
		for y_index in range(training_data.shape[1]):
			for x_index in range(training_data.shape[2]):
				sum += training_data[picture_index][y_index][x_index]
		if sum <= 0:
			picture_index += 2
		else:
			nonzero_x_array = []
			inner_array = []  # [x_min, x_center_left, x_center_right, x_max]
			for y_index in range(training_data.shape[1]):
				for x_index in range(training_data.shape[2]):
					if training_data[picture_index][y_index][x_index] > 0:
						# print(picture_index)
						nonzero_x_array.append(x_index)

			left_of_center = [-5]
			right_of_center = [34]
			for elem in nonzero_x_array:
				if elem <= 13:
					left_of_center.append(elem)
				else:
					right_of_center.append(elem)

			inner_array.append(min(nonzero_x_array))
			inner_array.append(max(left_of_center))
			inner_array.append(min(right_of_center))
			inner_array.append(max(nonzero_x_array))

			xmin.append(min(nonzero_x_array))
			xmax.append(max(nonzero_x_array))
			# print(inner_array)

			outer_array.append(inner_array)
			labels.append(label_data[picture_index])
			picture_index += 2

	result_data = np.array(outer_array)
	result_label = np.array(labels)


	min_max = plt.figure(2)
	plt.scatter(xmin, xmax)
	plt.title('Min-Max')
	plt.ylabel('xmax')
	plt.xlabel('xmin')
	plt.show(block=False)

	return [result_data, result_label]


def cluster(numbers_used, data, labels):
	kmeans = KMeans(n_clusters=len(numbers_used), n_init=40).fit(data)
	y_pred_kmeans = kmeans.predict(data)
	# Scoring
	score = sklearn.metrics.rand_score(labels, y_pred_kmeans)
	print("score is: ", score)
	# Confusion matrix
	from sklearn.metrics import confusion_matrix

	cm = confusion_matrix(y_true=labels, y_pred=y_pred_kmeans)

	fig = plt.figure()
	import seaborn as sns;
	sns.set()
	ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", )
	plt.title('Confusion matrix')
	plt.xlabel('Predicted')
	plt.ylabel('True')
	plt.show()

	from scipy.optimize import linear_sum_assignment as linear_assignment

	def _make_cost_m(cm):
		s = np.max(cm)
		return (- cm + s)

	indexes = linear_assignment(_make_cost_m(cm))
	js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
	cm2 = cm[:, js]
	# sns.heatmap(cm2, annot=True, fmt="d", cmap="Blues")
	acc = np.trace(cm2) / np.sum(cm2)
	print("accuracy is: ", acc)
	return [score, acc]


def main():
	# Load data
	X_test = np.load('data/test_X.npy', mmap_mode='r')
	X_train = np.load('data/train_X.npy', mmap_mode='r')
	Y_test = np.load('data/test_Y.npy', mmap_mode='r')
	Y_train = np.load('data/train_Y.npy', mmap_mode='r')
	numbers_used = [2, 3]
	# print(Y_test)
	train_mask = np.isin(Y_train, numbers_used)
	test_mask = np.isin(Y_test, numbers_used)
	X_train, Y_train = X_train[train_mask], Y_train[train_mask]
	X_test, Y_test = X_test[test_mask], Y_test[test_mask]
	[clust_data, clust_labels] = x_values(X_train, Y_train)
	plt.show()
	cluster(numbers_used, clust_data, clust_labels)



if __name__ == "__main__":
	main()