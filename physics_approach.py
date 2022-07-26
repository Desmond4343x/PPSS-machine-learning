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


def x_values(data_arr, label_arr):
	outer_array = []
	labels = []
	xmin = []
	xmax = []
	for index in range(len(data_arr)):
		picture_index = 0

		training_data = data_arr[index]
		label_data = label_arr[index]
		while picture_index <= (training_data.shape[0] - 1):
			n_nonzero_pixels = 0
			sum = 0
			for y_index in range(training_data.shape[1]):
				for x_index in range(training_data.shape[2]):
					sum += training_data[picture_index][y_index][x_index]
			if sum <= 0:
				picture_index += 1
			else:
				nonzero_x_array = []
				inner_array = []  # [x_min, x_center_left, x_center_right, x_max, n_nonzero pixels]
				for y_index in range(training_data.shape[1]):
					for x_index in range(training_data.shape[2]):
						if training_data[picture_index][y_index][x_index] > 0:
							#print(picture_index)
							nonzero_x_array.append(x_index)
							n_nonzero_pixels += 1

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
				inner_array.append(n_nonzero_pixels)

				xmin.append(min(nonzero_x_array))
				xmax.append(max(nonzero_x_array))
				# print(inner_array)

				outer_array.append(inner_array)
				labels.append(label_data[picture_index])
				picture_index += 1

	result_data = np.array(outer_array)
	result_label = np.array(labels)

	plt.scatter(xmin, xmax)
	plt.title('Min-Max')
	plt.ylabel('xmax')
	plt.xlabel('xmin')
	plt.show(block=False)

	return [result_data, result_label]

#def file_merge(training_files, testing_files):



def cluster(classes_used, train_data, test_data, test_labels):
	kmeans = KMeans(n_clusters=classes_used, n_init=40).fit(train_data)
	y_pred_kmeans = kmeans.predict(test_data)
	# Scoring
	score = sklearn.metrics.rand_score(test_labels, y_pred_kmeans)
	print("score is: ", score)
	# Confusion matrix
	from sklearn.metrics import confusion_matrix

	cm = confusion_matrix(y_true=test_labels, y_pred=y_pred_kmeans)

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
	X_test_DD = np.load('current_phys_data/data_DD/test_X.npy', mmap_mode='r')
	X_train_DD = np.load('current_phys_data/data_DD/train_X.npy', mmap_mode='r')
	Y_test_DD = np.load('current_phys_data/data_DD/test_Y.npy', mmap_mode='r')
	Y_train_DD = np.load('current_phys_data/data_DD/train_Y.npy', mmap_mode='r')

	X_test_SD = np.load('current_phys_data/data_SD/test_X.npy', mmap_mode='r')
	X_train_SD = np.load('current_phys_data/data_SD/train_X.npy', mmap_mode='r')
	Y_test_SD = np.load('current_phys_data/data_SD/test_Y.npy', mmap_mode='r')
	Y_train_SD = np.load('current_phys_data/data_SD/train_Y.npy', mmap_mode='r')

	X_test_ND = np.load('current_phys_data/data_ND/test_X.npy', mmap_mode='r')
	X_train_ND = np.load('current_phys_data/data_ND/train_X.npy', mmap_mode='r')
	Y_test_ND = np.load('current_phys_data/data_ND/test_Y.npy', mmap_mode='r')
	Y_train_ND = np.load('current_phys_data/data_ND/train_Y.npy', mmap_mode='r')



	train_data_arr = [X_train_SD]
	test_data_arr = [X_test_SD]
	train_labels_arr = [Y_train_SD]
	test_labels_arr = [Y_test_SD]
	print(test_labels_arr, "test label arr")


	[clust_train_data, clust_train_labels] = x_values(train_data_arr, train_labels_arr)
	print(clust_train_labels.shape, "train labels shape")
	print(clust_train_data, "clust train data")
	print(clust_train_data.shape, "clustdata shape")
	plt.show(block=False)
	[clust_test_data, clust_test_labels] = x_values(test_data_arr, test_labels_arr)
	classes_used = 2
	cluster(classes_used, clust_train_data, clust_test_data, clust_test_labels)



if __name__ == "__main__":
	main()