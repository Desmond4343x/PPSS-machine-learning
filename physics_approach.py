import numpy as np

"""Pseudocode. 
def x_extremeties
	returns file with dimensions (10000, 4, 1). [x_min, x_left_center, x_right_center, x_max] where 0 <= x_i <= 28

def cluster
	returns cluster score from KMeans and plots confusion matrix
	
def main
	load data with eventual filter and call every function"""








def main():
	# Load data
	X_test = np.load('/home/pontus/Desktop/DNN PPSS/PPSS-machine-learning/data/test_X.npy', mmap_mode='r')
	X_train = np.load('/home/pontus/Desktop/DNN PPSS/PPSS-machine-learning/data/train_X.npy', mmap_mode='r')
	Y_test = np.load('/home/pontus/Desktop/DNN PPSS/PPSS-machine-learning/data/test_Y.npy', mmap_mode='r')
	Y_train = np.load('/home/pontus/Desktop/DNN PPSS/PPSS-machine-learning/data/train_Y.npy', mmap_mode='r')
	numbers_used = [1, 2, 3, 4]
	# print(Y_test)
	train_mask = np.isin(Y_train, numbers_used)
	test_mask = np.isin(Y_test, numbers_used)
	X_train, Y_train = X_train[train_mask], Y_train[train_mask]
	X_test, Y_test = X_test[test_mask], Y_test[test_mask]

