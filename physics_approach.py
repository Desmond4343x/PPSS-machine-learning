import numpy as np

"""Pseudocode. 
def x_extremeties
	returns file with dimensions (10000, 4, 1). [x_min, x_left_center, x_right_center, x_max] where 0 <= x_i <= 28

def cluster
	returns cluster score from KMeans and plots confusion matrix
	
def main
	load data with eventual filter and call every function"""


def x_values(training_data):
	outer_array = []
	picture_index = 0

	while picture_index <= (training_data.shape[0]-1):
		sum = 0
		for y_index in range(training_data.shape[1]):
			for x_index in range(training_data.shape[2]):
				sum =+ training_data[picture_index][y_index][x_index]
		if sum <= 0:
			picture_index += 2
		nonzero_x_array = []
		inner_array = [] #[x_min, x_center_left, x_center_right]
		for y_index in range(training_data.shape[1]):
			for x_index in range(training_data.shape[2]):
				if training_data[picture_index][y_index][x_index] > 0:
					print(picture_index)
					nonzero_x_array.append(x_index)


		left_of_center = []
		right_of_center = []
		for elem in nonzero_x_array:
			if elem <= 13:
				left_of_center.append(elem)
			else:
				right_of_center.append(elem)

		inner_array.append(min(nonzero_x_array))
		inner_array.append(max(left_of_center))
		inner_array.append(min(right_of_center))
		inner_array.append(max(nonzero_x_array))

		outer_array.append(inner_array)
		picture_index += 2





	new_arr = np.array(outer_array)
	print(new_arr)
	print(new_arr.shape)









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
	x_values(X_train)



if __name__ == "__main__":
	main()