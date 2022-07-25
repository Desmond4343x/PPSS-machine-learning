import numpy as np

X_test_DD = np.load('current_phys_data/data_DD/test_X.npy', mmap_mode='r')
X_train_DD = np.load('current_phys_data/data_DD/train_X.npy', mmap_mode='r')
Y_test_DD = np.load('current_phys_data/data_DD/test_Y.npy', mmap_mode='r')
Y_train_DD = np.load('current_phys_data/data_DD/train_Y.npy', mmap_mode='r')

X_test_ND = np.load('current_phys_data/data_ND/test_X.npy', mmap_mode='r')
X_train_ND = np.load('current_phys_data/data_ND/train_X.npy', mmap_mode='r')
Y_test_ND = np.load('current_phys_data/data_ND/test_Y.npy', mmap_mode='r')
Y_train_ND = np.load('current_phys_data/data_ND/train_Y.npy', mmap_mode='r')

X_test_SD = np.load('current_phys_data/data_SD/test_X.npy', mmap_mode='r')
X_train_SD = np.load('current_phys_data/data_SD/train_X.npy', mmap_mode='r')
Y_test_SD = np.load('current_phys_data/data_SD/test_Y.npy', mmap_mode='r')
Y_train_SD = np.load('current_phys_data/data_SD/train_Y.npy', mmap_mode='r')

print(X_train_SD.shape)
elem = [X_test_DD, X_test_SD]
print(X_train_SD)
print(np.array(elem).shape)

