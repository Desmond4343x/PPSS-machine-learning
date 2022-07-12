
#cosh
encoded = Dense(units=628, activation='relu')(input_img)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=66, activation='relu')(encoded)
encoded = Dense(units=19, activation='relu')(encoded)
encoded = Dense(units=8, activation='relu')(encoded)
encoded = Dense(units=4, activation='relu')(encoded)

decoded = Dense(units=8, activation='relu')(encoded)
decoded = Dense(units=19, activation='relu')(decoded)
decoded = Dense(units=66, activation='relu')(decoded)
decoded = Dense(units=180, activation='relu')(decoded)
decoded = Dense(units=628, activation='relu')(decoded)


#150log
encoded = Dense(units=708, activation='relu')(input_img)
encoded = Dense(units=663, activation='relu')(encoded)
encoded = Dense(units=416, activation='relu')(encoded)
encoded = Dense(units=25, activation='relu')(encoded)
encoded = Dense(units=8, activation='relu')(encoded)
encoded = Dense(units=4, activation='relu')(encoded)

decoded = Dense(units=8, activation='relu')(encoded)
decoded = Dense(units=25, activation='relu')(decoded)
decoded = Dense(units=416, activation='relu')(decoded)
decoded = Dense(units=663, activation='relu')(decoded)
decoded = Dense(units=708, activation='relu')(decoded)

#x²
encoded = Dense(units=441, activation='relu')(input_img)
encoded = Dense(units=256, activation='relu')(encoded)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=25, activation='relu')(encoded)
encoded = Dense(units=16, activation='relu')(encoded)
encoded = Dense(units=4, activation='relu')(encoded)

decoded = Dense(units=16, activation='relu')(encoded)
decoded = Dense(units=25, activation='relu')(decoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=256, activation='relu')(decoded)
decoded = Dense(units=441, activation='relu')(decoded)