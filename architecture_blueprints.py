
#cosh
encoded = Dense(units=628, activation='relu')(input_img)
encoded = Dense(units=180, activation='relu')(encoded)
encoded = Dense(units=66, activation='relu')(encoded)
encoded = Dense(units=19, activation='relu')(encoded)
encoded = Dense(units=8, activation='relu')(encoded)
encoded = Dense(units=4, activation='linear')(encoded)

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
encoded = Dense(units=4, activation='linear')(encoded)

decoded = Dense(units=16, activation='relu')(encoded)
decoded = Dense(units=25, activation='relu')(decoded)
decoded = Dense(units=100, activation='relu')(decoded)
decoded = Dense(units=256, activation='relu')(decoded)
decoded = Dense(units=441, activation='relu')(decoded)

#fun one
encoded = Dense(units=8000, activation='relu')(input_img)
encoded = Dense(units=40, activation='relu')(encoded)
encoded = Dense(units=2, activation='linear')(encoded)

decoded = Dense(units=40, activation='relu')(encoded)
decoded = Dense(units=384, activation='relu')(decoded)

#fun two
encoded = Dense(units=400, activation='relu')(input_img)
encoded = Dense(units=40, activation='relu')(encoded)
encoded = Dense(units=2, activation='relu')(encoded)

decoded = Dense(units=40, activation='relu')(encoded)
decoded = Dense(units=384, activation='relu')(decoded)

#fun3
encoded = Dense(units=10000, activation='relu')(input_img)
encoded = Dense(units=2000, activation='relu')(encoded)
encoded = Dense(units=1000, activation='relu')(encoded)
encoded = Dense(units=500, activation='relu')(encoded)
encoded = Dense(units=100, activation='relu')(encoded)
encoded = Dense(units=3, activation='linear')(encoded)

decoded = Dense(units=100, activation='relu')(encoded)
decoded = Dense(units=200, activation='relu')(decoded)
decoded = Dense(units=300, activation='relu')(decoded)
decoded = Dense(units=628, activation='relu')(decoded)