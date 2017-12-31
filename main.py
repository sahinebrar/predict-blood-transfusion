from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import pandas as pd
# fix random seed for reproducibility
np.random.seed(7)

# load data for flags
dataset = np.loadtxt("transfusion.data.txt", dtype='str',delimiter=",")

inputs = dataset[:,0:4]
output = dataset[:,4:5]
#print inputs

# create model
model = Sequential()
model.add(Dense(12, input_dim=4, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(inputs, output, epochs=100, batch_size=10)

# evaluate the model
scores = model.evaluate(inputs, output)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


