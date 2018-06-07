from keras import models
from keras import layers
from keras import optimizers
import csv
import numpy as np
import matplotlib.pyplot as plt

file = open("ex1data1.csv")
fileObject = csv.reader(file)
m = sum(1 for row in fileObject)


file = open("ex1data1.csv")
fileObject=csv.reader(file)
line = next(fileObject)
n = len(line)-1

X = np.zeros((m,n))
Y = np.zeros((m,1))

file = open("ex1data1.csv")
fileObject = csv.reader(file)
index = 0
for filerow in fileObject:
    for i in range(n):
        X[index,i]=filerow[i]
    
    Y[index]=filerow[-1]
    index=index+1
    
plt.scatter(X,Y)
plt.xlabel("Population of a city")
plt.ylabel("Profit of a food truck in that city")
plt.show()


model = models.Sequential()
model.add(layers.Dense(1,input_shape=(X.shape[1],)))

optimizer = optimizers.SGD(lr=.001)

model.compile(loss='mean_squared_error',optimizer=optimizer)

history=model.fit(X,Y, epochs=1000,verbose=2)

history_dict = history.history
loss=history_dict['loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'b',label='Training Loss')

plt.title("Training loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()


predictions = model.predict(X)
print(predictions.shape)

plt.scatter(X,Y)
plt.plot(X,predictions,'r')
plt.show()

