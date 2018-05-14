import numpy as np
from keras.datasets import mnist


from keras import models
from keras import layers
from keras import optimizers

(x_train_orig,y_train),(x_test_orig,y_test) = mnist.load_data()

#print(x_test_orig.shape)


x_train = np.reshape(x_train_orig,(60000,-1))
x_test = np.reshape(x_test_orig,(10000,-1))

print(y_train[0])

def toenc(dim,x):
    arr = np.zeros((dim,10))
    for i in range(dim):
        arr[i,x[i]]=1
    return arr

y_train=toenc(60000,y_train)
y_test=toenc(10000,y_test)

model = models.Sequential()

model.add(layers.Dense(15,activation='relu',input_shape=(784,)))

model.add(layers.Dense(10,activation='softmax'))


model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics = ['accuracy'])

model.fit(x_train,y_train, epochs=20,batch_size=512)

result = model.evaluate(x_test,y_test)

print(result)


