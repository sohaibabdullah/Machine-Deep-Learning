import numpy as np
import matplotlib.pyplot as plt

from keras.datasets import mnist


from keras import models
from keras import layers
from keras import optimizers

(x_train_orig,y_train),(x_test_orig,y_test) = mnist.load_data()

#print(x_train_orig.shape)


x_train = np.reshape(x_train_orig,(60000,-1))
x_test = np.reshape(x_test_orig,(10000,-1))

#print(y_train[0])
#print(x_train.shape)



def toenc(dim,x):
    arr = np.zeros((dim,10))
    for i in range(dim):
        arr[i,x[i]]=1
    return arr

y_train=toenc(60000,y_train)
y_test=toenc(10000,y_test)

model = models.Sequential()

model.add(layers.Dense(16,activation='relu',input_shape=(784,)))





model.add(layers.Dense(10,activation='softmax'))

#model.add(layers.Dense(10,activation='softmax',input_shape=(784,)))
model.compile(optimizer='SGD',loss='categorical_crossentropy',metrics = ['accuracy'])

history=model.fit(x_train,y_train, epochs=20,batch_size=512,verbose=2,validation_split=.2)

result = model.evaluate(x_test,y_test,verbose=0)

print(result)




history_dict = history.history
loss=history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1,len(loss)+1)

plt.plot(epochs,loss,'b',label='Training Loss')
plt.plot(epochs,val_loss,'r',label='Validation Loss')
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()




