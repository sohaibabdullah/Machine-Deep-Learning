
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras import models
from keras import layers
from keras import optimizers

from keras.utils import np_utils

np.random.seed(1671)

EPOCHS=10
BATCH_SIZE=128
classes = 10
hidden_units1 = 128

(x_train_orig,y_train),(x_test_orig,y_test) = mnist.load_data()

print(x_test_orig[0].shape)
print(y_test[0])

plt.imshow(x_test_orig[0],cmap=plt.get_cmap('gray'))
plt.show()

    


x_train = x_train_orig.reshape(x_train_orig.shape[0],-1)
x_test=x_test_orig.reshape(x_test_orig.shape[0],-1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print(x_test[0].shape)




#Normalize
x_train /=255
x_test /=255

#convert label to one-hot matrix
y_train =np_utils.to_categorical(y_train,classes)
y_test = np_utils.to_categorical(y_test,classes)

model = models.Sequential()
model.add(layers.Dense(hidden_units1,activation='relu',input_shape=(x_train.shape[1],)))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(classes,activation='sigmoid'))
          

#optimizer = optimizers.SGD(lr=.01)
optimizer = optimizers.RMSprop(lr=0.001)

model.compile(loss='categorical_crossentropy',optimizer=optimizer,metrics=['accuracy'])

history=model.fit(x_train,y_train, epochs=EPOCHS,batch_size=BATCH_SIZE,verbose=2,validation_split=.2)

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

result = model.evaluate(x_test,y_test,verbose=0)

print('Test Score',result[0])
print('Test Accuracy',result[1]*100,'%')


img=x_test[0].reshape(1,784)
print(img.shape)

predictions = model.predict(img)
print(predictions.shape)
print(np.argmax(predictions))

