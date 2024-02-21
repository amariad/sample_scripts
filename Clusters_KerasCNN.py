import numpy as np
import sys, os
import h5py

from astropy.io import ascii

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
import keras_tuner
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization
from keras import backend as K
from keras import activations
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.optimizers import SGD

'''This is a simple script to learn Keras and Tensorflow.
It explores hyperparamter tuning class and simple training model'''

tf.config.list_physical_devices('GPU')

path      = 'type_path_here/'
fits_path = path+'Maps_xray/'

#########################################
#########################################
### LOAD DATA 

t = ascii.read(path+"M500_1e14_halos.dat")
h = 0.6774
M_500 = t["Group_M_Crit500"]*1.0e10/h
log_M = np.log10(M_500)

normed_images = np.load(path+'Normed_low_res_fits_data_1e14M500.npy')


#########################################
#########################################
### CREATE TRAIN, VAL, TEST SETS

image_size = normed_images.shape[1]

N_halos = len(normed_images)
indeces = np.arange(0, N_halos)

### shuffle the indeces to have a shuffled dataset
np.random.seed(seed=10)
np.random.shuffle(indeces)

train_size    = int(N_halos*0.7)+1
val_test_size = int(N_halos*0.3)
### make sure sizes are coorect:
print('check split sizes', N_halos-(train_size + val_test_size))

### start/stop locations
train_stop = train_size
val_stop   = int(train_size+(val_test_size/2))
test_start = int(train_size+(val_test_size/2))

### indeces
train_inds = np.arange(0,train_stop)
val_inds   = np.arange(train_size,val_stop)
test_inds  = np.arange(test_start,N_halos)

print('train', train_inds.shape)
print('val', val_inds.shape)
print('test', test_inds.shape)
tot = train_inds.shape[0] + val_inds.shape[0] + test_inds.shape[0]
print('tot', tot )

### split the data
X_data = normed_images[indeces]
y_data = log_M[indeces]
norm   = y_data.min() ### so y-values mostly between 0 and 1

X_train, y_train = X_data[train_inds].reshape(-1, image_size, image_size, 1), y_data[train_inds]-norm
X_val, y_val     = X_data[val_inds].reshape(-1, image_size, image_size, 1), y_data[val_inds]-norm
X_test, y_test   = X_data[test_inds].reshape(-1, image_size, image_size, 1), y_data[test_inds]-norm



#########################################
#########################################
### SEARCH FOR HYPERPARAMS

BATCH_SIZE = 16
EPOCHS     = 50
INIT_LR    = 1e-5
MAX_LR     = 1e-2

steps_per_epoch = len(X_train) // BATCH_SIZE
print('steps per epoch', steps_per_epoch)

### cyclical learning rate
clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=INIT_LR,
    maximal_learning_rate=MAX_LR,
    scale_fn=lambda x: 1/(2.**(x-1)),
    step_size= 10)#2 * steps_per_epoch)


input_shape = (image_size, image_size, 1)

### FUNCTION FOR SEARCHING HYPERPARAMS
def build_model(hp):
    model = Sequential()
    
    model.add(Conv2D(64, kernel_size=(5, 5), input_shape=input_shape,
                    activation = LeakyReLU()))
   
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(48, kernel_size=(3, 3),
                    activation = LeakyReLU()))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(32, kernel_size=(3, 3),
                    activation = LeakyReLU()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    ### TUNE HYPERPARAMS
    possibilities = []
    
    ### Tune weather to add annother CNN layer
    #if hp.Boolean('CNN_4'):
            #model.add(Conv2D(16, kernel_size=(3, 3),
                    #activation = LeakyReLU()))
            #model.add(MaxPooling2D(pool_size=(2, 2)))
            
    model.add(GlobalAveragePooling2D())
    
    ### Tune whether to use dropout.
    #if hp.Boolean("dropout_1"):
    min_do1  = 0.2
    max_do1  = 0.5
    step_do1 = 0.1
    model.add(Dropout(rate=hp.Float("drop_1", min_value=min_do1, max_value=max_do1, step=step_do1)))
    possibilities.append(len(np.arange(min_do1,max_do1, step=step_do1))+1)
    
    ### choose activation function for fully connected layer
    activation_nn = hp.Choice("activation_nn", ["relu", "leaky_relu"])
    possibilities.append(2)
    
    ### choose optimizer
    #optimizers = hp.Choice("optimizer", [SGD(learning_rate=clr), Adam(learning_rate=clr)])
    #possibilities.append(2)
    
    ### choose loss function
    loss_funcs = hp.Choice('loss func', ['mean_squared_error', 'mean_absolute_error'])
    possibilities.append(2)                 
    
    ### choose num neurons
    min_n1  = 20
    max_n1  = 200
    step_n1 = 10
    model.add(Dense(
        units=hp.Int('units_nn1', min_value=min_n1, max_value=max_n1, step=step_n1),
            activation = activation_nn))
    possibilities.append(len(np.arange(min_n1,max_n1, step=step_n1))+1)
    
    #if hp.Boolean("dropout_2"):
    min_do2  = 0.2
    max_do2  = 0.5
    step_do2 = 0.1
    model.add(Dropout(rate=hp.Float("drop_2", min_value=min_do2, max_value=max_do2, step=step_do2)))
    possibilities.append(len(np.arange(min_do2,max_do2, step=step_do2))+1)
    
    ### choose num neurons
    min_n2  = 20
    max_n2  = 150
    step_n2 = 10
    model.add(Dense(
        units=hp.Int('units_nn2', min_value=min_n2, max_value=max_n2, step=step_n2),
            activation = activation_nn))
    possibilities.append(len(np.arange(min_n2,max_n2, step=step_n2))+1)
      
    ### choose num neurons
    min_n3  = 5
    max_n3  = 50
    step_n3 = 5
    model.add(Dense(
        units=hp.Int('units_nn3', min_value=min_n3, max_value=max_n3, step=step_n3),
            activation = activation_nn))
    possibilities.append(len(np.arange(min_n3,max_n3, step=step_n3))+1)
    
    model.add(Dense(1, activation='linear'))

    model.compile(
        optimizer=Adam(learning_rate=clr), loss=loss_funcs, metrics= ["mean_absolute_percentage_error"])
    tot_combos = np.prod(possibilities)
    
    print('total possible combinations:', tot_combos)
    
    return model

build_model(keras_tuner.HyperParameters())

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_loss",
    max_trials=100,
    seed=None,
    #hyperparameters=None,
    tune_new_entries=True,
    allow_new_entries=True,
    max_retries_per_trial=0,
    overwrite=True,
    directory=path,
    project_name="hyperparams_search",
    max_consecutive_failed_trials=3,
)

print(tuner.search_space_summary())

tuner.search(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[stop_early])


#########################################
#########################################
### BUILD THE CNN USING BEST HYPERPARAMS

best_hp = tuner.get_best_hyperparameters()[0]
model   = tuner.hypermodel.build(best_hp)
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=clr))

tot_epochs    = 0
loss_list     = []
val_loss_list = []

start      = time.time()
epochs     = 5000
batch_size = 16 #lower this value if you get a memory error
hist       = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                 batch_size=batch_size, verbose=0,  epochs=epochs)#, callbacks=[reduce_lr])
stop       = time.time()
print('total training minutes:', (stop-start)/60)

prediction = model.predict(X_test, verbose=0, batch_size=batch_size).flatten()
true       = y_test + norm
pred       = prediction + norm
print('rmse:', np.sqrt(metrics.mean_squared_error(true, pred)))

### for plotting training and validation losses
tot_epochs +=epochs
for i in range(epochs):
    loss_list.append(hist.history['loss'][i])
    val_loss_list.append(hist.history['val_loss'][i])

np.save(path+'training_losses', np.array(loss_list))
np.save(path+'validation_losses', np.array(val_loss_list))

### for plotting the results
np.save(path+'CNN_ytrue', true)
np.save(path+'CNN_pred', pred)

