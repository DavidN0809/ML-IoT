import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'} value to print more messages
logging.getLogger('tensorflow').setLevel(logging.ERROR)
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Input, DepthwiseConv2D, Dropout, Add, GlobalAveragePooling2D, Input, SeparableConv2D

from tensorflow.keras.models import Sequential, Model  
from tensorflow.keras.callbacks import EarlyStopping

print(tf.__version__)
print(tf.test.gpu_device_name())


from tensorflow.keras.layers import Conv2D, BatchNormalization, Add, Activation, Dropout
from tensorflow.keras.initializers import he_normal
from tensorflow.keras import backend as K

#conv d and batchnorm should be 7
#dense should be 2
#params should be 704842
def build_model1():
    input_shape = (32, 32, 3)

    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        # Add a batchnorm layer here
        #BatchNormalization(),
        
        Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        Conv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        #finishing layers
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


#params should be 104138
def build_model2():
    input_shape = (32, 32, 3)

    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        # Add a batchnorm layer here
        #BatchNormalization(),
        
        SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(128, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        #finishing layers
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model

        
#params should be 709066
#add should be 3
# but add dropout layers after the convolutions and residual connections 
#(aka skip connections) around every two convolutional blocks 
#(a "block" here refers to a convolution layer, batchnorm, and dropout). 
#Start the skip connections after the first convolution layer,
def build_model3():
    input_shape = (32, 32, 3)

    # Define the input tensor
    inputs = tf.keras.Input(shape=input_shape)

    # First convolution block
    x = tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)

    # Second convolution block with residual connection
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    residual = x
    x = tf.keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.add([x, residual])
    
    # Third convolution block with residual connection
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    residual = tf.keras.layers.Conv2D(128, kernel_size=(1,1), strides=(2,2), padding='same')(residual)
    residual = tf.keras.layers.BatchNormalization()(residual)
    x = tf.keras.layers.add([x, residual])
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)


    # Fourth convolution block with residual connection
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    residual = x
    x = tf.keras.layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.add([x, residual])

    # Finishing layers
    x = tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10)(x)

    # Define the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model
       
#params less than 50000
def build_model50k():
    input_shape = (32, 32, 3)

    model = tf.keras.Sequential([
        Input(shape=input_shape),
        Conv2D(32, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), strides=(2,2), activation="relu", padding='same'),
        BatchNormalization(),
        
        # Add a batchnorm layer here
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),

        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),
        
        SeparableConv2D(64, kernel_size=(3,3), activation="relu", padding='same'),
        BatchNormalization(),


        #finishing layers
        MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(10)
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    
    return model


# no training or dataset construction should happen above this line
if __name__ == '__main__':

  ########################################
  ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # Now separate out a validation set.
    val_frac = 0.1
    num_val_samples = int(len(train_images)*val_frac)
    # choose num_val_samples indices up to the size of train_images, !replace => no repeats
    val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
    trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
    val_images = train_images[val_idxs, :,:,:]
    train_images = train_images[trn_idxs, :,:,:]

    val_labels = train_labels[val_idxs]
    train_labels = train_labels[trn_idxs]
    
    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    val_labels = val_labels.squeeze()

    input_shape  = train_images.shape[1:]
    train_images = train_images / 255.0
    test_images  = test_images  / 255.0
    val_images   = val_images   / 255.0
    print("Training Images range from {:2.5f} to {:2.5f}".format(np.min(train_images), np.max(train_images)))
    print("Test     Images range from {:2.5f} to {:2.5f}".format(np.min(test_images), np.max(test_images)))
    
    #####################################################################################################
    #building models
    
    ##model 1
    model1 = build_model1()

    # print the model summary
    print('Switching to model 1')
    model1.summary()
    print('Model 1 training')
    train_hist = model1.fit(train_images, train_labels, 
                      validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                      epochs=50)
    K.clear_session()
  ## Build, compile, and train model 2 (DS Convolutions)
    print('Switching to model 2')
    model2 = build_model2()
    
    # print the model summary
    model2.summary()
    
    print('Model 2 training')
    train_hist = model2.fit(train_images, train_labels, 
                      validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                      epochs=50)
    K.clear_session()
  ### Repeat for model 3 and your best sub-50k params model
    print('Switching to model 3')
    model3=build_model3()
    # print the model summary
    model3.summary()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    print('Model 3 training')
    train_hist = model3.fit(train_images, train_labels, 
                       validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                       epochs=50)
    K.clear_session()
    ###model 50k
    print('Switching to model 50k')
    model50k=build_model50k()
    
    # print the model summary
    model50k.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=5)
    print('Training model50k with early stop of patience', early_stop)
    
    train_hist = model50k.fit(train_images, train_labels, 
                       batch_size = 32, 
                       validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                       epochs=50,
                       callbacks=[early_stop])
    model50k.save('saved_models/best_model.h5')
    K.clear_session()
