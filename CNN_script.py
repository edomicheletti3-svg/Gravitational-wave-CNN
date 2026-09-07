from mly.datatools import DataPod, DataSet, generator


import os
import matplotlib.pyplot as plt
from math import ceil

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import pycbc.pnutils as ut
from pycbc.waveform import get_td_waveform
from pycbc.waveform import get_fd_waveform
from pycbc.detector import Detector
from scipy import signal

import numpy as np
from gwpy.timeseries import TimeSeries as gwTS

from mly.projectwave import *
from mly.exceptions import *
from mly.plugins import *
from mly.waveforms import *
import sys
from mly.datatools import *
from mly.tools import toCategorical

import time ,os
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense,Conv1D, MaxPool2D, Flatten, Input ,Activation, Lambda,Concatenate, Minimum, SpatialDropout2D,Activation
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint ,ReduceLROnPlateau
from tensorflow.keras import optimizers, initializers
from tensorflow.config import threading
from tensorflow.keras.layers import GlobalAveragePooling2D
from mly.tools import correlate   
from tensorflow.keras.layers import (Input, Conv2D, AveragePooling2D,
                                      Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from clr_callback import CyclicLR

training_set = DataSet.load('/home/edoardo.micheletti/model_tutorials/Models/training_data/50000_training_set_14to32.pkl')

parts = [np.load(f'/home/edoardo.micheletti/model_tutorials/Models/C_grams/subset_{i}_COGRAM.npy') for i in range(0,10)]

C_gram = np.concatenate(parts,axis =0 )
Y0 = training_set.exportLabels('type')
Y,translation = toCategorical(Y0,from_mapping=['noise','signal'])

C_gram_train, C_gram_test, Y_train, Y_test = train_test_split(
    C_gram, Y, test_size=0.1, random_state=0)

    
BS= 100
EP = 200
drop_out = 0.2
stride = 2

iterations = len(C_gram_train) / BS


step_size = 5 * iterations
base_lr = 2e-5
max_lr = 1e-4


def make_branch(input_shape):
    inp = Input(shape=input_shape, name='coherencegram')

    #Block 1

    x = Conv2D(32, kernel_size=(4, 10), padding='same',activation='relu')(inp)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(drop_out)(x)
  
   

    

    #Block 2
    
    x = Conv2D(64, kernel_size=(3, 3) ,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)

    x = Conv2D(64, kernel_size=(3, 3) ,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)

    
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(drop_out)(x)

    

    # Block 3
    x = Conv2D(32, kernel_size=(2, 2) ,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(32, kernel_size=(2, 2) ,padding='same',activation='relu')(x)
    x = BatchNormalization()(x)
    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = SpatialDropout2D(drop_out)(x)

    
 
    return Model(inp,x)

branch = make_branch((29, 60, 1))

inp = Input(shape=(29, 60, 3))

f = [branch(Lambda(lambda t,c=c: t[...,c:c+1])(inp)) for c in range(3)]

x = Concatenate()(f)   


x = Flatten()(x)



x = Dense(256, activation= 'relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.6)(x)

out = Dense(2, activation='softmax')(x)

model = Model(inp, out)
model.compile(optimizer=optimizers.Nadam(learning_rate=1e-4),
              loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


es = EarlyStopping(monitor='val_loss', patience = 10, restore_best_weights=True)
clr = CyclicLR(base_lr= base_lr, max_lr= max_lr,
                        step_size=step_size)


hist = model.fit(
    C_gram_train,
    Y_train,
    epochs=EP,
    batch_size=BS,
    validation_data=(C_gram_test, Y_test),
    callbacks=[es,clr]
)

