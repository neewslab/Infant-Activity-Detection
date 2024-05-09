# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 20:55:37 2023

@author: duttahr1
"""



import tensorflow as tf
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from tensorflow.keras.optimizers import SGD, Adagrad
from random import seed
from random import randint
import math
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import numpy as np
import copy


k_for_kfold = 10


kfold = KFold(k_for_kfold, True)


# load the dataset
# dataset = loadtxt("Data_master_sh_norm_eq.csv", delimiter=',')
# dataset = loadtxt("Master_dataset_v3_cut_sh3.csv", delimiter=',')
# dataset = loadtxt("Master_dataset_v4_cut_sh.csv", delimiter=',')
# dataset = loadtxt("Fall_down_sam_only_fall.csv", delimiter=',')
# dataset = loadtxt("Crawl_dataset.csv", delimiter=',')
# dataset = loadtxt("Walking_Cruising_dataset_v1.csv", delimiter=',')
# dataset = loadtxt("Set7\Sit_dataset_3.csv", delimiter=',')
# dataset = loadtxt("Set7\Fall_dataset_5.csv", delimiter=',')
# dataset = loadtxt("Set7\Kneel_dataset_2.csv", delimiter=',')
# dataset = loadtxt("Set7\Walk_dataset3.csv", delimiter=',')
# dataset = loadtxt("Set7\Cruise_dataset.csv", delimiter=',')
# dataset = loadtxt("Set7\Kneel_dataset_3.csv", delimiter=',')
# dataset = loadtxt("Set7\Fall_dataset_2.csv", delimiter=',')
# dataset = loadtxt("Entire_data\Fall_vs_all.csv", delimiter=',')
# dataset = loadtxt("Stand_dataset.csv", delimiter=',')
# dataset = loadtxt("Sit_dataset.csv", delimiter=',')
# X_train=dataset[:,0:297]
# y_train=dataset[:,297]









# dataset = loadtxt("Three_sensors\Sit_kneel_vs_rest.csv", delimiter=',')
dataset = loadtxt("Three_sensors\Walk_vs_rest.csv", delimiter=',')


# X=dataset[:,0:297]
# y=dataset[:,297]





# model = Sequential()

# model.add(Dense(297, input_dim=297, activation='relu'))
# # model.add(Dense(297, activation='relu'))
# model.add(Dense(200, activation='relu'))
# model.add(Dense(100, activation='relu'))
# model.add(Dense(1, activation='sigmoid'))

# model.add(Dropout(0.2, input_shape=(297,)))
# model.add(Dense(297, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(400, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(500, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(600, activation='relu', kernel_constraint=maxnorm(3)))
# # model.add(Dense(800, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(500, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(400, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(200, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(100, activation='relu', kernel_constraint=maxnorm(3)))
# model.add(Dense(1, activation='sigmoid'))
# Compile model
# sgd = SGD(lr=0.01, momentum=0.9)
# model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])



# fld = kfold.split(dataset)

shuff_data_tr = []
shuff_data_test = []

for train, test in kfold.split(dataset):
    shuff_data_tr.append(dataset[train])
    shuff_data_test.append(dataset[test])


hist = []
scr_tr = []
scr_va = []

for i in range(k_for_kfold):
    
    print('Run:',i)
    
    history = 0
    
       
    model = Sequential()

    model.add(Dense(225, input_dim=225, activation='relu'))
    model.add(Dense(400, activation='relu'))
    # model.add(Dense(500, activation='relu'))
    model.add(Dense(400, activation='relu'))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    # model.add(Dense(25, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    sgd = SGD(lr=0.01, momentum=0.9)
    # adagrad = Adagrad(lr=0.01,initial_accumulator_value=0.1,
    # epsilon=1e-07,
    # name="Adagrad")
    # model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy',tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy',tf.keras.metrics.TrueNegatives(),tf.keras.metrics.FalsePositives(),tf.keras.metrics.TruePositives(),tf.keras.metrics.FalseNegatives()])
    
    
    X_tr = shuff_data_tr[i][:,0:225]
    y_tr = shuff_data_tr[i][:,225]
    X_va = shuff_data_test[i][:,0:225]
    y_va = shuff_data_test[i][:,225]
    history=model.fit(X_tr, y_tr, validation_data=(X_va,y_va), epochs=300, batch_size=10,verbose=0)
    hist.append(history)
    scr_tr.append(model.evaluate(X_tr, y_tr, verbose=0))
    scr_va.append(model.evaluate(X_va, y_va, verbose=0))
    
    
#7,21,75

true_p = []   

false_p = []

for i in range(k_for_kfold):
    false_p.append(scr_va[i][3]/(scr_va[i][2]+scr_va[i][3]))
    true_p.append(scr_va[i][4]/(scr_va[i][4]+scr_va[i][5]))





print('True_positive:',sum(true_p)/k_for_kfold)
print('False_positive:',sum(false_p)/k_for_kfold)



#%% To be coded

# test_dataset = loadtxt("Fall_down_sam_tp_fp.csv", delimiter=',')

# test_pos = test_dataset[0:492,:]
# test_neg = test_dataset[492:,:]

# shuff_pos = []
# shuff_neg = []

# for train, test in kfold.split(test_pos):
#     shuff_pos.append(test_dataset[test])
    
# for train, test in kfold.split(test_neg):
#     shuff_neg.append(test_dataset[test])
    
# tp = []
# fp = []
    
# for k in range(10):
#     X_pos = shuff_pos[k][:,0:75]
#     y_pos = shuff_pos[k][:,75]
#     X_neg = shuff_neg[k][:,0:75]
#     y_neg = shuff_neg[k][:,75]
#     sc_t = model.evaluate(X_pos, y_pos, verbose=0)
#     sc_f = model.evaluate(X_neg, y_neg, verbose=0)
#     tp.append(copy.deepcopy(sc_t))
#     fp.append(copy.deepcopy(sc_f))
    


# # evaluate the keras model
# #_, accuracy = model.evaluate(X, y)
# #print('Accuracy: %.2f' % (accuracy*100))


# # evaluate the model
# scores = model.evaluate(X_train, y_train, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
# #print("%s: %.2f" % (model.metrics_names[2], scores[2]))
# #print("%s: %.2f" % (model.metrics_names[3], scores[3]))
# # save model and architecture to single file
# model.save("model.h5")
# print("Saved model to disk")






# import matplotlib.pyplot as plt

# # summarize history for accuracy
# plt.plot(history.history['accuracy'])
# #plt.plot(history.history['val_accuracy'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# #plt.legend(['train', 'validation'], loc='lower right')
# plt.show()
# # summarize history for loss
# plt.plot(history.history['loss'])
# #plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# #plt.legend(['train', 'validation'], loc='upper right')
# plt.show()







# # test_data = loadtxt('Data_master_sh_sort_eq.csv', delimiter=',')
# # test_data = loadtxt('Master_dataset_V3_cut.csv', delimiter=',')
# # test_data = loadtxt('Master_dataset_V4_cut.csv', delimiter=',')
# test_data = loadtxt('Fall_test_V4.csv', delimiter=',')
# true_x = test_data[0:25,0:297]
# true_y = test_data[0:25,297]
# false_x = test_data[25:,0:297]
# false_y = test_data[25:,297]


# score = model.evaluate(true_x, true_y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))



# score = model.evaluate(false_x, false_y, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))