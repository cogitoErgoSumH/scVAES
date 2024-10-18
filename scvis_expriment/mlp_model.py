import warnings
warnings.filterwarnings("ignore")
import pickle
# import tensorflow.compat.v1 as tf
from keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score, confusion_matrix
from keras.utils import to_categorical
from keras import models
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def train_model(batch_size,dimention_count=None,all_data=None):
    batch_size = batch_size
    epochs =300
    number=3
    if all_data is None:
        all_data=pd.read_csv('./gene_name_exp_train_df_{}.csv'.format(dimention_count))
    # all_data=load_feature_data('','')
    train_data_last_index =str(all_data.shape[1] - 1)
    # all_data=all_data[all_data[train_data_last_index]!=2]
    print(all_data.columns)
    test_data=all_data.iloc[:int(all_data.shape[0]*0.33)]
    train_data=all_data.iloc[int(all_data.shape[0]*0.33):]
    print(train_data.columns)
    # test_data=all_data.iloc[int(all_data.shape[0]*0.66):]
    # train_data=all_data.iloc[:int(all_data.shape[0]*0.66)]

    y_train=train_data[train_data_last_index]
    train_data.drop(train_data_last_index,axis=1,inplace=True)
    x_train=train_data
    print(x_train.shape)
    y_test=test_data[train_data_last_index]
    test_data.drop(train_data_last_index,axis=1,inplace=True)
    x_test=test_data
    print(x_test.shape)

    x_train_numpy = np.array(x_train)
    x_test_numpy = np.array(x_test)
    y_train_numpy = y_train.values
    y_test_numpy = y_test.values
    num_features = x_train_numpy.shape[1]
    num_classes = len(set(list(y_train_numpy)))
    print(num_classes)

    # scaler = StandardScaler()
    # scaler.fit(x_train_numpy)
    # x_train_numpy = scaler.transform(x_train_numpy)
    # x_test_numpy = scaler.transform(x_test_numpy)

    y_train_one_hot = to_categorical(y_train_numpy, num_classes=num_classes)
    y_test_one_hot = to_categorical(y_test_numpy, num_classes=num_classes)

    keras.backend.clear_session()
    model = models.Sequential()
    # model.add(layers.Dense(256, activation='relu',input_shape=(num_features,)))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(256, activation='relu'))
    # # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(128, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(32, activation='relu'))
    # #以下是新加的
    # model.add(layers.Dropout(0.5))
    # model.add(layers.Dense(64, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(32, activation='relu'))
    # model.add(layers.Dense(512, activation='relu'))

    print('项目的特征数为'+str(num_features))
    model.add(layers.Dense(512, activation='relu', input_shape=(num_features,)))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(512, activation='relu'))
    # model.add(layers.Dropout(0.25))
    # model.add(layers.Dense(16, activation='relu'))
    # model.add(layers.Dropout(0.25))
    if num_classes==2:
        model.add(layers.Dense(num_classes, activation='sigmoid'))
    elif num_classes==3:
        model.add(layers.Dense(num_classes, activation='softmax',))

    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    # model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    # model.fit(x_train_numpy, y_train_one_hot, epochs=9, batch_size=512, shuffle=True)
    file_path_hd5="weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"

    early_stopping = EarlyStopping(monitor='val_acc', patience=300, verbose=2, mode='auto', restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=file_path_hd5, monitor='val_acc', verbose=2,
                                 # save_best_only=True, mode='max', save_freq='epoch')
                                 save_best_only=True, mode='max')#黄加的
    callbacks_list = [checkpoint, early_stopping]
    # callbacks_list = [checkpoint]
    # rmsprop =keras.optimizers.adam(lr=0.01)#黄加的
    rmsprop = keras.optimizers.RMSprop(lr=0.001)
    model.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    history = model.fit(x_train_numpy, y_train_one_hot, epochs=epochs,
                        batch_size=batch_size, validation_split=0.2, shuffle=True, callbacks=callbacks_list)

    # history = model.fit(x_train_numpy, y_train_one_hot, epochs=epochs, batch_size=batch_size,
    #                     validation_data=(x_test_numpy, y_test_one_hot), shuffle=True, callbacks=callbacks_list)
    print(model.summary())
    plt.figure(figsize=(10, 6))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.grid()
    plt.tight_layout()
    plt.savefig('./end_result.pdf')





    y_pred = model.predict(x_test_numpy)

    plt.figure(figsize=(10, 6))
    for i in range(3):
        y_test_x = [j[i] for j in y_test_one_hot]
        y_predict_x = [j[i] for j in y_pred]
        fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
        plt.subplot(1, 3, i + 1)
        plt.plot(fpr, tpr)
        plt.grid()
        plt.plot([0, 1], [0, 1])
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        auc = np.trapz(tpr, fpr)
        # print('AUC:', auc)
        plt.tight_layout()
        plt.title('label' + str(i) + ', AUC:' + str(round(auc,2)))
    plt.savefig('./end_3labels.pdf')

    y_pred2 = y_pred.argmax(axis=1)
    accu_score=accuracy_score(y_test_numpy, y_pred2)
    print(accu_score)
    aaa = confusion_matrix(y_test_numpy, y_pred2)
    # plot_Matrix(aaa,classes=[0,1,2])

    bbb = (aaa.T / aaa.sum(axis=1)).T
    bbb=pd.DataFrame(bbb)
    print(pd.DataFrame(bbb))
    bbb.to_csv('./a'+str(batch_size)+'.csv')
    return accu_score


    yy_test_numpy = keras.utils.to_categorical(y_test_numpy, 3)
    plt.figure(figsize=(20, 6))
    for i in range(3):
        y_test_x = [j[i] for j in y_test_one_hot]
        y_predict_x = [j[i] for j in y_pred]
        fpr, tpr, thresholds = metrics.roc_curve(y_test_x, y_predict_x, pos_label=1)
        plt.subplot(1, 3, i + 1)
        plt.plot(fpr, tpr)
        plt.grid()
        plt.plot([0, 1], [0, 1])
        plt.xlabel('FP')
        plt.ylabel('TP')
        plt.ylim([0, 1])
        plt.xlim([0, 1])
        auc = np.trapz(tpr, fpr)
        print('AUC:', auc)
        plt.title('label' + str(i) + ', AUC:' + str(auc))
    # plt.show()

    plt.savefig('./'+str(time.time())+'auc.jpg')


all_data=pd.read_csv('./data/two_gene_.csv')
train_model(1024,0,all_data)