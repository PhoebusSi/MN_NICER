from __future__ import print_function
import numpy as np
from keras.layers import concatenate
import pickle
import tensorflow as tf
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Lambda
from keras.layers import Embedding,Bidirectional
from keras.layers import Convolution1D, LSTM
from keras.datasets import imdb
from keras import backend as K
from keras.layers import Add
from keras.optimizers import Adadelta
from preprocessing import *
from keras.preprocessing import sequence as sq
from keras.layers import Dense, Dropout, Activation, Lambda,merge,Input,TimeDistributed,Flatten,multiply,Multiply
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.engine.topology import Layer
import keras.backend.tensorflow_backend as K

from sklearn.metrics import classification_report
# set parameters:
#max_features = 87927+1
max_features = 34350+1 #length of vocab
is_concat_the_semantic_in_beginning = True
is_exits_bert_buffer=True
maxlen = 70
batch_size = 300
embedding_dims = 300
nb_filter = 150
draw_2dim=True
filter_length = 3
hidden_dims = 150
nb_epoch = 5
drop_out = 0.3
accs = []
class MyLayer(Layer):

    def __init__(self, output_dim,**kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_layer):
        # 为该层创建一个可训练的权重
        self.kernel = self.add_weight(name='kernel',
                                      shape=(1,self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(MyLayer, self).build(input_layer)  # 一定要在最后调用它

    def call(self,input_layer):
        #print('x',input_layer.shape)
        return tf.tile(self.kernel, [tf.shape(input_layer)[0], 1])

def get_data(flag=0):
    print('Loading data ...')
    if flag==5 or flag == 6:
        train_save, test_save, val_save,y_list, sents_aft_bert , semantic_of_sents = init_data(flag)
        X_train, X_test, X_val,y_train, y_test, y_val =load_data2(train_save, test_save, val_save, y_list)

    print(len(X_train), 'train sequences')
    print(len(X_test), 'test sequences')

    X_train = sq.pad_sequences(X_train, maxlen=maxlen)
    X_test = sq.pad_sequences(X_test, maxlen=maxlen)
    X_val = sq.pad_sequences(X_val, maxlen=maxlen)
    if flag==5 or flag == 6:
        return X_train, X_test, X_val, y_train, y_test, y_val, sents_aft_bert, semantic_of_sents

def multi_model(flag,semantic_prob,X_train, X_val,y_train, y_val,sents,semantic_sents_vec ):
#def multi_model(X_train, X_val,y_train, y_val,sents,semantic_sents_vec ):

    print('Build model...')
    model = Sequential()
    input_layer = Input(shape=(maxlen,),dtype='int32', name='main_input')
    bc = BertClient()
    print("shape of bert",bc.encode(['我不知道啊']).shape)

    bert_input_layer = Input(shape=(768,),dtype='float32', name='bert_input')

    if flag==5:
        if is_concat_the_semantic_in_beginning:
            semantic_input_layer = Input(shape=(maxlen, 8,), dtype='float32', name='semantic_input')
        else:
            semantic_input_layer = Input(shape=(8,), dtype='float32', name='semantic_input')
    else:
        if is_concat_the_semantic_in_beginning:
            semantic_input_layer = Input(shape=(maxlen,7,), dtype='float32', name='semantic_input')
        else:
            semantic_input_layer = Input(shape=(7,),dtype='float32', name='semantic_input')

    emb_layer = Embedding(max_features,
                          embedding_dims,
                          input_length=maxlen
                          )(input_layer)

    if is_concat_the_semantic_in_beginning:
        emb_layer = concatenate([emb_layer,semantic_input_layer])  #当没有semantic的时候，emb_layer就是词向量

    def max_1d(X):
        return K.max(X, axis=1)


    con3_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)


    pool_con3_layer = Lambda(max_1d, output_shape=(nb_filter,))(con3_layer)

    con4_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)
    print("con4_layer",con4_layer)

    pool_con4_layer = Lambda(max_1d, output_shape=(nb_filter,))(con4_layer)
    print("pool_con4_layer",pool_con4_layer)


    con5_layer = Convolution1D(nb_filter=nb_filter,
                        filter_length=7,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1)(emb_layer)

    pool_con5_layer = Lambda(max_1d, output_shape=(nb_filter,))(con5_layer)
    print("con5_layer",con5_layer)
    print("pool_con5_layer",pool_con5_layer)


    cnn_layer = concatenate([pool_con3_layer, pool_con5_layer,pool_con4_layer ])#, mode='concat')
    print("cnn_layer",cnn_layer)
    #cnn_layer(?,450)
    #LSTM

    #去掉lstm
    x = Embedding(max_features, embedding_dims, input_length=maxlen)(input_layer)
    if is_concat_the_semantic_in_beginning:
        #将情绪向量合并到词向量的后面！！！一会儿把这里的注释取消掉！
        x = concatenate([x, semantic_input_layer])
        #不加入情绪向量
        #x=x



    #lstm_layer=LSTM(128)(x)
    lstm_layer = Bidirectional(LSTM(128,return_sequences=False),merge_mode='sum')(x)
    print("lstm_layer",lstm_layer)
    #lstm_layer(?,128)
    cnn_lstm_layer = concatenate([lstm_layer, cnn_layer])#, mode='concat')
    print("cnn_lstm_layer", cnn_lstm_layer)
    #cnn_lstm(?,578)
    cnn_lstm_bert_layer=concatenate([cnn_lstm_layer,bert_input_layer])
    #print("cnn_lstm_bert_layer",cnn_lstm_bert_layer)
    cnn_bert_layer=concatenate([cnn_layer,bert_input_layer])
    lstm_bert_layer=concatenate([lstm_layer,bert_input_layer])
    #做对比实验到时候在这里选择将哪几个通道结合在一起
    dense_layer = Dense(hidden_dims*3, activation='sigmoid')(cnn_lstm_bert_layer)
    #dense_layer = Dense(hidden_dims*3, activation='sigmoid')(cnn_bert_layer)
    #dense_layer = Dense(hidden_dims*3, activation='sigmoid')(lstm_bert_layer)
    #dense_layer = Dense(hidden_dims*3, activation='sigmoid')(cnn_lstm_layer)


    #Dense?
    #dense_layer(?,200)
    drop_out_layer= Dropout(drop_out)(dense_layer)
    if draw_2dim:
        #新加一层全连接用来绘画出当前的分布效果，在获取最佳结果(不是为了画图的时候)不添加这一层
        draw= Dense(2, trainable=True,name='draw_sth')(drop_out_layer)

    if flag==5:
        if draw_2dim:
            output_no_semantic_layer = Dense(8, trainable=True, activation='softmax')(draw)  #
        else:
            output_no_semantic_layer = Dense(8, trainable=True, activation='softmax')(drop_out_layer)  #
    else:
        if draw_2dim:
            output_no_semantic_layer = Dense(7, trainable=True, activation='softmax')(draw)
        else:
            output_no_semantic_layer = Dense(7, trainable=True,activation='softmax')(drop_out_layer)# 8 classes
    if not is_concat_the_semantic_in_beginning:
        #一开始将情感向量和embedding连接在一起了，所以后面不用再添加这个了
        #这个分支是用来比较:
        #   1.构造情感词向量
        #   2.将一整句话所有词的情感向量加在一起作为整个句子的情感向量然后按照一定比例并入多通道的输出
        #两者哪种关系好

        if flag==5:
            w_batch = MyLayer(output_dim=8)(output_no_semantic_layer)
        else:
            w_batch=MyLayer(output_dim=7)(output_no_semantic_layer)

        semantic_output_layer=Multiply()([w_batch,semantic_input_layer])
        #先给每一个情感元素可训练的权重，然后统一再给情感向量一个权重

        def dot_w(X):
            return X* semantic_prob

        if flag==5:
            semantic_output_layer = Lambda(dot_w, output_shape=(8,))(semantic_output_layer)  #
        else:
            semantic_output_layer=Lambda(dot_w,output_shape=(7,))(semantic_output_layer)#这个手动设定，也可以

        output_layer= Add()([output_no_semantic_layer,semantic_output_layer])

        print("dense_layer,output_no_semantic_layer ,output_layer",dense_layer,output_no_semantic_layer ,output_layer)
    output_layer=output_no_semantic_layer


    model = Model(input=[input_layer,bert_input_layer,semantic_input_layer], output=[output_layer])
    #model = Model(input=[input_layer，bert_layer], output=[output_layer])
    adadelta = Adadelta(lr=1.0, rho=0.95, epsilon=1e-06)
    def focal_loss(y_true, y_pred):
        #a取比较小的值来降低负样本（多的那类样本）的权重。
        #[disgust,sadness,anger,happiness,like,fear,superise]
        #[307,239,186,287,410,50,78]

        if flag == 6:
            a = sum([1/307,1/239,1/186,1/287,1/410,1/50,1/78])
            return -tf.multiply( [1/307/a,1/239/a,1/186/a,1/287/a,1/410/a,1/50/a,1/78/a],((y_true-y_pred)**2)*tf.log(y_pred))
        if flag == 5:
            a = sum([1/292,1/408,1/521,1/168,1/131,1/34,1/25,1/85])
            return -tf.multiply( [1/292/a,1/408/a,1/521/a,1/168/a,1/131/a,1/34/a,1/25/a,1/85/a],((y_true-y_pred)**2)*tf.log(y_pred))
        return -((y_true-y_pred)**2)*tf.log(y_pred)

    model.compile(loss=focal_loss,
                  optimizer="adamax",
                  metrics=['accuracy'])

    checkpoint = ModelCheckpoint('CNN-LSTM-weights/'+'weights.hdf5',
                                 monitor='val_acc', verbose=0, save_best_only=True,mode='max')


    if flag==5 or flag==6:
        model.fit([X_train, np.array(sents['bert_train']), np.array(semantic_sents_vec['semantic_train'])], y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  callbacks=[checkpoint],
                  validation_data=(
                      [X_val, np.array(sents['bert_test']), np.array(semantic_sents_vec['semantic_test'])], y_val))
    else:
        model.fit([X_train, np.array(sents['bert_train']), np.array(semantic_sents_vec['semantic_train'])], y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  callbacks=[checkpoint],
                  validation_data=(
                  [X_val, np.array(sents['bert_val']), np.array(semantic_sents_vec['semantic_val'])], y_val))
    model.load_weights('CNN-LSTM-weights/' + 'weights.hdf5')
    #model.compile(loss='categorical_crossentropy',
    model.compile(loss=focal_loss,
                  optimizer="adamax",
                  metrics=['accuracy'])

    return model
def test(X_test, y_test, sents, semantic_sents,model):
    #用来获取当前分类的成绩
    score, acc = model.evaluate([X_test,np.array(sents),np.array(semantic_sents)], y_test, batch_size=batch_size)
    y_pred = model.predict([X_test, np.array(sents), np.array(semantic_sents)], batch_size=batch_size)

    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0
    #print("yy", y_test, y_pred, y_test.shape, y_pred.shape)
    print(classification_report(y_test, y_pred,digits=4))
    print("", acc)
    #print(model.metrics_names)
    return acc
def test2(X_test, y_test, sents, semantic_sents,model):
    # 得出绘画所需要的二维的点，将高维的点映射成二维空间

    y_pred=model.predict([X_test,np.array(sents),np.array(semantic_sents)], batch_size=batch_size)

    print("yy", y_test, y_pred, y_test.shape, y_pred.shape)
    f = open('jiaochashangloss_7class_allbert_IntentDetectionOutput.txt', 'w')
    out_int=[]
    pos_x=[]
    pos_y=[]
    for i in range(len(y_test)):
        intent_num=np.argmax(y_test[i])+1
        out_int.append(intent_num)
        pos_x.append(y_pred[i][0])
        pos_y.append(y_pred[i][1])
    #x = (x - min)/(max - min)
    x_min = np.min(pos_x)
    y_min = np.min(pos_y)
    x_max = np.max(pos_x)
    y_max = np.max(pos_y)
    for i in range(len(y_test)):
        f.write(str(out_int[i]) +";"+ str((pos_x[i]-x_min)/(x_max-x_min))+";"+str((pos_y[i]-y_min)/(y_max-y_min))+"\n")
        print(str(out_int[i]) +";"+ str((pos_x[i]-x_min)/(x_max-x_min))+";"+str((pos_y[i]-y_min)/(y_max-y_min))+"\n")
    f.close()


def train(flag):
    #if flag=5, the data is Chinese_DATA_2
    X_train, X_test, X_val, y_train, y_test, y_val, bert_sents, semantic_sents_vec= get_data(
        flag)



    if is_concat_the_semantic_in_beginning:
        model = multi_model(flag,0, X_train, X_val, y_train, y_val, bert_sents, semantic_sents_vec)
        draw_model = Model(input=model.input,output=model.get_layer("draw_sth").output)
        #draw_output= draw_model.predict(0, X_train, X_val,flag,0, X_train, X_val,)
        print(test(X_test, y_test, bert_sents['bert_test'], semantic_sents_vec['semantic_test'], model))
        test2(X_train, y_train, bert_sents['bert_train'], semantic_sents_vec['semantic_train'], draw_model)
        #test2(X_test, y_test, bert_sents['bert_test'], semantic_sents_vec['semantic_test'], draw_model)
        print('test_done')

    else:
        semantic_prob_list=[0,0.0001,0.001,0.01,0.1,1]
        #semantic_prob_list=[0.001]
        for semantic_prob in semantic_prob_list:
            model=multi_model(flag,semantic_prob,X_train, X_val,y_train, y_val,bert_sents,semantic_sents_vec)
            print(test(X_test, y_test, bert_sents['bert_test'],semantic_sents_vec['semantic_test'],model))
            #print ("X_test",X_test,X_test.shape)
            print('test_done')

if __name__ == "__main__":
    #函数train中的flag要么为5要么为6，分别对应两种数据
    #train(5)
    train(6)

