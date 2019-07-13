# coding=utf-8
import importlib
import sys
importlib.reload(sys)
import numpy as np
import re
import os
import pickle
import numpy
import jieba
import tensorflow as tf
from keras.layers import concatenate
from bert_serving.client import BertClient
from MN_NICER import is_concat_the_semantic_in_beginning,is_exits_bert_buffer,maxlen as is_max_len
#运行前这里的flag要和传入train()函数中的flag值保持一致
flag=6
if flag==5:
    base_dir="Chinese_DATA_2"
    test_data=base_dir+"/test_data.txt"
    train_data=base_dir+"/train_data.txt"
    val_data=base_dir+"/test_data.txt"
if flag==6:
    base_dir="DATA_clean"
    test_data=base_dir+"/test_data.txt"
    train_data=base_dir+"/train_data.txt"
    val_data=base_dir+"/val_data.txt"
dict={}
y_list={}

def init_data(flag=0):

    open_files2 = [train_data, test_data, val_data]
    word2id= {}
    id2word={}
    index = 1
    maxlen = 0
    avglen = 0
    count100 = 0
    print('initing data')
    if flag==5:
        test_save = "Chinese_DATA_2/test_save.npy"
        train_save = "Chinese_DATA_2/train_save.npy"
        val_save = "Chinese_DATA_2/val_save.npy"
        dict = load_dict('Dict/new_dict.txt')
    elif flag==6:
        #print("DATA_CLEAN_DATA_IS_7_class")
        test_save = "DATA_clean/test_save.npy"
        train_save = "DATA_clean/train_save.npy"
        val_save = "DATA_clean/val_save.npy"
        dict = load_dict('Dict/EmotionLexicon.txt')

    save_files2 = [train_save, test_save, val_save]
    sents_aft_bert = {}

    for open_file, save_file in zip(open_files2,save_files2):
        print(open_files2,save_files2)
        print(open_file,save_file,'OPEN,SAVE')
        pos=[]

        file = open(open_file, 'r', encoding='utf-8')
        semantic_of_sents = locals()
        semantic_of_all_word = []
        if flag==5:
            data_type="train" if  "Chinese_DATA_2/train" in str(open_file) else "test" if "Chinese_DATA_2/test" in str(open_file) else "val"
        if flag==6:
            data_type="train" if  "DATA_clean/train" in str(open_file) else "test" if "DATA_clean/test" in str(open_file) else "val"
        y_list['y_' + data_type] = []

        sents_aft_bert['bert_'+data_type.replace('english_',"")] = []
        semantic_of_sents['semantic_'+data_type.replace('english_',"")]=[]


        bc = BertClient()
        for aline in file.readlines():
            aline = aline.replace('\n', "")
            #print (aline)
            labels,aline = aline.split('\t',1)
            if flag==5:
                'Sorrow anxiety love joy expect anger surprise hate'
                if labels=='Sorrow':
                    y_list['y_' + data_type].append([ 1,0, 0, 0, 0, 0, 0,0])
                elif labels=='Anxiety':
                    y_list['y_' + data_type].append([ 0, 1,0, 0, 0, 0, 0,0])
                elif labels=='Love':
                    y_list['y_' + data_type].append([ 0, 0, 1,0, 0, 0, 0,0])
                elif labels=='Joy':
                    y_list['y_' + data_type].append([ 0, 0, 0, 1,0, 0, 0,0])
                elif labels == 'Expect':
                    y_list['y_' + data_type].append([ 0, 0, 0, 0, 1,0, 0, 0])
                elif labels=='Anger':
                    y_list['y_' + data_type].append([ 0, 0, 0, 0, 0, 1,0,0])
                elif labels=='Surprise':
                    y_list['y_' + data_type].append([ 0, 0, 0, 0, 0, 0,1,0])
                elif labels=='Hate':
                    y_list['y_' + data_type].append([ 0, 0, 0, 0, 0, 0,0,1])

            else:
                if labels=='0':
                    y_list['y_' + data_type].append([1,0])
                elif labels=='1':
                    y_list['y_' + data_type].append([0, 1])
                elif labels=='disgust':
                    y_list['y_' + data_type].append([1,0,0,0,0,0,0])
                elif  labels=='sadness':
                    y_list['y_' + data_type].append([0,1,0,0,0,0,0])
                elif  labels=='anger':
                    y_list['y_' + data_type].append([0,0,1,0,0,0,0])
                elif  labels=='happiness':
                    y_list['y_' + data_type].append([0,0,0,1,0,0,0])
                elif  labels=='like':
                    y_list['y_' + data_type].append([0,0,0,0,1,0,0])
                elif  labels=='fear':
                    y_list['y_' + data_type].append([0,0,0,0,0,1,0])
                else:
                    #labels=='surprise':
                    y_list['y_' + data_type].append([0,0,0,0,0,0,1])

            aline_list = jieba.cut(aline)
            if flag==5:
                sents_semantic_vec = np.array([0.0] * 8)
            else:
                sents_semantic_vec = np.array([0.0] * 7)

            ids = np.array([], dtype='int32')

            if is_concat_the_semantic_in_beginning:
                sents_semantic_mat = []
            else:

                if flag==5:
                    sents_semantic_mat = np.array([0.0] * 8)
                else:
                    sents_semantic_mat = np.array([0.0]*7)
            for word in aline_list:

                if flag==5:
                    word_semantic_vec = np.array([0.0] * 8)
                else:
                    word_semantic_vec = np.array([0.0]*7)

                word = word.lower()
                if word in word2id:
                    ids = np.append(ids, word2id[word])
                else:
                    if word != '':
                        # print (word, "not in vocalbulary")
                        word2id[word] = index
                        id2word[index] = word
                        ids = np.append(ids, index)
                        index = index + 1
                if word in dict.keys():
                    word_semantic_vec=dict[word]

                else:
                    #print(word,"not in dict")
                    if not is_concat_the_semantic_in_beginning:
                        continue
                    else:
                        if flag ==5:
                            word_semantic_vec = [0, 0, 0, 0, 0, 0, 0,0]
                        else:
                            word_semantic_vec=[0,0,0,0,0,0,0]
                if is_concat_the_semantic_in_beginning:
                    sents_semantic_mat.append(word_semantic_vec)

                else:
                    sents_semantic_mat=np.vstack((sents_semantic_mat,word_semantic_vec))


            if is_concat_the_semantic_in_beginning:

                word_semantic_vec=list(sents_semantic_mat)
                while len(word_semantic_vec)<is_max_len:

                    if flag==5:
                        word_semantic_vec.append([0, 0, 0, 0, 0, 0, 0,0])
                    else:
                        word_semantic_vec.append([0,0,0,0,0,0,0])

                if len(word_semantic_vec)>is_max_len:
                    word_semantic_vec=word_semantic_vec[:is_max_len]
                word_semantic_vec=np.array(word_semantic_vec)
                #print('word_semantic_vec.',word_semantic_vec.shape)
                semantic_of_sents['semantic_' + data_type.replace('english_',"")].append(word_semantic_vec)
            else:
                sents_semantic_vec=np.sum(sents_semantic_mat,axis=0)

                if sents_semantic_vec.all() == 0.0:
                    if flag==5:
                        semantic_of_sents['semantic_' + data_type.replace('english_', "")].append(
                            [0, 0, 0, 0, 0, 0, 0,0])
                    else:
                        semantic_of_sents['semantic_' + data_type.replace('english_',"")].append([0,0,0,0,0,0,0])
                else:
                    semantic_of_sents['semantic_'+data_type.replace('english_',"")].append(sents_semantic_vec)
            if len(ids) > 0:
                pos.append(ids)

            aline_no_space=re.sub('\s', ',', aline)


            if not is_exits_bert_buffer:

                sents_aft_bert['bert_'+data_type.replace('english_',"")].append(bc.encode([aline_no_space]).ravel())


        if is_concat_the_semantic_in_beginning:
            print(np.array(semantic_of_sents['semantic_' + data_type.replace('english_',"")]).shape,"SENTS")
        file.close()

        np.save(save_file, pos)#word2ids
        for li in pos:
            if maxlen < len(li):
                maxlen = len(li)
            avglen += len(li)
            if len(li) > 70:
                count100 += 1

        if flag==5:
            if not is_exits_bert_buffer:
                #output = open('Chinese_DATA_2_fine_tuning_dict.pkl', 'wb')
                #output = open('erroe_Chinese_DATA_2_Nofine_tuning_dict.pkl', 'wb')
                output = open('New_Chinese_DATA_2_Nofine_tuning_dict.pkl', 'wb')
                pickle.dump(sents_aft_bert, output)
                output.close()
            else:
                if os.path.exists('Chinese_DATA_2_fine_tuning_dict.pkl'):
                #if os.path.exists('New_Chinese_DATA_2_Nofine_tuning_dict.pkl'):
                    pkl_file = open('Chinese_DATA_2_fine_tuning_dict.pkl', 'rb')
                    print("using no_finetune data")
                    #pkl_file = open('New_Chinese_DATA_2_Nofine_tuning_dict.pkl', 'rb')
                    sents_aft_bert = pickle.load(pkl_file)
        else:
            if not is_exits_bert_buffer:
                output = open('1050fine_tuning_data_dict.pkl', 'wb')
                pickle.dump(sents_aft_bert, output)
                output.close()
            else:
                if os.path.exists('1050fine_tuning_data_dict.pkl'):
                    pkl_file = open('1050fine_tuning_data_dict.pkl', 'rb')
                    sents_aft_bert = pickle.load(pkl_file)

    return train_save, test_save, val_save, y_list, sents_aft_bert, semantic_of_sents

def load_data2(train_save,test_save,val_save,y_list):
    train = np.load(train_save)
    test = np.load(test_save)
    val = np.load(val_save)

    #print(y_list.keys())
    y_train = np.array(y_list['y_train'])
    y_test =np.array(y_list["y_test"])
    y_val=np.array(y_list["y_test"])
    print("train",train)
    return train,test,val,y_train,y_test,y_val


def load_dict(dict_path):

    f=open(dict_path,'r',encoding="utf-8")
    f.readline()
    lines=f.readlines()
    for i in lines:
        word,vec=i.split('=')
        vec_list=vec.strip().split(' ')
        for i in range(len(vec_list)):
            vec_list[i]=np.float(vec_list[i])

        vec_array=np.array(vec_list)

        dict[word]=vec_array

    return dict

