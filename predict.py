# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 18:19:09 2021

@author: bkorkmaz
"""

import pandas as pd
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Bidirectional, Embedding, concatenate
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras.models import load_model
from keras import Input, Model
from utils import *
from rnn_utils import *
from attention import Attention
import time

start = time.time()
train_df = pd.read_csv('resample_data_1000000.csv')
end = time.time()
print(end-start)
train_df['impressions'] = train_df['impressions'].astype(str)
train_df['impressions'] = train_df['impressions'].map(lambda x: x.lstrip('[').rstrip(']'))

impressions_df = pd.DataFrame(train_df.impressions.str.split('|').tolist())

train_df.drop(columns=['impressions'], inplace = True)


train_df['prices'] = train_df['prices'].astype(str)

prices_df = pd.DataFrame(train_df.prices.str.split('|').tolist())

train_df = pd.concat([train_df, impressions_df], axis=1)
cols = train_df.columns.to_list()
cols[11:] = ['impressions_'+str(i) for i in range(25)]
train_df.columns = cols

city_df = train_df['city'].unique()
hotel_df = impressions_df.dropna()
hotel_df = hotel_df.values.flatten()
hotel_df = pd.DataFrame(hotel_df, columns =['item_id'])['item_id'].unique()


city_hotel = dict()
for city in city_df:
    hotels = []
    filtered_df = train_df[train_df.city == city]
    for i in range(filtered_df.shape[0]):
        for j in range(25):
            h = filtered_df.iloc[i]['impressions_'+str(j)]
            if (h != None and h != 'nan' and (h not in hotels)):
                hotels.append(int(h))
            else:
                break
    print(city)
    city_hotel[city] = hotels

hotel_city = {}
for t in city_hotel.items():
    city = t[0]
    for hotel in t[1]:
        hotel_city[hotel] = city


hotel_price = {}
not_nan_impressions= impressions_df[impressions_df[0] != 'nan'].index     
for idx in not_nan_impressions:
    for j in range(25):
        h = train_df.iloc[idx]['impressions_'+str(j)]
        if (h not in hotel_price.keys() and h != None):
            hotel_price[int(h)] = int(prices_df.iloc[idx][j])
            #print(h)


last_clicked_item = []
for idx in not_nan_impressions:
    for j in range(25):
        h = train_df.iloc[idx]['impressions_'+str(j)]
        if (h == None or j == 24):
            last_clicked_item.append(int(train_df.iloc[idx]['impressions_'+str(j-1)]))
            break


city_index = {city_df[i]:i for i in range(len(city_df))}

embedding_model = load_model('hotel_embedding.h5')
hotel_weights = extract_weights('hotel_embedding', embedding_model)

def catch(func, handle=lambda e : e, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return False

def predict_with_only_embeddings():
    
    last_impression_item = []
    for idx in not_nan_impressions:
        for j in range(25):
            h = train_df.iloc[idx]['impressions_'+str(j)]
            if ((h == None or j == 23) and j > 1 ):
                last_impression_item.append(int(train_df.iloc[idx]['impressions_'+str(j-2)]))
                break
            elif(h == None and j == 1):
                last_impression_item.append(int(train_df.iloc[idx]['impressions_'+str(0)]))
                break
     
    hotel_index, index_hotel, cols, properties_index = load_dictionaries()
    
    dicts = {'hotel_index': hotel_index,
             'index_hotel': index_hotel,
             'cols' : cols,
             'properties_index' : properties_index}
    
    submissions = []
    for item in last_impression_item:
        try:
            dists, closest_idx, closest_hotels = find_similar(dicts,item,hotel_weights,return_dist=True)
            closest_hotels_dists = {hotel:dists[hotel_index[hotel]] for hotel in closest_hotels}
            same_city_hotels= city_hotel[hotel_city[item]]
            similarities = {h:closest_hotels_dists[h] for h in same_city_hotels}
            sorted_hotels = sorted(similarities, key=similarities.get)[::-1]
        
        except:
            sorted_hotels = [item]
        print(last_impression_item.index(item))
        submissions.append(sorted_hotels)
    
    mrr_f = mrr(last_clicked_item, submissions)
    print(mrr_f)
    return mrr_f

def predict_with_embeddings_average():
    
    not_nan_impressions_seq = impressions_df[impressions_df[0] != 'nan']
    seq_avgs = []
    for i in range(not_nan_impressions_seq.shape[0]):# for each sample
        seq_avg = np.zeros(hotel_weights[5101].shape)
        not_null_count = not_nan_impressions_seq.iloc[i][:].count()-1
        
        for j in range(not_null_count):# average all clicks in the sessions
            try:
                seq_avg += hotel_weights[hotels_to_index[int(not_nan_impressions_seq.iloc[i][j])]]
            except:
                print(not_nan_impressions_seq.iloc[i][j])
                pass
        seq_avgs.append(seq_avg)
    
    hotel_index, index_hotel, cols, properties_index = load_dictionaries()
    
    dicts = {'hotel_index': hotel_index,
             'index_hotel': index_hotel,
             'cols' : cols,
             'properties_index' : properties_index}
    
    submissions = []
    i = 0
    for item in seq_avgs:
        try:
            dists, closest_idx, closest_hotels = find_similar(dicts,-1,hotel_weights,index_name = 'vector', return_dist=True, vec=item)
            closest_hotels_dists = {hotel:dists[hotel_index[hotel]] for hotel in closest_hotels}
            same_city_hotels= city_hotel[hotel_city[int(not_nan_impressions_seq.loc[not_nan_impressions[i]][0])]]
            similarities = {h:closest_hotels_dists[h] for h in same_city_hotels}
            sorted_hotels = sorted(similarities, key=similarities.get)[::-1]
        
        except:
            sorted_hotels = [5101] #to avoid nan issues
        submissions.append(sorted_hotels)
        print(str(i))
        i+=1

    mrr_avg = mrr(last_clicked_item, submissions)
    print(mrr_avg)
    return mrr_avg


def multi_input_one_output_RNN(seq_length, embedding_size, input_dim, num_class):
    seq_input = Input(shape=(seq_length,)) 
    meta_input = Input(shape=(1,))
    emb = Embedding(hotel_weights.shape[0],30, weights=[hotel_weights],
                            input_length=24,trainable=False)(seq_input) 
    seq_out = Bidirectional(LSTM(24,return_sequences=False))(emb) 
    concat = concatenate([seq_out, meta_input]) 
    classifier = Dense(36, activation='relu')(concat) 
    output = Dense(num_class, activation='softmax')(classifier) 
    model = Model(inputs=[seq_input , meta_input], outputs=[output])
    return model

def multi_input_one_output_with_attention(seq_length, embedding_size, input_dim, num_class):
    seq_input = Input(shape=(seq_length,)) 
    meta_input = Input(shape=(1,))
    emb = Embedding(hotel_weights.shape[0],30, weights=[hotel_weights],
                            input_length=24,trainable=False)(seq_input) 
    preattention = Bidirectional(LSTM(24,return_sequences=True))(emb) 
    seq_out = Attention()(preattention)
    concat = concatenate([seq_out, meta_input]) 
    classifier = Dense(36, activation='relu')(concat) 
    output = Dense(num_class, activation='softmax')(classifier) 
    model = Model(inputs=[seq_input , meta_input], outputs=[output])
    return model

def hotel_idx_seq(x):
    if x != None:
        try:
            return hotel_index[int(x)] 
        except:
            return 0
    else:
        return np.nan


def sample_city(x):
    if x != None:
        try:
            return city_index[hotel_city[int(x)]] 
        except:
            return np.nan
    else:
        return np.nan


def train_test_split():
    not_nan_impressions_seq = impressions_df[impressions_df[0] != 'nan']
    not_nan_impressions_seq_idx = not_nan_impressions_seq.applymap(hotel_idx_seq)
    meta_input = pd.DataFrame(not_nan_impressions_seq[0]).applymap(sample_city)

    X = []
    y = []
    meta = []
    for i in range(not_nan_impressions_seq_idx.shape[0]):# for each sample
        not_null_count = not_nan_impressions_seq_idx.iloc[i][:].count()-1
          
        if(not_null_count > 0):
            y.append(not_nan_impressions_seq_idx.iloc[i][not_null_count])
            seq = list(not_nan_impressions_seq_idx.iloc[i][:not_null_count].values)
            
            if(not_null_count < 24):
                seq = seq + [0]*(24-not_null_count)
        
            X.append(np.asarray(seq))
            meta.append(meta_input.iloc[i])
    
    X = np.stack(X,axis=0)
    size = X.shape[0]
    
    test_size = int(size * 0.2)
    
    X_train = X[:size-test_size]
    X_test = X[-test_size:]
    
    y_train = np.array(y[:size-test_size]).reshape(-1,1)
    y_test = np.array(y[-test_size:]).reshape(-1,1)
    
    meta_input_train = np.array(meta[:X_train.shape[0]])
    meta_input_test = np.array(meta[X_train.shape[0]:])


    return X_train, y_train, X_test, y_test, meta_input_train, meta_input_test

hotel_index, index_hotel, cols, properties_index = load_dictionaries()

X_train, y_train, X_test, y_test, meta_input_train, meta_input_test = train_test_split()

model = multi_input_one_output_with_attention(24, 30, 30, hotel_weights.shape[0])
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=100)]
)

model.summary()

model.fit([X_train,meta_input_train],y_train,batch_size=8,epochs=3)
model.save('./lstm_with_50000_sample_attention_batch8.h5')
model= load_model('./lstm_with_50000_sample_attention_batch8.h5')
preds = model.predict([X_test,meta_input_test],batch_size=8)

lstm_mrr = mrr(y_test,preds,meta_input_test)

###100000 with
model = multi_input_one_output_RNN(24, 30, 30, hotel_weights.shape[0])
model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=[tf.keras.metrics.TopKCategoricalAccuracy(k=100)]
)

model.summary()

model.fit([X_train,meta_input_train],y_train,batch_size=8,epochs=5)
model.save('./lstm_with_50000_sample_batch8.h5')
del model

model= load_model('./lstm_with_50000_sample_batch8.h5')
preds = model.predict([X_test,meta_input_test],batch_size=8)

lstm_mrr_2 = mrr(y_test,preds,meta_input_test)