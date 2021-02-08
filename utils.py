# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:15:26 2021

@author: bkorkmaz
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import csv
import h5py
import ast
import time


def save_obj(obj, name ):
    with open('obj/'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

def create_dictionaries():
    
    embedding_df = pd.read_csv('embedding_df.csv')
    item_id = embedding_df['item_id'].values
    cols =embedding_df.columns.to_list()[1:]
    properties = embedding_df[cols].values
    
    hotel_index = {item_id: idx for idx, item_id in enumerate(item_id)}
    index_hotel = {idx: item_id for item_id, idx in hotel_index.items()}
    index_properties = {idx: properties for idx, properties in  enumerate(properties)}
    properties_index = {prop:idx  for idx, prop in  enumerate(cols)}
    
    pairs = []
    #positive pairs
    for item in item_id:
        pairs.extend((hotel_index[item], cols[p]) for p in range(len(cols)) if (embedding_df.loc[hotel_index[item]][cols[p]] == 1))
        if(hotel_index[item] % 1000 == 0):
            print(str(hotel_index[item]))
    
    pairs_set = set(pairs)
    
    return pairs_set, hotel_index, index_hotel, cols, index_properties, properties_index

def save_dictionaries():

    pairs_set, hotel_index, index_hotel, cols, index_properties, properties_index = create_dictionaries()
    
    save_obj(hotel_index, 'hotel_index')
    save_obj(index_hotel, 'index_hotel')
    save_obj(index_properties, 'index_properties')
    save_obj(properties_index, 'properties_index')
    save_obj(pairs_set, 'pairs_set')
    
    return

def store_table(table,filename):
    
    with h5py.File('obj/'+ filename + '.h5', "w") as file:
        file.create_dataset('dataset_1', data=str(table))


def load_table(filename):
    file = h5py.File('obj/'+ filename + '.h5', "r")
    data = file.get('dataset_1')[...].tolist()
    file.close();
    return ast.literal_eval(data)



def load_dictionaries():
    
    hotel_index = load_obj('hotel_index')
    index_hotel = load_obj('index_hotel')
    properties_index = load_obj('properties_index')
    
    #store_table(hotel_index, 'hotel_index')
    #store_table(index_hotel, 'index_hotel')
    #store_table(properties_index, 'properties_index')
    
    #hotel_index = load_table('hotel_index')
    #index_hotel = load_table('index_hotel')
    #properties_index = load_table('properties_index')
    

    #index_properties = load_obj('index_properties')
    #pairs_set = load_obj('pairs_set')
    
    embedding_df = pd.read_csv('embedding_df.csv')
    item_id = embedding_df['item_id'].values
    cols =embedding_df.columns.to_list()[1:]
    
    #return pairs_set, hotel_index, index_hotel, cols, index_properties, properties_index
    return  hotel_index, index_hotel, cols, properties_index


def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # Normalize
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights



def find_similar(dicts, hotel_id, weights, index_name = 'hotel_id', n = 10, least = False, return_dist = False, plot = False, vec=None):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""

    hotel_index = dicts['hotel_index']
    index_hotel = dicts['index_hotel']
    cols = dicts['cols']
    properties_index = dicts['properties_index']
    
    # Select index and reverse index
    if index_name == 'hotel_id':
        index = hotel_index
        rindex = index_hotel
    elif index_name == 'prop':
        index = properties_index
        rindex = cols
    elif index_name == 'vector':
        index = hotel_index
        rindex = index_hotel
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        if(index_name != 'vector'):
            dists = np.dot(weights, weights[hotel_index[hotel_id]])
        else:
            dists = np.dot(weights, vec)
    except KeyError:
        print(f'{hotel_id} Not Found.')
        return
    
    # Sort distance indexes from smallest to largest
    sorted_dists = np.argsort(dists)
    
    # Plot results if specified
    if plot:
        
        # Find furthest and closest items
        furthest = sorted_dists[:(n // 2)]
        closest = sorted_dists[-n-1: len(dists) - 1]
        items = [rindex[c] for c in furthest]
        items.extend(rindex[c] for c in closest)
        
        # Find furthest and closets distances
        distances = [dists[c] for c in furthest]
        distances.extend(dists[c] for c in closest)
        
        colors = ['r' for _ in range(n //2)]
        colors.extend('g' for _ in range(n))
        
        data = pd.DataFrame({'distance': distances}, index = items)
        
        # Horizontal bar chart
        data['distance'].plot.barh(color = colors, figsize = (10, 8),
                                   edgecolor = 'k', linewidth = 2)
        plt.xlabel('Cosine Similarity');
        plt.axvline(x = 0, color = 'k');
        
        # Formatting for italicized title
        if(index_name != 'vector'):
            name_str = f'{index_name.capitalize()}s Most and Least Similar to'
            for word in str(hotel_id).split():
                # Title uses latex for italize
                name_str += ' $\it{' + word + '}$'
        else:
            name_str = f'{index_name.capitalize()}s Most and Least Similar to given vector'
    
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
        
        if(index_name != 'vector'):
            print(f'{index_name.capitalize()}s furthest from {hotel_id}.\n')
        else:
            print(f'{index_name.capitalize()}s furthest from the given vector.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        #closest = sorted_dists[-n:]
        closest = sorted_dists[::-1]
        # Need distances later on
        if return_dist:
            closest_hotels = []
            for idx in closest:
                closest_hotels.append(rindex[idx])
            
            return dists, closest , closest_hotels    
        
        if(index_name != 'vector'):
            print(f'{index_name.capitalize()}s closest to {hotel_id}.\n')
        else:
            print(f'{index_name.capitalize()}s closest to the given vector.\n')
       
        
    # Need distances later on
    if return_dist:
        closest_hotels = []
        for idx in closest:
            closest_hotels.append(rindex[idx])
        
        return dists, closest , closest_hotels
    
    
    # Print formatting
    max_width = max([len(str(rindex[c])) for c in closest])
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')

def mrr(clicked_items, submissions,meta_input_test):
    
    def sort(sub):
        return sorted(range(len(sub)), key=lambda k: sub[k], reverse = True)
            
    
    def convert_cities(sub, i):
        inds = []
        for s in range(len(sub)):
            try:
                idx = city_index[hotel_city[index_hotel[sub[s]]]]
                if(idx == meta_input_test[i]):
                    inds.append(s)                     
            except:
                pass #demek ki bizim sampleladigimiz otellerden biri degil                          
        return inds
    
    now = time.time()
    indices = []
    for i in range(0,len(submissions)):
        submissions[i] = np.array(sort(list(submissions[i])))
        indices.append(convert_cities(submissions[i], i))
        if(i%10==0):
            print('Done: ', (i/len(submissions))*100)
        
    last = time.time()
    print(last - now)
    
    rank = 0.0
    for i in range(len(clicked_items)):
        imp_submissions = submissions[i][indices[i]]
        for j in range(len(imp_submissions)):
            try:
                if(int(imp_submissions[j]) == int(clicked_items[i])):
                    rank += 1/j
                    break
            except:
                pass
        print(rank)
    
    mrr = rank/len(clicked_items)
    return mrr
