# -*- coding: utf-8 -*-
"""
Created on Sat Jan  2 13:43:17 2021

@author: bkorkmaz
"""

import pandas as pd
import numpy as np
import random
random.seed(100)

from keras.layers import Input, Embedding, Dot, Reshape, Dense
from keras.models import Model

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15


item_metadata = pd.read_csv('item_metadata.csv')


properties_df = pd.DataFrame(item_metadata.properties.str.split('|').tolist())

column_values = properties_df.values.ravel()
unique_values =  pd.unique(column_values)
cols = ['item_id'] + unique_values.tolist()

frame_list = []

for row in range(item_metadata.shape[0]):
    empty_row = dict.fromkeys(cols,0)
    empty_row['item_id'] = item_metadata['item_id'][row]
    for value in unique_values:
        if value in properties_df.loc[row].values:
            empty_row[value] = 1
        else:
            empty_row[value] = 0
    frame_list.append(empty_row)
    if(row%1000 == 0):
        print(str(row))

embedding_df = pd.DataFrame(frame_list)
embedding_df.to_csv('embedding_df.csv', index = False)

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
    
def generate_batch(pairs, n_positive = 30, negative_ratio = 1.0, classification = False):
    """Generate batches of samples for training"""
    batch_size = n_positive * (1 + negative_ratio)
    batch = np.zeros((batch_size, 3))
    
    # Adjust label based on task
    if classification:
        neg_label = 0
    else:
        neg_label = -1
    
    # This creates a generator
    while True:
        # randomly choose positive examples
        for idx, (hotel_id,prop) in enumerate(random.sample(pairs, n_positive)):
            batch[idx, :] = (hotel_id, properties_index[prop], 1)

        # Increment idx by 1
        idx += 1
        
        # Add negative examples until reach batch size
        while idx < batch_size:
            
            # random selection
            random_idx = random.randrange(embedding_df.shape[0])
            random_prop = random.randrange(len(cols))
            
            # Check to make sure this is not a positive example
            if (random_idx, random_prop) not in pairs_set:
                
                # Add to batch and increment index
                batch[idx, :] = (random_idx, random_prop, neg_label)
                idx += 1
                
        # Make sure to shuffle order
        np.random.shuffle(batch)
        yield {'hotel_id': batch[:, 0], 'property': batch[:, 1]}, batch[:, 2]

next(generate_batch(pairs, n_positive = 2, negative_ratio = 2))

def hotel_embedding_model(embedding_size = 30, classification = False):
    """Model to embed hotels and props using the functional API."""
    
    # Both inputs are 1-dimensional
    hotel_id = Input(name = 'hotel_id', shape = [1])
    prop = Input(name = 'property', shape = [1])
    
    # Embedding the hotel (shape will be (None, 1, 30))
    hotel_embedding = Embedding(name = 'hotel_embedding',
                               input_dim = embedding_df.shape[0],
                               output_dim = embedding_size)(hotel_id)
    
    # Embedding the prop (shape will be (None, 1, 30))
    prop_embedding = Embedding(name = 'prop_embedding',
                               input_dim = len(cols),
                               output_dim = embedding_size)(prop)
    
    # Merge the layers with a dot product along the second axis (shape will be (None, 1, 1))
    merged = Dot(name = 'dot_product', normalize = True, axes = 2)([hotel_embedding, prop_embedding])
    
    # Reshape to be a single number (shape will be (None, 1))
    merged = Reshape(target_shape = [1])(merged)
    
    # If classifcation, add extra layer and loss function is binary cross entropy
    if classification:
        merged = Dense(1, activation = 'sigmoid')(merged)
        model = Model(inputs = [hotel_id, prop], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    # Otherwise loss function is mean squared error
    else:
        model = Model(inputs = [hotel_id, prop], outputs = merged)
        model.compile(optimizer = 'Adam', loss = 'mse')
    
    return model

# Instantiate model and show parameters
model = hotel_embedding_model()
model.summary()

n_positive = 1024

gen = generate_batch(pairs, n_positive, negative_ratio = 2)

# Train
h = model.fit_generator(gen, epochs = 15, 
                        steps_per_epoch = len(pairs) // n_positive,
                        verbose = 2)

model.save('./hotel_embedding.h5')

hotel_layer = model.get_layer('hotel_embedding')
hotel_weights = hotel_layer.get_weights()[0]
hotel_weights.shape

hotel_weights = hotel_weights / np.linalg.norm(hotel_weights, axis = 1).reshape((-1, 1))
hotel_weights[0][:10]
np.sum(np.square(hotel_weights[0]))


def find_similar(hotel_id, weights, index_name = 'hotel_id', n = 10, least = False, return_dist = False, plot = False):
    """Find n most similar items (or least) to name based on embeddings. Option to also plot the results"""
    
    # Select index and reverse index
    if index_name == 'hotel_id':
        index = hotel_index
        rindex = index_hotel
    elif index_name == 'prop':
        index = properties_index
        rindex = cols
    
    # Check to make sure `name` is in index
    try:
        # Calculate dot product between book and all others
        dists = np.dot(weights, weights[hotel_index[hotel_id]])
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
        name_str = f'{index_name.capitalize()}s Most and Least Similar to'
        for word in str(hotel_id).split():
            # Title uses latex for italize
            name_str += ' $\it{' + word + '}$'
        plt.title(name_str, x = 0.2, size = 28, y = 1.05)
        
        return None
    
    # If specified, find the least similar
    if least:
        # Take the first n from sorted distances
        closest = sorted_dists[:n]
         
        print(f'{index_name.capitalize()}s furthest from {hotel_id}.\n')
        
    # Otherwise find the most similar
    else:
        # Take the last n sorted distances
        closest = sorted_dists[-n:]
        
        # Need distances later on
        if return_dist:
            return dists, closest
        
        
        print(f'{index_name.capitalize()}s closest to {hotel_id}.\n')
        
    # Need distances later on
    if return_dist:
        return dists, closest
    
    
    # Print formatting
    max_width = max([len(str(rindex[c])) for c in closest])
    
    # Print the most similar and distances
    for c in reversed(closest):
        print(f'{index_name.capitalize()}: {rindex[c]:{max_width + 2}} Similarity: {dists[c]:.{2}}')
        
find_similar(5101, hotel_weights)

def extract_weights(name, model):
    """Extract weights from a neural network model"""
    
    # Extract weights
    weight_layer = model.get_layer(name)
    weights = weight_layer.get_weights()[0]
    
    # Normalize
    weights = weights / np.linalg.norm(weights, axis = 1).reshape((-1, 1))
    return weights

model = load_model('hotel_embedding.h5')
hotel_weights = extract_weights('hotel_embedding', model)
prop_weights = extract_weights('prop_embedding', model)

#from tensorflow.keras.models import load_model
#modeldeneme = load_model('hotel_embedding.h5')
#hoteldeneme_weights = extract_weights('hotel_embedding', modeldeneme)

from umap import UMAP

def reduce_dim(weights, components = 3, method = 'tsne'):
    """Reduce dimensions of embeddings"""
    if method == 'tsne':
        return TSNE(components, metric = 'cosine').fit_transform(weights)
    elif method == 'umap':
        # Might want to try different parameters for UMAP
        return UMAP(n_components=components, metric = 'cosine', 
                    init = 'random', n_neighbors = 5).fit_transform(weights)

def plot_closest(item, weights, index_name, n, plot_data):
    """Plot n most closest items to item"""
    
    # Find the closest items
    dist, closest = find_similar(item, weights, index_name, n, return_dist=True)
    
    # Choose mapping for look up
    if index_name == 'hotel':
        index = hotel_index
        rindex = index_hotel
    elif index_name == 'prop':
        index = properties_index
        rindex = cols

    plt.figure(figsize = (10, 9))
    plt.rcParams['font.size'] = 14
    
    # Limit distances
    dist = dist[closest]
    
    # Plot all of the data
    plt.scatter(plot_data[:, 0], plot_data[:, 1], alpha = 0.1, color = 'goldenrod')
    
    # Plot the item
    plt.scatter(plot_data[closest[-1], 0], plot_data[closest[-1], 1], s = 600, edgecolor = 'k', color = 'forestgreen')
    
    # Plot the closest items
    p = plt.scatter(plot_data[closest[:-1], 0], plot_data[closest[:-1], 1], 
                c = dist[:-1], cmap = plt.cm.RdBu_r, s = 200, alpha = 1, marker = '*')
    
    # Colorbar management
    cbar = plt.colorbar()
    cbar.set_ticks([])
    
    tick_labels = []
    # Tick labeling for colorbar
    for idx, distance in zip(closest[:-1], dist[:-1]):
        name_str = str (rindex[idx])
        name_str += ': ' + str(round(distance, 2))
        tick_labels.append(name_str)
    
    for j, lab in enumerate(tick_labels):
        cbar.ax.text(1, (2 * j + 1) / ((n - 1) * 2), lab, ha='left', va='center', size = 12)
    cbar.ax.set_title(f'{index_name.capitalize()} and Cosine Distance', loc = 'left', size = 14)
    
    # Formatting for italicized title
    name_str = f'{index_name.capitalize()}s Most Similar to'
    word = str(item)
    name_str += ' $\it{' + word + '}$'
    
    # Labeling
    plt.xlabel('TSNE 1'); plt.ylabel('TSNE 2'); 
    plt.title(name_str);
    
embedding_df = pd.read_csv('embedding_df.csv')
count_props = embedding_df.sum(axis = 0, skipna = True)
indices = count_props.index.to_list()[1:11]

counts = {idx:count_props[idx] for idx in indices}
all_indices = count_props.index.to_list()
del all_indices[all_indices == 'item_id']
all_indices

ints, inds = pd.factorize(all_indices)
inds[:5]

hotel_ru = reduce_dim(hotel_weights, components = 2, method = 'umap')
plot_closest(5101, hotel_weights, 'hotel', 10, hotel_ru)

plt.figure(figsize = (10, 8))

# Plot embedding
plt.scatter(hotel_ru[:, 0], hotel_ru[:, 1], c = ints, cmap = plt.cm.tab10)

# Add colorbar and appropriate labels
cbar = plt.colorbar()
cbar.set_ticks([])
for j, lab in enumerate(inds):
    cbar.ax.text(1, (2 * j + 1) / ((10) * 2), lab, ha='left', va='center')
cbar.ax.set_title('Property', loc = 'left')


plt.xlabel('UMAP 1'); plt.ylabel('UMAP 2'); plt.title('UMAP Visualization of Hotel Embeddings');



