# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 18:13:40 2021

@author: bkorkmaz
"""

from collections import Counter
import numpy as np 
import matplotlib.pyplot as plt  

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['font.size'] = 15

hotel_city_count = pd.DataFrame.from_dict(Counter(hotel_city.values()), orient='index', columns=['count'])
indices = hotel_city_count.index.to_list()

counts = {idx:hotel_city_count.loc[idx].values[0] for idx in indices}
counts = dict(sorted(counts.items(), key=lambda item: item[1]))

hotels = list(counts.keys())[-100:] 
values = list(counts.values())[-100:] 
   
fig = plt.figure(figsize = (20, 32)) 
  
# creating the bar plot 
plt.barh(hotels, values, color = 'g') 
  
plt.xlabel("No. of hotels") 
plt.ylabel("Cities") 
plt.title("Hotels Distribution by Cities") 
plt.tight_layout()
plt.show() 


#####price dist
import seaborn as sns


hotel_price_df = pd.DataFrame.from_dict(hotel_price, orient='index', columns = ['price'])
hotel_city_df = pd.DataFrame.from_dict(hotel_city, orient='index', columns = ['city'])
hotel_price_city = pd.concat([hotel_price_df,hotel_city_df],axis=1)
sns.set() 
plt.figure(figsize=(20,20))
sns.displot(hotel_price_city, x="price", hue="city")
plt.ylim(0, None)
plt.xlim(0, 600)


