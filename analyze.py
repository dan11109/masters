
from sklearn.cluster import DBSCAN
import numpy as np
import json 
import pickle
import math
import re
import random
from collections import Counter
from build_index import Index
from scipy import spatial
import time



with open('outputSets.pkl', 'rb') as inp:
    lsts = pickle.load(inp)

with open('cosineclusters.pkl', 'rb') as inp:
    clusters = pickle.load(inp)


f = open('data/info.json')
info = json.load(f)

# BERT COSINE: 2
# TFIDF COSINE: 0  

tmp = lsts[0].intersection(lsts[2])

only_bert = lsts[2] - tmp


print("Clusters found in Cosine BERT but not in Cosine tfidf:")

with open('embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_order = stored_data['order']
        stored_embeddings = stored_data['embeddings']


embedded_dict = {}
temp_order = []
temp_embeddings = []

for i in range(2000):  
    embedded_dict[stored_order[i]] = stored_embeddings[i]
    temp_order.append(stored_order[i])
    temp_embeddings.append(stored_embeddings[i])



only_bert_clust = []
cosines = []

for i in clusters[1]:
    tmp = []
    count = 0
    for j in i:
        if(j in only_bert):
            count += 1
            tmp.append(j)

    if(count > 1):
         only_bert_clust.append(tmp)

for i in only_bert_clust:
    max_cos = 0.0
    for j in range(len(i)-1):
        d = (1-spatial.distance.cosine(embedded_dict[i[0]], embedded_dict[i[j+1]] ))
        if(d > max_cos):
            max_cos = d

    cosines.append((max_cos,i))

for i in sorted(cosines):
    print(i)




'''
out_string = ''
for i in clusters[1]:
    tmp = ''
    count = 0
    for j in i:
        if(j in only_bert):
            count += 1
            tmp += '\n' + info[j]['title'] + '\n\t' + info[j]['url']
    if(count > 1):
        out_string += '\n\n' + tmp        
            
print(out_string)       

'''


