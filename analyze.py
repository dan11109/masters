
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

'''
for i in only_bert:

    print(info[i]['title'])
    print(info[i]['url'])
    print()
'''
print("Clusters found in Cosine BERT but not in Cosine tfidf:")


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
            '''
            print(info[j]['title'])
            print('',end='\t')
            print(info[j]['url'])
            '''
print(out_string)       




