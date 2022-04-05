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
#import plotly.express as px
from sklearn.decomposition import PCA
import pandas as pd 


with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']


clustering = DBSCAN(eps=.30, min_samples=10, metric='cosine').fit(stored_embeddings)


clusters = {}
vects = []
for i in range(len(clustering.labels_)):
	temp = clustering.labels_[i]
	if(temp != -1):
		vects.append(stored_embeddings[i])
		if(temp in clusters.keys()):
			clusters[temp].append(stored_order[i])
		else:
			clusters[temp] = [stored_order[i]]

df = pd.DataFrame(vects) 

print(df)
print(len(stored_embeddings[0]))
