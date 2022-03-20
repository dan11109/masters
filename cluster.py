
from sklearn.cluster import DBSCAN
import numpy as np
import json 
import pickle


# ssh steved7@acidburn.cs.rpi.edu


with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']


clustering = DBSCAN(eps=.83, min_samples=10).fit(stored_embeddings)


lst = clustering.labels_

print(clustering.labels_)
print(max(clustering.labels_))
print(np.count_nonzero(clustering.labels_ == -1))

clusters = {}

for i in clustering.labels_:
	if(i != -1):
		if(i in clusters.keys()):
			clusters[i] += 1
		else:
			clusters[i] = 1


print(clusters)
