
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

#print(clustering.labels_)
#print(max(clustering.labels_))
print(np.count_nonzero(clustering.labels_ == -1))
print( np.count_nonzero(clustering.core_sample_indices_ != -1))

clusters = {}

for i in range(len(clustering.labels_)):
	temp = clustering.labels_[i]
	if(temp != -1):
		if(temp in clusters.keys()):
			clusters[temp].append(stored_order[i])
		else:
			clusters[temp] = [stored_order[i]]


with open('clusters.pkl', 'wb') as outp:
    pickle.dump(clusters, outp, pickle.HIGHEST_PROTOCOL)
