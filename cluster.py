
from sklearn.cluster import DBSCAN
import numpy as np
import json 
import pickle


#X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
#clustering = DBSCAN(eps=10, min_samples=2).fit(X)

#print(clustering.labels_)

#print(clustering.n_features_in_)
#print(clustering.components_)
#print(clustering.core_sample_indices_)


with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']


clustering = DBSCAN(eps=.5, min_samples=10).fit(stored_embeddings)


lst = clustering.labels_

print(clustering.labels_)
print(max(clustering.labels_))
print(np.count_nonzero(clustering.labels_ == -1))

