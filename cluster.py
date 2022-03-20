
from sklearn.cluster import DBSCAN
import numpy as np
import json 
import pickle
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial


# ssh steved7@acidburn.cs.rpi.edu

with open('all_data.pkl', 'rb') as inp:
    inn = pickle.load(inp)

with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']


clustering = DBSCAN(eps=.83, min_samples=10).fit(stored_embeddings)


lst = clustering.labels_

#print(clustering.labels_)
#print(max(clustering.labels_))
#print(np.count_nonzero(clustering.labels_ == -1))
#print( np.count_nonzero(clustering.core_sample_indices_ != -1))

clusters = {}

for i in range(len(clustering.labels_)):
	temp = clustering.labels_[i]
	if(temp != -1):
		if(temp in clusters.keys()):
			clusters[temp].append(stored_order[i])
		else:
			clusters[temp] = [stored_order[i]]





#store 
#with open('clusters.pkl', 'wb') as outp:
#    pickle.dump(clusters, outp, pickle.HIGHEST_PROTOCOL)

tfidf = {}
for word in inn.doc_sim_score.keys(): #sorted(inn.doc_sim_score.keys()):
	
	count = 0
	for doc in inn.doc_sim_score[word]:
		if(doc[1] != 0):
			count += 1

	if(count <= 1): #filter out 
		continue

	for doc in inn.doc_sim_score[word]:
		if(doc[0] in tfidf.keys()): 
			tfidf[doc[0]].append(doc[1])
		else:
			tfidf[doc[0]]= [doc[1]]



centers = [0] * len(clusters.keys())

for i in clusters.keys():

	for p in clustering.core_sample_indices_:
		if(stored_order[p] in clusters[i]):
			centers[i] = stored_order[p]
			break

counter = 0
clust = clusters[3]
for pt in clust:
	if(pt != centers[3]):
		counter+= 1
		print(((1-spatial.distance.cosine(tfidf[pt], tfidf[centers[3]])), pt) )
print(couter)




dists = [[]] * len(clusters.keys())

for i in clusters.keys():
	clust = clusters[i]
	
	for pt in clust:
		if(pt != centers[i]):
			dists[i].append(((1-spatial.distance.cosine(tfidf[pt], tfidf[centers[i]])), pt) )

			
'''
for d in dists:
	print()
	print("For center: " + str())
	print("Count: " + str(len(d)))
	for i in d:
		print('\t' + str(i[0]) + ' ' + str(i[1]))

'''


