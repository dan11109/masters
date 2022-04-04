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



# ssh steved7@acidburn.cs.rpi.edu


def KL(vec1, vec2):
	p = np.asarray(vec1)
	q = np.asarray(vec2)
	p  = p  / np.linalg.norm(p )
	q = q / np.linalg.norm(q)
	epsilon = 0.00001
	p = p+epsilon
	q = q+epsilon
	divergence = np.sum(p*np.log(p/q))
	return divergence 





f = open('data/info.json')
info = json.load(f)

with open('all_data.pkl', 'rb') as inp:
    inn = pickle.load(inp)

with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']



temp_order = []
temp_embeddings = []
rand_idxs = random.sample(range(0, 2000), 1000)
for i in rand_idxs: #filter out 
	temp_order.append(stored_order[i])
	temp_embeddings.append(stored_embeddings[i])

stored_order = temp_order
stored_embeddings = temp_embeddings




clustering = DBSCAN(eps=.30, min_samples=10, metric='cosine').fit(stored_embeddings)


lst = clustering.labels_

#print(clustering.labels_)
print(max(clustering.labels_))
print(np.count_nonzero(clustering.labels_ != -1))
print(np.count_nonzero(clustering.core_sample_indices_ != -1))
	


clusters = {}

for i in range(len(clustering.labels_)):
	temp = clustering.labels_[i]
	if(temp != -1):
		if(temp in clusters.keys()):
			clusters[temp].append(stored_order[i])
		else:
			clusters[temp] = [stored_order[i]]




tfidf = {}
for word in inn.doc_sim_score.keys(): 
	
	count = 0
	for doc in inn.doc_sim_score[word]:
		if(doc[1] != 0):
			count += 1

	if(count <= 1): #filter out single words 
		continue

	for doc in inn.doc_sim_score[word]:
		if(doc[0] in tfidf.keys()): 
			tfidf[doc[0]].append(doc[1])
		else:
			tfidf[doc[0]]= [doc[1]]



##########COSINE##############
start = time.time()
# [(d1,d1)] = cos#
set_cos = set()
d = {}
number = 0
clusters_cos = []

for i in clusters.keys():

	tmp = clusters[i].copy()
	while(len(tmp) > 1):

		idx = random.randrange(0,len(tmp))
		cent = tmp.pop(idx)
		j = 0
		first = True
		while(j < len(tmp)):
			dist = (1-spatial.distance.cosine(tfidf[cent], tfidf[tmp[j]] ))
			if(dist > .8):
				d[(cent,tmp[j])] = dist
				set_cos.add(tmp[j])
				if(first):
					clusters_cos.append([cent])
					set_cos.add(cent)
					number += 1
				first = False
				clusters_cos[-1].append(tmp[j])
				tmp.pop(j)
				number+=1

			else:
				j+=1
end = time.time()
print("Time for Cosine: " + str(end - start))
print("Number of articles flagged (cos): " + str(number))

for i in d.keys():
	print(i,end=': ')
	print(d[i])

print()
for i in clusters_cos:	
	print(i)

sources = set()
for i in set_cos:
	sources.add(info[i]['source'])

print("Number of sources (COS): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(clusters_cos))


##############KL##############
start = time.time()
counter = 0
d = {}
number = 0
set_kl = set()
for i in clusters.keys():

	tmp = clusters[i].copy()
	while(len(tmp) > 1):

		idx = random.randrange(0,len(tmp))
		cent = tmp.pop(idx)
		j = 0
		first = True
		while(j < len(tmp)):
			dist = KL(tfidf[cent], tfidf[tmp[j]])
			if(dist < 8.0):
				d[(cent,tmp[j])] = dist
				set_kl.add(tmp[j])
				tmp.pop(j)
				if(first):
					set_kl.add(cent)
					number += 1
					counter += 1
				first = False
				number+=1

			else:
				j+=1
end = time.time()
print("Time for KL: " + str(end - start))
print("Number of articles flagged (kl): " + str(number))
print("Number in commom KL and Cosine: " + str(len(set_cos.intersection(set_kl))))


for i in d.keys():
	print(i,end=': ')
	print(d[i])
	

sources = set()
for i in set_kl:
	sources.add(info[i]['source'])

print("Number of sources (KL): ", end = '')
print(len(sources))		

print("Number of clusters: ",end ='')
print(counter)


##########BASELINE COSINE##############

base_clust = []
base_set = set()
start = time.time()
tmp = stored_order.copy()
while(len(tmp) > 1):

	idx = random.randrange(0,len(tmp))
	cent = tmp.pop(idx)
	j = 0
	first = True
	while(j < len(tmp)):
		dist = (1-spatial.distance.cosine(tfidf[cent], tfidf[tmp[j]] ))
		if(dist > .8):
			if(first):
				base_clust.append([cent])
				base_set.add(cent)
				number += 1
			first = False
			base_clust[-1].append(tmp[j])
			base_set.add(tmp[j])
			tmp.pop(j)
			number+=1

		else:
			j+=1

end = time.time()
print("Time for bruit force: " + str(end - start))
print("Number of articles flagged (BASELINE): " + str(number))

print()
for i in base_clust:	
	print(i)



print("Number in Cosine and BASELINE: " + str(len(set_cos.intersection(set(base_set)))))

sources = set()
for i in base_set:
	sources.add(info[i]['source'])

print("Number of sources (BASELINE): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(base_clust))


