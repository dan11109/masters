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
	p  = p  / np.linalg.norm(p)
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


embedded_dict = {}
temp_order = []
temp_embeddings = []
n = 1000#sys.argv[1]
rand_idxs = random.sample(range(0, 2000), n)
for i in rand_idxs: #filter out 
	embedded_dict[stored_order[i]] = stored_embeddings[i]
	temp_order.append(stored_order[i])
	temp_embeddings.append(stored_embeddings[i])

stored_order = temp_order
stored_embeddings = temp_embeddings


clustering = DBSCAN(eps=.30, min_samples=7, metric='cosine').fit(stored_embeddings)


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



##### baseline cosine tf-idf #####
number = 0
base_clust = []
set1 = set()
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
				set1.add(cent)
				number += 1
			first = False
			base_clust[-1].append(tmp[j])
			set1.add(tmp[j])
			tmp.pop(j)
			number+=1

		else:
			j+=1

end = time.time()
print("Baseline cosine tf-idf: " + str(end - start))
print("Number of articles flagged (BASELINE): " + str(number))

sources = set()
for i in set1:
	sources.add(info[i]['source'])

print("Number of sources (BASELINE): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(base_clust))



##### baseline KL tf-idf #####
number = 0
base_clust = []
set2 = set()
start = time.time()
tmp = stored_order.copy()
while(len(tmp) > 1):

	idx = random.randrange(0,len(tmp))
	cent = tmp.pop(idx)
	j = 0
	first = True
	while(j < len(tmp)):
		dist = KL(tfidf[cent], tfidf[tmp[j]])
		if(dist < 10.0):
			if(first):
				base_clust.append([cent])
				set2.add(cent)
				number += 1
			first = False
			base_clust[-1].append(tmp[j])
			set2.add(tmp[j])
			tmp.pop(j)
			number+=1

		else:
			j+=1

end = time.time()
print()
print("Baseline KL tf-idf: " + str(end - start))
print("Number of articles flagged (BASELINE): " + str(number))

sources = set()
for i in set2:
	sources.add(info[i]['source'])

print("Number of sources (BASELINE): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(base_clust))


##### baseline cosine bert #####
number = 0
base_clust = []
set3 = set()
start = time.time()
tmp = stored_order.copy()
while(len(tmp) > 1):

	idx = random.randrange(0,len(tmp))
	cent = tmp.pop(idx)
	j = 0
	first = True
	while(j < len(tmp)):
		dist = (1-spatial.distance.cosine(embedded_dict[cent], embedded_dict[tmp[j]] ))
		if(dist > .8):
			if(first):
				base_clust.append([cent])
				set3.add(cent)
				number += 1
			first = False
			base_clust[-1].append(tmp[j])
			set3.add(tmp[j])
			tmp.pop(j)
			number+=1

		else:
			j+=1

end = time.time()
print()
print("Baseline cosine BERT: " + str(end - start))
print("Number of articles flagged (BASELINE): " + str(number))

sources = set()
for i in set3:
	sources.add(info[i]['source'])

print("Number of sources (BASELINE): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(base_clust))

##### baseline KL bert #####
number = 0
base_clust = []
set4 = set()
start = time.time()
tmp = stored_order.copy()
while(len(tmp) > 1):

	idx = random.randrange(0,len(tmp))
	cent = tmp.pop(idx)
	j = 0
	first = True
	while(j < len(tmp)):
		dist = KL(embedded_dict[cent], embedded_dict[tmp[j]])
		if(dist < 10.0):
			if(first):
				base_clust.append([cent])
				set4.add(cent)
				number += 1
			first = False
			base_clust[-1].append(tmp[j])
			set4.add(tmp[j])
			tmp.pop(j)
			number+=1

		else:
			j+=1

end = time.time()
print()
print("Baseline KL BERT: " + str(end - start))
print("Number of articles flagged (BASELINE): " + str(number))

sources = set()
for i in set4:
	sources.add(info[i]['source'])

print("Number of sources (BASELINE): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(base_clust))


##### cluster bert, cosine tf-idf ##### 
start = time.time()
set5 = set()
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
				set5.add(tmp[j])
				if(first):
					clusters_cos.append([cent])
					set5.add(cent)
					number += 1
				first = False
				clusters_cos[-1].append(tmp[j])
				tmp.pop(j)
				number+=1

			else:
				j+=1
end = time.time()
print()
print("Time for clustering with BERT, and Cosine TF-IDF: " + str(end - start))
print("Number of articles flagged (cos): " + str(number))

sources = set()
for i in set5:
	sources.add(info[i]['source'])

print("Number of sources (COS): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(clusters_cos))

##### cluster bert, KL tf-idf #####
start = time.time()
counter = 0
d = {}
number = 0
set6 = set()
for i in clusters.keys():

	tmp = clusters[i].copy()
	while(len(tmp) > 1):

		idx = random.randrange(0,len(tmp))
		cent = tmp.pop(idx)
		j = 0
		first = True
		while(j < len(tmp)):
			dist = KL(tfidf[cent], tfidf[tmp[j]])
			if(dist < 10.0):
				d[(cent,tmp[j])] = dist
				set6.add(tmp[j])
				tmp.pop(j)
				if(first):
					set6.add(cent)
					number += 1
					counter += 1
				first = False
				number+=1

			else:
				j+=1

end = time.time()
print()
print("Time for clustering with BERT, and KL TF-IDF: " + str(end - start))
print("Number of articles flagged (kl): " + str(number))
	
sources = set()
for i in set6:
	sources.add(info[i]['source'])

print("Number of sources (KL): ", end = '')
print(len(sources))		

print("Number of clusters: ",end ='')
print(counter)


##### cluster bert, cosine bert #####
start = time.time()
set7 = set()
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
			dist = (1-spatial.distance.cosine(embedded_dict[cent], embedded_dict[tmp[j]] ))
			if(dist > .8):
				d[(cent,tmp[j])] = dist
				set7.add(tmp[j])
				if(first):
					clusters_cos.append([cent])
					set7.add(cent)
					number += 1
				first = False
				clusters_cos[-1].append(tmp[j])
				tmp.pop(j)
				number+=1

			else:
				j+=1
end = time.time()
print()
print("Time for clustering with BERT, and Cosine BERT: " + str(end - start))
print("Number of articles flagged (cos): " + str(number))

sources = set()
for i in set7:
	sources.add(info[i]['source'])

print("Number of sources (COS): ", end = '')
print(len(sources))

print("Number of clusters: ",end ='')
print(len(clusters_cos))




##### cluster bert, KL bert #####
start = time.time()
counter = 0
d = {}
number = 0
set8 = set()
for i in clusters.keys():

	tmp = clusters[i].copy()
	while(len(tmp) > 1):

		idx = random.randrange(0,len(tmp))
		cent = tmp.pop(idx)
		j = 0
		first = True
		while(j < len(tmp)):
			dist = KL(embedded_dict[cent], embedded_dict[tmp[j]])
			if(dist < 10.0):
				d[(cent,tmp[j])] = dist
				set8.add(tmp[j])
				tmp.pop(j)
				if(first):
					set8.add(cent)
					number += 1
					counter += 1
				first = False
				number+=1

			else:
				j+=1

end = time.time()
print()
print("Time for clustering with BERT, and KL BERT: " + str(end - start))
print("Number of articles flagged (kl): " + str(number))
	
sources = set()
for i in set8:
	sources.add(info[i]['source'])

print("Number of sources (KL): ", end = '')
print(len(sources))		

print("Number of clusters: ",end ='')
print(counter)




print("Articles in common: ")
print()
print('Baseline cosine tf-idf:')
print(str(len(set1.intersection(set2))))
print(str(len(set1.intersection(set3))))
print(str(len(set1.intersection(set4))))
print(str(len(set1.intersection(set5))))
print(str(len(set1.intersection(set6))))
print(str(len(set1.intersection(set7))))
print(str(len(set1.intersection(set8))))

print()
print('Baseline KL tf-idf:')
print(str(len(set2.intersection(set1))))
print(str(len(set2.intersection(set3))))
print(str(len(set2.intersection(set4))))
print(str(len(set2.intersection(set5))))
print(str(len(set2.intersection(set6))))
print(str(len(set2.intersection(set7))))
print(str(len(set2.intersection(set8))))

print()
print('Baseline cosine BERT:')
print(str(len(set3.intersection(set1))))
print(str(len(set3.intersection(set2))))
print(str(len(set3.intersection(set4))))
print(str(len(set3.intersection(set5))))
print(str(len(set3.intersection(set6))))
print(str(len(set3.intersection(set7))))
print(str(len(set3.intersection(set8))))

print()
print('Baseline KL BERT:')
print(str(len(set4.intersection(set1))))
print(str(len(set4.intersection(set2))))
print(str(len(set4.intersection(set3))))
print(str(len(set4.intersection(set5))))
print(str(len(set4.intersection(set6))))
print(str(len(set4.intersection(set7))))
print(str(len(set4.intersection(set8))))

print()
print('Clustering with BERT, and Cosine TF-IDF:')
print(str(len(set5.intersection(set1))))
print(str(len(set5.intersection(set2))))
print(str(len(set5.intersection(set3))))
print(str(len(set5.intersection(set4))))
print(str(len(set5.intersection(set6))))
print(str(len(set5.intersection(set7))))
print(str(len(set5.intersection(set8))))

print()
print('Clustering with BERT, and KL TF-IDF:')
print(str(len(set6.intersection(set1))))
print(str(len(set6.intersection(set2))))
print(str(len(set6.intersection(set3))))
print(str(len(set6.intersection(set4))))
print(str(len(set6.intersection(set5))))
print(str(len(set6.intersection(set7))))
print(str(len(set6.intersection(set8))))

print()
print('Clustering with BERT, and Cosine BERT:')
print(str(len(set7.intersection(set1))))
print(str(len(set7.intersection(set2))))
print(str(len(set7.intersection(set3))))
print(str(len(set7.intersection(set4))))
print(str(len(set7.intersection(set5))))
print(str(len(set7.intersection(set6))))
print(str(len(set7.intersection(set8))))

print()
print('Clustering with BERT, and KL BERT:')
print(str(len(set8.intersection(set1))))
print(str(len(set8.intersection(set2))))
print(str(len(set8.intersection(set3))))
print(str(len(set8.intersection(set4))))
print(str(len(set8.intersection(set5))))
print(str(len(set8.intersection(set6))))
print(str(len(set8.intersection(set7))))


