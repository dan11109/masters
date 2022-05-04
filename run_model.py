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
import csv 
import sys

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


#load the data
f = open('data/info.json')
info = json.load(f)

with open('all_data.pkl', 'rb') as inp:
    inn = pickle.load(inp)

with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']



#sample n from the articles dataset 
embedded_dict = {}
temp_order = []
temp_embeddings = []
n = int(sys.argv[1]) # 1000

random.seed(10)
rand_idxs = random.sample(range(0, len(stored_embeddings)), n)
for i in rand_idxs: #filter out 
	embedded_dict[stored_order[i]] = stored_embeddings[i]
	temp_order.append(stored_order[i])
	temp_embeddings.append(stored_embeddings[i])

stored_order = temp_order
stored_embeddings = temp_embeddings

#cluster with BERT embedding 
clustering = DBSCAN(eps=.30, min_samples=7, metric='cosine').fit(stored_embeddings)

lst = clustering.labels_


clusters = {}
for i in range(len(clustering.labels_)):
	temp = clustering.labels_[i]
	if(temp != -1):
		if(temp in clusters.keys()):
			clusters[temp].append(stored_order[i])
		else:
			clusters[temp] = [stored_order[i]]

#generate tfidf vectors 
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



#---------- cluster ----------
if(len(sys.argv) == 6 and sys.argv[5] == 'cluster'): 
	start = time.time()
	set5 = set()
	d = {}
	number = 0
	final_clusters = []

	for i in clusters.keys():
		tmp = clusters[i].copy()
		while(len(tmp) > 1):
			idx = random.randrange(0,len(tmp))
			cent = tmp.pop(idx)
			j = 0
			first = True
			while(j < len(tmp)):

				if(sys.argv[3] == 'tfidf'):
					if(sys.argv[2] == 'cosine'):
						dist = (1-spatial.distance.cosine(tfidf[cent], tfidf[tmp[j]]))
						threshold = dist > .8
					else: #KL
						dist = KL(tfidf[cent], tfidf[tmp[j]])
						threshold = dist < 10.0
				else: #BERT
					if(sys.argv[2] == 'cosine'):
						dist = (1-spatial.distance.cosine(embedded_dict[cent], embedded_dict[tmp[j]]))
						threshold = dist > .8
					else: #KL
						dist = KL(embedded_dict[cent], embedded_dict[tmp[j]])
						threshold = dist < 10.0

				if(threshold):
					d[(cent,tmp[j])] = dist
					set5.add(tmp[j])
					if(first):
						final_clusters.append([(cent,1)])
						set5.add(cent)
						number += 1
					first = False
					final_clusters[-1].append((tmp[j],dist))
					tmp.pop(j)
					number+=1

				else:
					j+=1
	end = time.time()
	print()
	print("Time in seconds: " + str(end - start))
	print("Number of articles flagged: " + str(number))
	print(len(final_clusters))

#---------- no cluster ----------
elif(len(sys.argv) == 5):
	
	number = 0
	final_clusters = []
	set1 = set()
	start = time.time()
	tmp = stored_order.copy()
	while(len(tmp) > 1):

		idx = random.randrange(0,len(tmp))
		cent = tmp.pop(idx)
		j = 0
		first = True
		while(j < len(tmp)):

			if(sys.argv[3] == 'tfidf'):
				if(sys.argv[2] == 'cosine'):
					dist = (1-spatial.distance.cosine(tfidf[cent], tfidf[tmp[j]]))
					threshold = dist > .8
				else: #KL
					dist = KL(tfidf[cent], tfidf[tmp[j]])
					threshold = dist < 10.0
			else: #BERT
				if(sys.argv[2] == 'cosine'):
					dist = (1-spatial.distance.cosine(embedded_dict[cent], embedded_dict[tmp[j]]))
					threshold = dist > .8
				else: #KL
					dist = KL(embedded_dict[cent], embedded_dict[tmp[j]])
					threshold = dist < 10.0

			if(threshold):
				if(first):
					final_clusters.append([(cent, 1.0)])
					set1.add(cent)
					number += 1
				first = False
				final_clusters[-1].append( (tmp[j], dist) )
				set1.add(tmp[j])
				tmp.pop(j)
				number+=1

			else:
				j+=1

	end = time.time()

	print("Time in seconds: " + str(end - start))
	print("Number of articles flagged: " + str(number))
	
	print(len(final_clusters))


fields = ['Cluster', 'Title', 'Score', 'url', 'Source'] 
rows = []
num = 0
for c in final_clusters:
	for art in c:
		rows.append([num, info[art[0]]['title'], art[1], info[art[0]]['url'], info[art[0]]['source'] ])
	num += 1

out_file = sys.argv[4]

with open(out_file, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)


