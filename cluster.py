
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



##########COSINE##############
# [(d1,d1)] = cos#
set_cos = set()
d = {}
number = 0

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
				tmp.pop(j)
				if(first):
					set_cos.add(cent)
					number += 1
				first = False
				number+=1

			else:
				j+=1

print("Number of articles flagged (cos): " + str(number))

for i in d.keys():
	print(i,end=': ')
	print(d[i])
			

file = open("outClust.html","w")
file.write("<br />\n")

for i in d.keys():
	a = i[0]
	b = i[1]

	file.write(info[a]['title'])
	file.write("<br />\n")
	#file.write(info[a]['source'])
	#file.write("<br />\n")
	#file.write('<a href="' + info[a]['url'] + '"> Link </a>')
	#file.write("<br />\n")
	file.write(info[b]['title'])
	#file.write("<br />\n")
	#file.write(info[b]['source'])
	#file.write("<br />\n")
	#file.write('<a href="' + info[b]['url'] + '"> Link </a>')
	#file.write("<br />\n")
	#file.write('Cosine score: ' + str(d[i]))
	file.write("<br />\n")
	file.write("<br />\n")
	
file.close()


##############KL##############
# [(d1,d1)] = cos#
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
			if(dist < 4.0):
				d[(cent,tmp[j])] = dist
				set_kl.add(tmp[j])
				tmp.pop(j)
				if(first):
					set_kl.add(cent)
					number += 1
				first = False
				number+=1

			else:
				j+=1

print("Number of articles flagged (kl): " + str(number))
print("Number in commom KL and Cosine: " + str(len(set_cos.intersection(set_kl))))


for i in d.keys():
	print(i,end=': ')
	print(d[i])
			

file = open("outClustKL.html","w")
file.write("<br />\n")

for i in d.keys():
	a = i[0]
	b = i[1]

	file.write(info[a]['title'])
	file.write("<br />\n")
	file.write(info[a]['source'])
	file.write("<br />\n")
	file.write('<a href="' + info[a]['url'] + '"> Link </a>')
	file.write("<br />\n")
	file.write(info[b]['title'])
	file.write("<br />\n")
	file.write(info[b]['source'])
	file.write("<br />\n")
	file.write('<a href="' + info[b]['url'] + '"> Link </a>')
	file.write("<br />\n")
	file.write('KL score: ' + str(d[i]))
	file.write("<br />\n")
	file.write("<br />\n")
	

file.close()






















'''
# Old approach  
centers = [0] * len(clusters.keys())
for i in clusters.keys():

	for p in clustering.core_sample_indices_:
		if(stored_order[p] in clusters[i]):
			centers[i] = stored_order[p]
			break

dists = {}
for i in clusters.keys():
	clust = clusters[i]
	
	for pt in clust:
		if(pt != centers[i]):
			if(i in dists.keys()):
				dists[i].append(((1-spatial.distance.cosine(tfidf[pt], tfidf[centers[i]])), pt) )
			else:
				dists[i] = [((1-spatial.distance.cosine(tfidf[pt], tfidf[centers[i]])), pt)]		
'''



'''
for j in dists.keys():
	d = dists[j]
	print()
	print("For center: " + str(centers[j]))
	print("Count: " + str(len(d)))
	for i in sorted(d,reverse = True):
		print('\t' + str(i[0]) + ' ' + str(i[1]))
'''


'''
file = open("outClust.html","w")
file.write("<br />\n")

for j in dists.keys():
	d = dists[j]
	file.write("<br />\n")
	file.write("Center "+ centers[j] +": ")
	file.write(info[centers[j]]['title'])
	file.write("<br />\n")
	file.write(info[centers[j]]['url'])
	file.write("<br />\n")
	file.write('<a href="' + info[centers[j]]['url'] + '"> Link </a>')
	file.write("<br />\n")
	file.write("Count: " + str(len(d)))
	file.write("<br />\n")
	file.write("<br />\n")
	for i in sorted(d,reverse = True):

		if(i[0] < 0.30):
			continue

		file.write(str(j) + ': Cos score: ' + str(i[0]))
		file.write("<br />\n")
		file.write(str(i[1]) + ': ' + info[i[1]]['title'])
		file.write("<br />\n")
		file.write(info[i[1]]['url'])
		file.write("<br />\n")
		file.write('<a href="' + info[i[1]]['url'] + '"> Link </a>')
		file.write("<br />\n")
		file.write("<br />\n")
		#print('\t' + str(i[0]) + ' ' + str(i[1]))


file.close()
'''






