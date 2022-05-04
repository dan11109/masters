from KL_Divergence import get_result
import nltk
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial
import cosine
import json 
import pickle


multi_source = False 

if __name__ == "__main__":


	f = open('data/info.json')
	info = json.load(f)

	with open('clusters.pkl', "rb") as fIn:
	    clusters = pickle.load(fIn)
	    
	for c in clusters.keys():
		clust = clusters[c]

		




	# tfidf.keys():


