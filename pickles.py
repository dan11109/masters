import pickle
import urllib.request, json 
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial
import pickle
from bert.sentenceTransformers.sentence_transformers import SentenceTransformer
#model = SentenceTransformer('all-MiniLM-L6-v2')

query_doc = "d1.txt"
query_doc1 = query_doc[:-4]





with open('all_data.pkl', 'rb') as inp:
    temp1 = pickle.load(inp)
    

    




