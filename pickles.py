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







with open('all_data.pkl', 'rb') as inp:
    inn = pickle.load(inp)
    

print(inn.all_words)




