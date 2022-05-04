
import urllib.request, json 
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial
import pickle
import sys
from bert.sentenceTransformers.sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

query_doc = "d1.txt"
query_doc1 = query_doc[:-4]


arg1 = sys.argv[1] # = 'data/2022-02-01to03.json'

#f = open('data/rpi_school.json')
#f = open('data/dec2020.json')
f = open(arg1)
data = json.load(f)
n = int(sys.argv[2]) #2000 ##math.inf; #100

files = 'd.txt' # 'data/d.txt'
number = 1
title_url = {}
size_cont = 0
size_tit = 0

while(True):
    for art in data['results']:

        if(len(art['content']) < 500): #filter out small articles 
            continue

        if(len(art['title']) > 120): #filter out small articles 
            continue

        size_cont += len(art['content'])
        size_tit += len(art['title'])

        file = 'data/' + files[:1] + str(number) + files[1:]

        with open(file, 'w') as f:
            f.write(art['content'])
  
        title_url[files[:1] + str(number)] = {} # (art['content']) # source url title
        title_url[files[:1] + str(number)]['source'] = art['source']
        title_url[files[:1] + str(number)]['url'] = art['url']
        title_url[files[:1] + str(number)]['title'] = art['title']
        
        number += 1
        if(number > n):
            break

    if(data['next'] == None):
        break

    if(number > n):
        break

    with urllib.request.urlopen(data['next']) as url:
        data = json.loads(url.read().decode())


print(size_cont/n)
print(size_tit/n)


# Serializing json 
json_object = json.dumps(title_url, indent = 4)
  
# Writing to sample.json
with open('data/info.json', 'w') as outfile:
    outfile.write(json_object)

f.close()


## create object and read data into pickle file
inn=Index()
inn.retrieve_file()
inn.tok_lem_stem(type_op='lemmatize')
inn.inverted_index_constr()
inn.calculate_tf_idf(test_file=query_doc)
inn.tfidf_of_query(query_doc1)

with open('all_data.pkl', 'wb') as outp:
    pickle.dump(inn, outp, pickle.HIGHEST_PROTOCOL)


