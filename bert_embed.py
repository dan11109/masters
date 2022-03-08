
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial
import pickle
from bert.sentenceTransformers.sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')



query_doc = "d1.txt"
query_doc1 = query_doc[:-4]

inn=Index()
inn.retrieve_file()
inn.tok_lem_stem(type_op='lemmatize')
inn.inverted_index_constr()
inn.calculate_tf_idf(test_file=query_doc)
inn.tfidf_of_query(query_doc1)


sentences = []
order = []
for file in inn.all_files.keys():
    tmp = file + ".txt"
    string =  ' '.join(inn.preprocess_query_doc(filename=tmp))
    sentences.append(string)
    order.append(file)
    
  


#Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)



#Store sentences & embeddings on disc
#with open('embeddings.pkl', "wb") as fOut:
#    pickle.dump({'sentences': sentences, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)


with open('embeddings.pkl', "wb") as fOut:

    pickle.dump({'order': order, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

