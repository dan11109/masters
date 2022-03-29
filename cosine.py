
import math
import re
from collections import Counter
from build_index import Index
from scipy import spatial
import numpy as np



WORD = re.compile(r"\w+")


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


def get_cosine(vec1, vec2):

    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in list(vec1.keys())])
    sum2 = sum([vec2[x] ** 2 for x in list(vec2.keys())])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def text_to_vector(text):

    words = WORD.findall(text)
    return Counter(words)



def compute(target):
	query_doc = target
	query_doc1 = target[:-4]

	inn=Index()
	inn.retrieve_file()
	inn.tok_lem_stem(type_op='lemmatize')
	inn.inverted_index_constr()
	inn.calculate_tf_idf(test_file=query_doc)
	inn.tfidf_of_query(query_doc1)


	#print(inn.doc_sim_score['assume'])
	# print(inn.tfidf_query_doc)


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

	#print(len(tfidf['d1']))

	targ_vector = text_to_vector( ' '.join(inn.dict_list[query_doc1]) )
	vectors = []
	for key in inn.dict_list.keys():
		if(key != query_doc1):
			v = text_to_vector( ' '.join(inn.dict_list[key]) )
			vectors.append((key,v))
			
			

	#print(vectors[0])

	cosines = []
	#for tup in vectors:
	#	cosines.append((get_cosine(tup[1], targ_vector),tup[0]))

	for doc in tfidf.keys():
		if(doc == query_doc1):
			continue

		cosines.append( ( (1-spatial.distance.cosine(tfidf[doc], tfidf[query_doc1])), doc) )
		#cosines.append((get_cosine(tfidf[doc], tfidf[query_doc1]),doc))



	result = sorted(cosines,reverse=True)

	#return result,inn.all_files[result[0][1]],inn.all_files[result[1][1]]
	return result,inn.all_files[result[0][1]],inn.all_files[result[1][1]]



	'''	
	for tup in sorted(cosines,reverse=True):
		print("Cosine: " + tup[1] + ":\t" + str(tup[0]))

	'''


if __name__ == "__main__":

	print(KL([.2,.3,.3,.2],[.1,.4,.4,.1]))
	print(KL([.2,.4,.4,.2],[.0,.0,.5,.5]))



