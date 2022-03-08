
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

	source_file = "d1.txt"

	cos,file1, file2 = cosine.compute(source_file)

	kl,source,kf1,kf2 = get_result(source_file) #sorted(get_result('d1.txt').items(), key = lambda kv:(kv[1], kv[0]),reverse=True)



	#Load sentences & embeddings from disc

	with open('embeddings.pkl', "rb") as fIn:
	    stored_data = pickle.load(fIn)
	    stored_order = stored_data['order']
	    stored_embeddings = stored_data['embeddings']





	cos_bert = []
	for i in range(len(stored_embeddings)): 

		if(stored_order[i] == source_file[:-4]):
			continue

		emb = stored_embeddings[i]
		dist = 1 - spatial.distance.cosine(stored_embeddings[0], emb)
		
		cos_bert.append( (dist,stored_order[i]) )



	cos_bert.sort(reverse=True)
	
	bert_order = []
	top_bert = []
	top_cos = []
	top_kl = []
	kl_order = []
	cos_order = []
	print('Art\t\tKL\t\t\t\tArt\t\tCOSINE\t\t\t\tArt\t\tBERT')

	for i in range(len(kl)): #filter for sourse 

		if(multi_source or (info[source_file[:-4]]['source'] != info[kl[i][0]]['source'])):
			top_kl.append(kl[i][0])
			kl_order.append(kl[i][1])
		
		if(multi_source or (info[source_file[:-4]]['source'] != info[cos[i][1]]['source'])):
			top_cos.append(cos[i][1])
			cos_order.append(cos[i][0])

		if(multi_source or (info[source_file[:-4]]['source'] != info[cos_bert[i][1]]['source'])):
			top_bert.append(cos_bert[i][1])
			bert_order.append(cos_bert[i][0])
		

	for i in range(20):
		print(top_kl[i] +"\t\t"+str(kl_order[i]),end = '\t\t')
		print(top_cos[i] + '\t\t' + str(cos_order[i]),end ='\t\t')
		print(top_bert[i] + '\t\t' + str(bert_order[i]))




	set_bert = set(top_bert[:20])
	set_cos = set(top_cos[:20])
	set_kl = set(top_kl[:20])


	only_cos = set_cos - set_kl - set_bert
	only_kl = set_kl - set_cos - set_bert
	only_bert = set_bert - set_cos - set_kl



	print()
	print('Number in commom: bert and cosine top 10: ')
	print(len(  set(top_bert[:10]).intersection(set(top_cos[:10]))  ))
	print()
	print('Number in commom: KL and cosine top 10: ')
	print(len(  set(top_kl[:10]).intersection(set(top_cos[:10]))  ))
	print()
	print('Number in commom: bert and KL top 10: ')
	print(len(  set(top_bert[:10]).intersection(set(top_kl[:10]))  ))
	print()
	print('Number in commom all for top 10:')
	print(len(  set(top_bert[:10]).intersection(set(top_kl[:10])).intersection(set(top_cos[:10]))  ))
	print()

	print()
	print('Number in commom: bert and cosine top 50: ')
	print(len(  set(top_bert[:50]).intersection(set(top_cos[:50]))  ))
	print()
	print('Number in commom: KL and cosine top 50: ')
	print(len(  set(top_kl[:50]).intersection(set(top_cos[:50]))  ))
	print()
	print('Number in commom: bert and KL top 50: ')
	print(len(  set(top_bert[:50]).intersection(set(top_kl[:50]))  ))
	print()
	print('Number in commom all for top 50:')
	print(len(  set(top_bert[:50]).intersection(set(top_kl[:50])).intersection(set(top_cos[:50]))  ))
	print()

	print()
	print('Number in commom: bert and cosine top 100: ')
	print(len(  set(top_bert[:100]).intersection(set(top_cos[:100]))  ))
	print()
	print('Number in commom: KL and cosine top 100: ')
	print(len(  set(top_kl[:100]).intersection(set(top_cos[:100]))  ))
	print()
	print('Number in commom: bert and KL top 100: ')
	print(len(  set(top_bert[:100]).intersection(set(top_kl[:100]))  ))
	print()
	print('Number in commom all for top 100:')
	print(len(  set(top_bert[:100]).intersection(set(top_kl[:100])).intersection(set(top_cos[:100]))  ))
	print()

	#write to html
	file = open("output.html","w")

	file.write("<br />\n")
	file.write("Only in cos top 20:")
	file.write("<br />\n")
	for i in only_cos:
		file.write('KL index: ' + str(top_kl.index(i)))
		file.write("<br />\n")
		file.write(info[i]['title'])
		file.write("<br />\n")
		file.write(info[i]['url'])
		file.write("<br />\n")
		file.write('<a href="' + info[i]['url'] + '"> Link </a>')
		file.write("<br />\n")
		file.write("<br />\n")

	file.write("Only in KL top 20:")
	file.write("<br />\n")
	for i in only_kl:
		file.write('Cos index: ' + str(top_cos.index(i)))
		file.write("<br />\n")
		file.write(info[i]['title'])
		file.write("<br />\n")
		file.write(info[i]['url'])
		file.write("<br />\n")
		file.write('<a href="' + info[i]['url'] + '"> Link </a>')
		file.write("<br />\n")
		file.write("<br />\n")


	file.write("Only in BERT top 20:")
	file.write("<br />\n")
	for i in only_bert:
		file.write('Cos index: ' + str(top_cos.index(i)))
		file.write("<br />\n")
		file.write('KL index: ' + str(top_kl.index(i)))
		file.write("<br />\n")
		file.write(info[i]['title'])
		file.write("<br />\n")
		file.write(info[i]['url'])
		file.write("<br />\n")
		file.write('<a href="' + info[i]['url'] + '"> Link </a>')
		file.write("<br />\n")
		file.write("<br />\n")
		


	file.write("Common to all:")
	file.write("<br />\n")
	for i in set_cos.intersection(set_kl).intersection(set_bert):
		file.write(info[i]['title'])
		file.write("<br />\n")
		file.write(info[i]['url'])
		file.write("<br />\n")
		file.write('<a href="' + info[i]['url'] + '"> Link </a>')
		file.write("<br />\n")
		file.write("<br />\n")

		

	file.write("Source article:") # source url title
	file.write("<br />\n")
	file.write(info[source_file[:-4]]['title'])
	file.write("<br />\n")
	file.write(info[source_file[:-4]]['url'])
	file.write("<br />\n")
	file.write('<a href="' + info[source_file[:-4]]['url'] + '"> Link </a>')



	file.close()





