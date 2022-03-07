
from KL_Divergence import get_result
import nltk
import math
import re
from collections import Counter
from build_index import Index
import cosine
import json 


multi_source = False 

if __name__ == "__main__":


	f = open('data/info.json')
	info = json.load(f)

	source_file = "d1.txt"

	cos,file1, file2 = cosine.compute(source_file)

	kl,source,kf1,kf2 = get_result(source_file) #sorted(get_result('d1.txt').items(), key = lambda kv:(kv[1], kv[0]),reverse=True)

	top_cos = []
	top_kl = []
	kl_order = []
	cos_order = []
	print('Art\t\tKL\t\t\t\tArt\t\tCOSINE')

	for i in range(len(kl)):

		kl_order.append(kl[i][0])
		cos_order.append(cos[i][1])

		if(len(top_kl) < 20 or len(top_cos) < 20):
			if(multi_source or (info[source_file[:-4]]['source'] != info[kl[i][0]]['source'])):
				print(kl[i][0]+"\t\t"+str(kl[i][1]),end = '\t\t')
				top_kl.append(kl[i][0])
			

			if(multi_source or (info[source_file[:-4]]['source'] != info[cos[i][1]]['source'])):
				print(cos[i][1] + '\t\t' + str(cos[i][0]))
				top_cos.append(cos[i][1])
		


	set_cos = set(top_cos)
	set_kl = set(top_kl)
	only_cos = set_cos - set_kl
	only_kl = set_kl - set_cos

	print("Only in cos top 20:")
	print()
	for i in only_cos:
		print(kl_order.index(i))
		print(info[i]['title'])
		print(info[i]['url'])
		print()


	print("Only in kl top 20:")
	print()
	for i in only_kl:
		print(cos_order.index(i))
		print(info[i]['title'])
		print(info[i]['url'])
		print()


	print("Common to both:")
	print()
	for i in set_cos.intersection(set_kl):
		print(info[i]['title'])
		print(info[i]['url'])
		print()
		

	print("Source article:") # source url title
	print(info[source_file[:-4]]['title'])
	print(info[source_file[:-4]]['url'])
	#print(info[source_file[:-4]]['source'])

	print()
	print("Cosine closest")
	print(info[top_cos[0]]['title'])
	print(info[top_cos[0]]['url'])
	print()



	print('KL closest')
	print(info[top_kl[0]]['title'])
	print(info[top_kl[0]]['url'])
	print()


	'''

	print()
	print("Source article:")
	print(source)
	print()

	print("Cosine closest")
	print()
	print(file1)
	print()
	print(file2)

	print('KL closest')
	print()
	print(kf1)
	print()
	print(kf2)
	print()


	'''


