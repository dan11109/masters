#!/bin/bash

#read in data from the json file
python3 read_data.py data/2022-02-01to03.json 10 # read in data 

#gernate BERT embeddings 
python3 bert_embed.py 

#run the model with distance metric, vector type, and cluster
#python3 run_model.py num_samples cosine/KL tfidf/BERT cluster(leave blank if not)
#python3 run_model.py 10 cosine tfidf cluster 
python3 run_model.py 10 cosine tfidf

#display the final output
python3 display_output.py 