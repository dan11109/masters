#!/bin/bash

#read in data from the json file
python3 read_data.py data/2022-02-01to03.json 2000 # read in data 

#gernate BERT embeddings 
python3 bert_embed.py 

#run the model with distance metric, vector type, and cluster
#python3 run_model.py num_samples cosine/KL tfidf/BERT out_put.csv cluster(leave blank if not)
#python3 run_model.py 10 cosine tfidf output.csv cluster 
python3 run_model.py 1000 cosine tfidf output.csv

#display the final output
python3 display_output.py output.csv # display output 
#python3 display_output.py output.csv output2.csv common #output in common 
#python3 display_output.py output.csv output2.csv difference #output in first csv but not second