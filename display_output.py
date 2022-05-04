import sys
import csv
 
# filename = "output.csv"
filename = sys.argv[1]

fields = [] #Cluster Title Score url Source
rows = []
urls = [] #used as a unique identifier

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)
        urls.append(row[3])

indices = list(range(0,len(rows)))

if(len(sys.argv) == 4):
	filename2 = sys.argv[2] #second file
	fields2 = [] #Cluster Title Score url Source
	rows2 = []
	urls2 = []
	indices = []

	with open(filename2, 'r') as csvfile:
	    csvreader = csv.reader(csvfile)
	    fields2 = next(csvreader)
	    for row in csvreader:
	        rows2.append(row)
	        urls2.append(row[3])

	if(sys.argv[3] == 'common'): #in both files
		for i in range(len(urls)):
			if(urls[i] in urls2):
				indices.append(i)

	elif(sys.argv[3] == 'difference'): #in the first file but not the second
		for i in range(len(urls)):
			if(urls[i] not in urls2):
				indices.append(i)



file = open("Display_output.html","w")
file.write("<br />\n")

for idx in indices:
	i = rows[idx]
	file.write('Cluster: ' + str(i[0]))
	file.write("<br />\n")
	file.write('Score: ' + str(i[2]))
	file.write("<br />\n")
	file.write(str(i[1]))
	file.write(' <a href="' + str(i[3]) + '"> Link </a>')
	file.write("<br />\n")
	file.write("<br />\n")
	file.write("<br />\n")




