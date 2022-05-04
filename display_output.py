

import csv
 
filename = "output.csv"

fields = [] #Cluster Title Score url Source

rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
     
    fields = next(csvreader)

    for row in csvreader:
        rows.append(row)

file = open("Display_output.html","w")
file.write("<br />\n")

for i in rows:

    file.write('Cluster: ' + str(i[0]))
    file.write("<br />\n")
    file.write('Score: ' + str(i[2]))
    file.write("<br />\n")
    file.write(str(i[1]))
    file.write(' <a href="' + str(i[3]) + '"> Link </a>')
    file.write("<br />\n")
    file.write("<br />\n")
    file.write("<br />\n")




