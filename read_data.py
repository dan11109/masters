
import urllib.request, json 
import math


#f = open('data/rpi_school.json')
#f = open('data/dec2020.json')
f = open('data/2022-02-01to03.json')

data = json.load(f)


n = 2000##math.inf; #100

files = 'd.txt' # 'data/d.txt'
number = 1

title_url = {}

while(True):

    for art in data['results']:

        file = 'data/' + files[:1] + str(number) + files[1:]

        #art['content'] = art['content'].replace("\n\n", "\n")
       

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



# Serializing json 
json_object = json.dumps(title_url, indent = 4)
  
# Writing to sample.json
with open('data/info.json', 'w') as outfile:
    outfile.write(json_object)


f.close()



