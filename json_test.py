import json

dicts = [{ "name": "Stephen", "Number": 1 }
         ,{ "name": "Glinda", "Number": 2 }
         ,{ "name": "Elphaba", "Number": 3 }
         ,{ "name": "Nessa", "Number": 4 }]

dicts2= [{ "name": "Dorothy", "Number": 5 }
         ,{ "name": "Fiyero", "Number": 6 }]


f = open("test.json","w")
f.write(json.dumps(dicts))
f.close()

f2 = open("test.json","r+")
f2.seek(-1,2)
f2.write(json.dumps(dicts2).replace('[',',',1))
f2.close()

f3 = open('test.json','r')
f3.read()

if __name__ == "__main__":
    pass