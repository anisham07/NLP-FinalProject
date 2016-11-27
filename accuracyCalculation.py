from sklearn.metrics import accuracy_score
import csv

openfile = open("accuracy.csv","r",encoding="latin1")

reader = csv.reader(openfile, delimiter= ",",quotechar='|')
x = []
y = []
for line in reader:
#    print(x)
#    print(y)
    x.append(line[0])
    y.append(line[1])
print(x)
print(y)
print(accuracy_score(x, y))