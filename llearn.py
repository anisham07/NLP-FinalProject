from sklearn import linear_model
import numpy as np
import nltk,csv
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import enchant
reader = csv.reader(open('features.csv', 'rU'), delimiter= ",",quotechar='|')
next(reader)
reg = linear_model.LinearRegression()
X = []
Y = []
for line in reader:
    x = []
    y = []
    # print(line[0])
    x.append(float(line[0]))
    # x.append(float(line[1]))
    #x.append(float(line[2]))
    x.append(float(line[3]))
    x.append(float(line[4]))
    x.append(float(line[5]))
    x.append(float(line[6]))
    x.append(float(line[7]))
    x.append(float(line[8]))
    x.append(float(line[9]))
    x.append(float(line[10]))
    x.append(float(line[11]))
    x.append(float(line[12]))
    x.append(float(line[13]))
    x.append(float(line[14]))
    # x.append(float(line[15]))
    y.append(float(line[15]))
    X.append(x)
    Y.append(y)
# a = np.array(X).reshape(-1,1)
# b = np.array(Y).reshape(1,-1)
# print(X,Y)
reg.fit(X, Y,.01)
# print (reg.coef_, reg.intercept_)
stop_words = set(stopwords.words('english'))
d = enchant.Dict("en_US")
openFile = open("Resultlinear.csv","w",encoding="latin1")
writer = csv.writer(openFile)
reader = csv.reader(open('valid_set.tsv', 'rU', encoding="latin1"), delimiter= "\t",quotechar='|')
next(reader)
count = 0
for line in reader:
#    print(line[1])
    if line[1] != str(2):
#        print("here in ")
        spellCorrect = 0
        spellIncorrect =0
        wordCount =0
        lineCount=0
        NN = 0
        NNS = 0
        NNP= 0
        NNPS = 0
        IN = 0
        PRP = 0
        VB = 0
        JJ = 0
        VBG = 0
        VBZ = 0
        VBP = 0
        errors = 0
        ratio = 0
        sentence = sent_tokenize(line[2])
        tokens = word_tokenize(line[2])
        for t in tokens:
            # matches = tool.check(line[2])
            if t in stop_words:
                tokens.remove(t)
            else:
                if(d.check(t)):
                    spellCorrect += 1
                else:
                    spellIncorrect += 1
        tagged_essay = nltk.pos_tag(tokens)
        for (word,tag) in tagged_essay:
            if tag == 'NN':
                NN +=1
            if tag == 'NNP':
                NNP +=1
            if tag == 'VBZ':
                VBZ +=1
            if tag == 'NNPS':
                NNPS +=1
            if tag == 'NNS':
                NNS +=1
            if tag == 'IN':
                IN +=1
            if tag == 'PRP':
                PRP +=1
            if tag == 'VB':
                VB +=1
            if tag == 'JJ':
                JJ +=1
            if tag == 'VBP':
                VBP +=1
            if tag == 'VBG':
                VBG +=1
        errors = spellIncorrect
        ratio = len(tokens)/len(sentence)
        x1 = []
        X1 = []
        x1.append(len(tokens))
        # x1.append(len(sentence))
        #x1.append(ratio)
        x1.append(NN)
        x1.append(NNP)
        x1.append(VBZ)
        x1.append(NNPS)
        x1.append(NNS)
        x1.append(IN)
        x1.append(PRP)
        x1.append(VB)
        x1.append(JJ)
        x1.append(VBP)
        x1.append(VBG)
        x1.append(errors)
        X1.append(x1)
        # print(reg.coef_[0][1])
        output = reg.predict(X1)
        # output = (len(tokens)*reg.coef_[0][0] )
        count += 1
        # finalerror += (output - line[6])*(output - line[6])
        openFile.write(line[0] + "," + str(output[0][0]) + "\n")
        print(output[0][0])
#    openFile.write(line[0] +" " + "".join(str(output[0][0])) + "\n")
#    writer.writerows(zip(line[0],str(output[0])))
# finalerror = finalerror/count
# print(finalerror)
