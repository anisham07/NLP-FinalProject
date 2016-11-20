import nltk,csv
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
import enchant
# import language_check
reader = csv.reader(open('training_set_rel3.tsv', 'rU', encoding="latin1"), delimiter= "\t",quotechar='|')
next(reader)
stop_words = set(stopwords.words('english'))
write = open('eggs.csv', 'w', newline='')
writer = csv.writer(write, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer.writerow(['wordCount', 'sentenceCount','ratio','NN','NNP','VBZ','NNPS','NNS','IN','PRP','VB','JJ','VBP','VBG','error','countMatch','output' ])
d = enchant.Dict("en_US")
filereader = csv.reader(open(r'Essay/essay1.csv', 'rU', encoding="latin1"), delimiter= ",",quotechar='|')
fileTokens = []
for line in filereader:
    fileTokens.append(line[0])
# tool = language_check.LanguageTool("en-US");
for line in reader:
    if int(line[1]) == 1 :
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

        countMatch =0
        for t in tokens:
            # matches = tool.check(line[2])
            if t in stop_words:
                tokens.remove(t)
            else:
                if(d.check(t)):
                    spellCorrect += 1
                else:
                    spellIncorrect += 1
                if t in fileTokens:
                    countMatch += 1
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
        writer.writerow([len(tokens), len(sentence),ratio,NN,NNP,VBZ,NNPS,NNS,IN,PRP,VB,JJ,VBP,VBG,errors,countMatch, line[6]])
write.close()
