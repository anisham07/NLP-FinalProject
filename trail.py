import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from gensim.models import word2vec
import numpy as np
from sklearn import linear_model
from sklearn import ensemble
from sklearn import svm

tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
count = 0

def essays_to_sentences(essay, tokenizer, remove_stopWords = True):
    '''
    Split the essay into list of sentences.
    Each sentence must be further split into a list of words
    :param remove_stopWords: False, True
    :return: return a list of lists
    each nested list contains the words present in sentence.
    the entire list contains all the sentences in the essay.
    '''
    essay1 = essay
    # print " Processing essay %d" % count
    all_sentences = tokenizer.tokenize(essay1.strip())
    sentences = []
    for sentence in all_sentences:
        if(len(sentence) > 0 ):
            sentences.append(sentences_to_words(sentence, remove_stopWords))
    return sentences

def sentences_to_words(raw_essay, remove_stopwords):
    '''
    Given a sentence, split it into words and return a list of words
    :return: list of words
    '''
    # Remove html
    essay_text = raw_essay
    # BeautifulSoup(raw_essay).get_text()
    # Remove punctuation ?
    essay = re.sub("[^a-zA-Z]", " ", essay_text)
    # get individual words
    words= essay.lower().split()
    # remove stop words?
    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        words = [w for w in words if w not in stop_words]
    return words

def train_model(all_sentences, features):
    num_features = features
    num_workers = 4
    min_word_count = 1
    context = 10
    downsampling  = 1e-3

    model = word2vec.Word2Vec(all_sentences, workers=num_workers,
            size=num_features, min_count = min_word_count,
            window = context, sample = downsampling)
    # model.save("essays_model_300 features")
    return model

def expand_features(model, data, features):
    new_features = []
    for essay in data:
        words = sentences_to_words(essay, True)
        result = np.zeros(features)
        count = 0
        for word in words:
            if word in model:
                result += model[word]
                count += 1
        result = result/float(count)
        new_features.append(result)
    df = pd.DataFrame(new_features, index = data.index).fillna(0)
    return df

if __name__ == '__main__':
    df = pd.DataFrame.from_csv("training_set_rel3.tsv", sep='\t', header=0, encoding='latin1')
    # print df.shape
    all_essays = df['essay']
    input_essays = []
    features  = 300
    for essay in all_essays:
        input_essays += essays_to_sentences(essay, tokenizer)
    #
    # # Learn a model
    model = train_model(input_essays, features)
    df1 = expand_features(model, all_essays, features)
    # train_data = pd.concat([df['essay_set'], df1, df.drop(['essay_set','essay'], axis=1)], axis=1).fillna(0)
    temp_train_data = pd.concat([df['essay_set'], df1], axis=1).fillna(0)

    X_tr = np.array(temp_train_data)
    Y_tr = np.array(df['domain1_score'])
    reg = linear_model.LinearRegression()
    reg.fit(X_tr, Y_tr)
    reg2 = ensemble.RandomForestRegressor()
    reg2.fit(X_tr, Y_tr)
    '''
    Validation Data
    '''
    validation  = pd.DataFrame.from_csv("valid_set.tsv", sep='\t', header = 0,encoding='latin1')
    df2 = expand_features(model, validation['essay'], features)
    test_data = pd.concat([validation['essay_set'], df2], axis = 1).fillna(0)

    X_te = np.array(test_data)
    Y_te = reg.predict(X_te)
    Y2 = reg2.predict(X_te)
    # print "test shape", Y_te.shape
    np.savetxt("output_linear_wordvec.csv", Y_te, delimiter=',')
    np.savetxt("output_forest_wordvec.csv", Y2, delimiter=',')

    svm_trainer_rbf = svm.SVR(kernel='rbf', cache_size=1000, C=0.25)
    svm_trainer_rbf.fit(X_tr, Y_tr)
    Y_te_svm = svm_trainer_rbf.predict(X_te)
    np.savetxt("output_svmrbf_wordvec.csv", Y_te_svm, delimiter=',')

    # svm_poly = svm.SVR(kernel='poly', cache_size=1000, degree=)

    # svm_trainer_rbf = svm.SVR(kernel='rbf', cache_size=1000, C=0.25)
    # svm_trainer_rbf = svm.SVR(kernel='poly', degree= 10, cache_size=1000, C=0.25)
