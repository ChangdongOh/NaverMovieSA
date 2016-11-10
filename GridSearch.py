# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 02:58:58 2016

@author: ChangdongOh
"""

def read_data(filename): 
    with open(filename, 'r', encoding='UTF8') as f: 
        data = [line.split('\t') for line in f.read().splitlines()] 
        data = data # header
    
    return data
    
import random
train_data = random.sample(read_data('assignment_training_data.txt')[1:], 100000)
test_data = read_data('naver_movie_comments_assignments_data.txt')

 

from konlpy.tag import Twitter 
pos_tagger = Twitter()


def tokenize(doc): 
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True) if t[1]!='Josa']
  
  
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB

pipe =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize, ngram_range=(2, 2))),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=5))])
pipe=pipe.fit([t[0] for t in train_data] , numpy.asarray([int(t[1]) for t in train_data]))
               
import numpy as np

predicted = pipe.predict([t[1] for t in test_data])
print(np.mean(predicted == numpy.asarray([int(t[0]) for t in test_data])))



from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB


pipe =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB(alpha=1))])
pipe=pipe.fit([t[0] for t in train_data] , numpy.asarray([int(t[1]) for t in train_data]))


parameters = {'vect__ngram_range': [(1,1), (1,2)],
              'clf_alpha':(0.01, 0.1, 0.5, 1, 2, 5)}
              
gs_clf = GridSearchCV(pipe, parameters, n_jobs=-1)
gs_clf = gs_clf.fit([t[0] for t in train_data] , numpy.asarray([int(t[1]) for t in train_data]))

best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

print(score)



