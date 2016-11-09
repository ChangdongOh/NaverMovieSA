# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:03:32 2016

@author: laman
"""



def read_data(filename): 
    with open(filename, 'r', encoding='UTF8') as f: 
        data = [line.split('\t') for line in f.read().splitlines()] 
        #data = data[1:] # header
    
    return data
    
        
train_data = read_data('assignment_training_data.txt')[1:] 
test_data = read_data('naver_movie_comments_assignments_data.txt')


print(len(train_data)) # nrows: 150000 print(len(train_data[0])) # ncols: 3
print(len(test_data)) # nrows: 50000 print(len(test_data[0])) # ncols: 3



from konlpy.tag import Twitter 
pos_tagger = Twitter()


def tokenize(doc): 
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    
train_docs = [(tokenize(row[0]), row[1]) for row in train_data] 
test_docs = [(tokenize(row[1]), row[0]) for row in test_data]


from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs] 
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

from gensim.models import doc2vec

doc_vectorizer = doc2vec.Doc2Vec(size=500, alpha=0.025, min_alpha=0.025, seed=1234) 
doc_vectorizer.build_vocab(tagged_train_docs)

for epoch in range(10): 
    doc_vectorizer.train(tagged_train_docs) 
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate 
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
    
    
# To save 
# doc_vectorizer.save('doc2vec.model')

train_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_train_docs]
train_y = [doc.tags[0] for doc in tagged_train_docs]
len(train_x)
# => 150000
len(train_x[0])
# => 300

test_x = [doc_vectorizer.infer_vector(doc.words) for doc in tagged_test_docs]
test_y = [doc.tags[0] for doc in tagged_test_docs]
len(test_x) 
# => 50000
len(test_x[0]) 
# => 300


from sklearn.linear_model import LogisticRegression 
classifier = LogisticRegression() 
classifier.fit(train_x, train_y)
classifier.score(test_x, test_y)
