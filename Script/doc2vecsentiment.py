# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 16:03:32 2016

@author: ChangdongOh
"""



def read_data(filename): 
    with open(filename, 'r', encoding='UTF8') as f: 
        data = [line.split('\t') for line in f.read().splitlines()] 
    
    return data
    
import random 
       
train_data = random.sample(read_data('NaverMovieSA\\Data\\assignment_training_data.txt')[1:],100000)  
#data = data[1:] # header for training data
#sampling 100,000 training data
test_data = read_data('NaverMovieSA\\Data\\naver_movie_comments_assignments_data.txt')


print(len(train_data)) 
print(len(test_data)) 



from konlpy.tag import Twitter 
pos_tagger = Twitter()


def tokenize(doc): 
    return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True)]
    
train_docs = [(tokenize(row[0]), row[1]) for row in train_data] 
test_docs = [(tokenize(row[1]), row[0]) for row in test_data]


from collections import namedtuple
TaggedDocument = namedtuple('TaggedDocument', 'words tags')

#doc2vec에 맞는 형태로 변환
tagged_train_docs = [TaggedDocument(d, [c]) for d, c in train_docs] 
tagged_test_docs = [TaggedDocument(d, [c]) for d, c in test_docs]

from gensim.models import doc2vec

doc_vectorizer = doc2vec.Doc2Vec(size=500, alpha=0.025, min_alpha=0.025, workers=4) 
doc_vectorizer.build_vocab(tagged_train_docs)

for epoch in range(10): 
    doc_vectorizer.train(tagged_train_docs) 
    doc_vectorizer.alpha -= 0.002 # decrease the learning rate 
    doc_vectorizer.min_alpha = doc_vectorizer.alpha # fix the learning rate, no decay
    
# To save 
# doc_vectorizer.save('doc2vec.model')

doc_vectorizer.init_sims(replace=True)   

from pprint import pprint

pprint(doc_vectorizer.most_similar('영화/Noun'))
pprint(doc_vectorizer.most_similar('ㅠㅠ/KoreanParticle'))


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
