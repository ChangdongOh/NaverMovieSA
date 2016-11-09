# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 04:48:17 2016

@author: laman
"""



def modelcompare(sampless):

    def read_data(filename): 
        with open(filename, 'r', encoding='UTF8') as f: 
            data = [line.split('\t') for line in f.read().splitlines()] 
            data = data # header
    
        return data
    
    import random
    train_data = random.sample(read_data('assignment_training_data.txt')[1:], sampless)
    test_data= read_data('naver_movie_comments_assignments_data.txt')

    from konlpy.tag import Twitter 
    pos_tagger = Twitter()


    def tokenize(doc): 
        return ['/'.join(t) for t in pos_tagger.pos(doc, norm=True, stem=True) if t[1]!='Josa']
  
  
    from sklearn.feature_extraction.text import CountVectorizer

    count_vect = CountVectorizer(tokenizer=tokenize)
    train_count = count_vect.fit_transform([t[0] for t in train_data])

    from sklearn.feature_extraction.text import TfidfTransformer

    tfidf_tran = TfidfTransformer()
    tfidf_count = tfidf_tran.fit_transform(train_count)

    import numpy
    
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import MultinomialNB

    modellist=[LogisticRegression(n_jobs=-1),LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,intercept_scaling=1, loss='squared_hinge', max_iter=1000,multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,verbose=0),MultinomialNB()]

    def accuracy(pipeline):
        predicted = pipeline.predict([t[1] for t in test_data])
        return np.mean(predicted == numpy.asarray([int(t[0]) for t in test_data]))
    result=[]
    for i in modellist:
        pipe1 =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('clf', i)])
        pipe1=pipe1.fit([t[0] for t in train_data] , numpy.asarray([int(t[1]) for t in train_data]))
        pipe2 =  Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                     ('tfidf', TfidfTransformer()),
                     ('clf', i)])
        pipe2=pipe2.fit([t[0] for t in train_data] , numpy.asarray([int(t[1]) for t in train_data]))
        print(i)        
        print(accuracy(pipe1))
        print(accuracy(pipe2))
        result+=[accuracy(pipe1),accuracy(pipe2)]
        
    return result
        
        
        
for i in range(4,10):
    print(i*10000)
    result=modelcompare(i*10000)
    print(result)
        
        

    

