# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 16:17:34 2016

@author: laman
"""

import numpy as np
from gensim.matutils import unitvec

def test(model,positive,negative,test_words):

  mean = []
  for pos_word in positive:
    mean.append(1.0 * np.array(model[pos_word]))

  for neg_word in negative:
    mean.append(-1.0 * np.array(model[neg_word]))

  # compute the weighted average of all words
  mean = unitvec(np.array(mean).mean(axis=0))

  scores = {}
  for word in test_words:

    if word not in positive + negative:

      test_word = unitvec(np.array(model[word]))

      # Cosine Similarity
      scores[word] = np.dot(test_word, mean)

  print(sorted(scores, key=scores.get, reverse=True)[:1])
  return mean

positive=tagged_train_docs[0].words
negative=[]

mean=test(doc_vectorizer, positive, negative, doc_vectorizer.vocab)


doc_vectorizer.similar_by_vector(doc_vectorizer.infer_vector(tagged_train_docs[0].words),mean)