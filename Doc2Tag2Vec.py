# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:50:50 2019

@author: Soeren
"""

import csv
from glob import glob
from collections import OrderedDict
from scipy.spatial.distance import cosine
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


#TODO: documentation


class Doc2Tag2Vec(Doc2Vec):
    
    
    def __init__(self, **kwargs):
        
        super(Doc2Tag2Vec, self).__init__(**kwargs)
        
    
    def _data_loader(self, file_path, use_csv = True):
        
        corpus = []
        
        if use_csv:
            with open(file_path, 'r', encoding='utf8') as training_data:
                corpus = csv.reader(training_data, delimiter=",")
                for row in corpus:
                    yield TaggedDocument(row[1].split(), [row[0]])
        else:
            files = glob(file_path + "/*.txt")
            
            for file in files:
                
                # TODO: better
                idx = file.split("\\")[-1].replace(".txt", "")
                
                with open(file, 'r', encoding='utf8') as training_data:
                    yield TaggedDocument(training_data.read(), [idx])
                    
        return corpus
                    
                    
    def load_training_data(self, **kwargs):
        
        corpus = list(self._data_loader(**kwargs))
        
        self.build_vocab(corpus)
        
        return corpus
    
    
    # TODO: append to model.train method // make private
    def load_keywords(self, file_path, sep='\n'):
        
        self.keyword_vec_dict = OrderedDict()
        self.no_vector = []
        
        with open(file_path, 'r', encoding='utf8') as all_keywords:
            words = all_keywords.read().split(sep)
        
        for keyword in words:
            # TODO: edgecase
            keyword = keyword.replace(' ', '_').lower()
            keyword = keyword.replace('-', '_')
            keyword = keyword.replace('__', '_')
            if keyword[-1:] == '_':
                keyword = keyword[:-1]
            try:
                self.keyword_vec_dict[keyword] = self.wv.get_vector(keyword)
            except KeyError:
                self.no_vector.append(keyword)
        
        print("{} labels have a vector representation".format(len(self.keyword_vec_dict)))
        print("{} labels don't have a vector representation".format(len(self.no_vector)))
        
        return self.keyword_vec_dict
    
    
    def _load_keywords(self, file_path, sep='\n'):
        
        all_keywords = []
        
        with open(file_path, 'r', encoding='utf8') as file:
            words = file.read().split(sep)
            
        for keyword in words:
            
            keyword = keyword.replace(' ', '_').lower()
            keyword = keyword.replace('-', '_')
            keyword = keyword.replace('__', '_')
            
            if keyword[-1:] == '_':
                keyword = keyword[:-1]
                
            if keyword not in all_keywords:
                all_keywords.append(keyword)
                
        keyword_vec_dict = OrderedDict()
        no_vector = []
        
        for keyword in all_keywords:
            try:
                keyword_vec_dict[keyword] = self.vw.get_vector(keyword)
            except IndexError:
                # List of all labels that do not have a vector representation in the model
                no_vector.append(keyword)
                        
        return all_keywords
    
    
    def predict_keywords(self, document_id, nbr_of_keywords = -1):
        
        document_keywords = {}
        
        if not isinstance(document_id, str):
            for key, value in dict.items():
                similarity = cosine(document_id,value) - 1
                similarity = -similarity
                document_keywords[key] = similarity
        else:
            document_vector = self.docvecs[document_id]
            for key, value in dict.items():
                similarity = cosine(document_vector,value) - 1
                similarity = -similarity
                document_keywords[key] = similarity
                
        prediction = [(k, document_keywords[k]) for k in sorted(document_keywords, key=document_keywords.get, reverse=True)]
        
        if nbr_of_keywords > 0:
            prediction = prediction[:nbr_of_keywords]
        
        return prediction
    
        
if __name__ == "__main__":
    
    model = Doc2Tag2Vec(vector_size=150,
                min_count=1,
                epochs=15,
                window=10,
                min_alpha=0.0002,
                dm=1,
                workers=64,
                negative=1)
    
    corpus =  model.load_training_data(file_path="doctest", use_csv=False)

    model.train(corpus, total_examples=model.corpus_count,  epochs=model.epochs)
    tmp = model._load_keywords("doctest/sachdiskreptoren_preprocessed.txt")
    
    #print(tmp)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        