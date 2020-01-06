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


class Doc2Tag2Vec(Doc2Vec):
    
    
    def __init__(self, **kwargs):
        """
        Model initializer from gesim..models.doc2vec
        """
        super(Doc2Tag2Vec, self).__init__(**kwargs)
        
    
    def _data_loader(self, 
                     file_path: str, 
                     use_csv  : bool = True):
        
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
                    yield TaggedDocument(training_data.read().split(), [idx])
                    
        return corpus
                    
    
    def load_training_data(self, **kwargs):
        """Reads all text files from a given directory.
        
        file_path -> str : Path to the the directory containing all documents.
        use_csv   -> bool: If the files inside the directory are csv or txt 
                           files
        documents -> List[TaggesDocument]: Can be simply a list of 
                                           TaggedDocument elements, but for larger 
                                           corpora, consider an iterable 
                                           that streams the documents directly 
                                           from disk/network
        corpus_file -> str    : Path to a corpus file in LineSentence format. You 
                                may use this argument instead of documents to get 
                                performance boost. Only one of documents or 
                                corpus_file arguments need to be passed 
                               (not both of them)
        update -> bool        : If true, the new words in documents will be added to 
                                model’s vocab.
        progress_per -> int   : Indicates how many words to process before 
                                showing/updating the progress.
        keep_raw_vocab -> bool: If not true, delete the raw vocabulary after 
                                the scaling is done and free up RAM.
        """
        corpus = list(self._data_loader(**kwargs))
        
        self.build_vocab(corpus)
        
        return corpus
    
    
    def train(self,
              file_path : str ='keywords.txt',
              sep       : str = '\n',
              **kwargs):
        """
        Override the gensim.models.doc2vec train method. After running the 
        train method this method creates adn returns a Dictionary containing all
        keyword vectors.
        """
        
        print("Training model...")
        super(Doc2Tag2Vec, self).train(**kwargs)
        
        print("Loading keywords")
        
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
    
    
    def predict_keywords(self, 
                         document_id     : str, 
                         nbr_of_keywords : int = -1):
        """
        Returns the most similar keyword(s) for a given document id.
        """
        
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
    
    #corpus =  model.load_training_data(file_path="doctest", use_csv=False)
    
    #model.train(documents=corpus, total_examples=model.corpus_count,  epochs=model.epochs, file_path="doctest/sachdiskreptoren_preprocessed.txt")
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        