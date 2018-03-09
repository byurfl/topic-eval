import nltk.tokenize as tok
import numpy as np
import os.path
import src.algorithms.ankura.pipeline as pipeline

if False:
    import nltk
    nltk.download('punkt')

class Reader:
    def __init__(self, stopwords):
        self.doc_tokens = {}
        self.term_idx = 0
        self.terms_to_indices = {}
        self.vocabulary = {}
        self.documents = None

        self.stopwords = self.get_stopwords(stopwords if stopwords is not None else './data/english_stopwords.txt')

    def get_stopwords(self, stopwords_file):
        with open(stopwords_file, mode='r', encoding='utf-8') as sw_file:
            stopwords = set([line.strip() for line in sw_file])
        return stopwords

    def get_tokens(self, text):
        tokenizer = pipeline.stopword_tokenizer(pipeline.default_tokenizer(), stopwords=self.stopwords)
        # return [t for t in tok.word_tokenize(text) if t not in self.stopwords]
        return [t.token for t in tokenizer(text)]

    def add_to_vocab(self, token):
        if token in self.vocabulary:
            self.vocabulary[token] += 1
        else:
            self.vocabulary[token] = 1
            self.terms_to_indices[token] = self.term_idx
            self.term_idx += 1

    def get_doc_tokens(self, corpus):
        if os.path.isdir(corpus):
            for f in os.listdir(corpus):
                with open(os.path.join(corpus, f), mode='r', encoding='utf-8') as file:
                    text = file.read()
                tokens = self.get_tokens(text)
                self.doc_tokens[f] = tokens

        elif os.path.isfile(corpus):
            with open(corpus, mode='r', encoding='utf-8') as file:
                for line in file:
                    id,doc = line.split('\t')
                    tokens = self.get_tokens(doc)
                    self.doc_tokens[id] = tokens

    def build_vocab(self):
        for doc_id,tokens in self.doc_tokens.items():
            for t in tokens:
                self.add_to_vocab(t)

    def convert_docs_to_matrix(self):
        self.documents = np.zeros(shape=[len(self.vocabulary), len(self.doc_tokens)])
        doc_ids = sorted(self.doc_tokens.keys())
        doc_idx = 0
        for doc_id in doc_ids:
            for t in self.doc_tokens[doc_id]:
                self.documents[self.terms_to_indices[t]][doc_idx] += 1
            doc_idx += 1

    def read_corpus(self, corpus):
        self.get_doc_tokens(corpus)
        self.build_vocab()
        self.convert_docs_to_matrix()

