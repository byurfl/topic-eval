import nltk.tokenize as tok
import numpy as np
import os.path
import src.algorithms.ankura.pipeline as pipeline
import src.algorithms.sam.util as util
import traceback

if False:
    import nltk
    nltk.download('punkt')

class Reader:
    def __init__(self, stopwords, corpus_encoding = 'utf-8', use_vocab_dict = True, tfidf = False):
        self.tfidf = tfidf
        self.doc_tokens = {}
        self.term_idx = 0
        self.terms_to_indices = {}
        self.indices_to_terms = []
        self.vocabulary = {}
        self.vocab_size = 0
        self.documents = None
        self.corpus_encoding = corpus_encoding
        self.stopwords = self.get_stopwords(stopwords if stopwords is not None else './data/english_stopwords.txt')
        self.use_vocab_dict = use_vocab_dict
        if self.use_vocab_dict:
            self.vocab_dict = [word.decode() for word in list(np.load('./data/vocab20k.npy'))]

    def get_stopwords(self, stopwords_file):
        with open(stopwords_file, mode='r', encoding='utf-8') as sw_file:
            stopwords = set([line.strip() for line in sw_file])
        return stopwords

    def get_tokens(self, text):
        tokenizer = pipeline.stopword_tokenizer(pipeline.default_tokenizer(), stopwords=self.stopwords)
        # return [t for t in tok.word_tokenize(text) if t not in self.stopwords]

        # Don't add if not a valid word
        if self.use_vocab_dict:
            x = [t.token for t in tokenizer(text.lower()) if t.token in self.vocab_dict]
            return x
        else:
            return [t.token for t in tokenizer(text)]

    def add_to_vocab(self, token):

        if token in self.vocabulary:
            self.vocabulary[token] += 1
        else:
            self.vocabulary[token] = 1
            self.terms_to_indices[token] = self.term_idx
            self.indices_to_terms.append(token)
            self.term_idx += 1
        self.vocab_size += 1

    def get_doc_tokens(self, corpus, recursive = True, encoding = 'utf-8'):
        #n = 0
        #if n == 0:
        #encoding = self.get_codec(full_path)
        if corpus.find("news"):
            encoding = 'ansi'

        if os.path.isdir(corpus):
            if recursive:
                for dir, paths, files in (os.walk(corpus)):
                    for f in files:
                        full_path = os.path.join(dir, f)

                        with open(full_path, mode='r', encoding=encoding) as file:
                            text = file.read()
                        tokens = self.get_tokens(text)
                        self.doc_tokens[f] = tokens

            else:
                for f in os.listdir(corpus):
                    with open(os.path.join(corpus, f), mode='r', encoding=encoding) as file:
                        text = file.read()
                    tokens = self.get_tokens(text)
                    self.doc_tokens[f] = tokens

        elif os.path.isfile(corpus):
            with open(corpus, mode='r', encoding=encoding) as file:
                for line in file:
                    id,doc = line.split('\t')
                    tokens = self.get_tokens(doc)
                    self.doc_tokens[id] = tokens

    def build_vocab(self):
        for doc_id,tokens in self.doc_tokens.items():
            for t in tokens:
                self.add_to_vocab(t)

    def convert_docs_to_matrix(self, tfidf=False):
        self.documents = np.zeros(shape=[len(self.vocabulary), len(self.doc_tokens)])
        print(self.documents.shape)
        doc_ids = sorted(self.doc_tokens.keys())
        doc_idx = 0
        for doc_id in doc_ids:
            for t in self.doc_tokens[doc_id]:
                self.documents[self.terms_to_indices[t]][doc_idx] += 1
            doc_idx += 1

        # documents matrix now contains term frequency counts
        if tfidf:
            max_tfs = util.make_row_vector(np.max(self.documents, axis=0))
            doc_freqs = util.make_col_vector(np.sum(self.documents, axis=1))

            self.documents = self.documents / max_tfs
            self.documents = self.documents / (doc_freqs + 1.0)

            # keep top 5000 terms with maximum average tf-idf scores
            if self.documents.shape[0] > 5000:
                avg_tfidf_terms = np.sum(self.documents, axis=1) / self.documents.shape[1]
                sorted_tfidfs = np.argsort(avg_tfidf_terms)
                num_to_delete = len(sorted_tfidfs) - 5000
                to_delete = sorted_tfidfs[:num_to_delete]
                self.documents = np.delete(self.documents, to_delete, axis=0)
                self.vocab_size = 5000
                vocab = np.asarray(self.indices_to_terms)
                pruned = np.delete(vocab, to_delete)
                self.indices_to_terms = pruned.tolist()
                new_terms_to_indices = {}
                term_idx = 0
                for i in range(len(self.indices_to_terms)):
                    new_terms_to_indices[self.indices_to_terms[i]] = term_idx
                    term_idx += 1
                self.terms_to_indices = new_terms_to_indices

        for d in range(doc_idx):
            self.documents[:,d] = util.l2_normalize(self.documents[:,d])

    def read_corpus(self, corpus):
        self.get_doc_tokens(corpus)
        self.build_vocab()
        self.convert_docs_to_matrix(self.tfidf)

    def get_codec(self, f):
        for codec in ["utf-8", "ansi"]:
            try:
                with open(f, mode='r', encoding=codec) as file_obj:
                    text = file_obj.read()
                print("Codec is " + codec)
                return codec
            except:
                print("Codec " + codec + " didn't work")
                traceback.print_exc()

        import sys
        print("No codec found")
        STOP

    def open_file(self, file, encoding='utf-8'):
        for codec in [encoding, "utf-8", "ansi"]:
            try:
                with open(file, mode='r', encoding=codec) as file:
                    text = file.read()
                return text
            except UnicodeDecodeError:
                pass

    f = r"D:\PyCharm Projects\py-sam-master\topic-eval\data\corpus\20_newsgroups\alt.atheism\49960"
    def test(file, codec = 'utf-8'):
        with open(file, mode='r', encoding=codec) as file:
            text = file.read()
        print(text)
    #test(f, "ansi")
