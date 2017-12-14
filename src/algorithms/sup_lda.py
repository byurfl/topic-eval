### Standard LDA using scikit-learn ###
from sklearn.decomposition import LatentDirichletAllocation as lda
import sklearn.decomposition as skd
import sklearn.feature_extraction.text as skfet
import os, os.path as path

from algorithms.algorithm import Algorithm

class SupervisedLDA(Algorithm):
    def load_input(self):
        input_files = []
        for root, dirs, files in os.walk(self.input_path):
            input_files.extend([os.path.join(root,f) for f in files if f.endswith('_sentences.txt')])

        self.cv = skfet.CountVectorizer(input='filename', stop_words=None, max_df=0.7)
        self.doc_terms = self.cv.fit_transform(input_files)

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = 'Topic #{}:'.format(topic_idx)
            message += ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

    def run(self):
        self.model = lda(learning_method='online')
        self.model.fit(self.doc_terms)

    def write_output(self):
        feature_names = self.cv.get_feature_names()
        self.print_top_words(self.model, feature_names, 20)

    def evaluate(self):
        pass
