### Standard LDA using scikit-learn ###
from sklearn.decomposition import LatentDirichletAllocation as lda
import sklearn.feature_extraction.text as skfet
from sklearn.linear_model import LogisticRegression as lr
import os, random
import src.algorithms.ankura.validate as validate
from src.algorithms import anchor_words
import numpy as np
import gensim

from src.algorithms.algorithm import Algorithm

if os.environ["COMPUTERNAME"] == 'DALAILAMA':
    import sys
    path = r"D:\PyCharm Projects\py-sam-master\topic-eval\src"
    os.chdir(path)


class LDA(Algorithm):
    def load_input(self):
        input = self.get_files(self.input_path)
        stopwords = [w.strip() for w in open(r'..\data\english_stopwords.txt', 'r', encoding='utf-8')]
        self.cv = skfet.CountVectorizer(input='content', stop_words=stopwords, tokenizer=self.get_tokenizer())

        lines = []
        for d in input:
            for l in open(d, 'r', encoding='utf-8'):
                lines.append(l)

        self.doc_terms = self.cv.fit_transform(lines)

    def print_top_words(self, model, feature_names, n_top_words):
        for topic_idx, topic in enumerate(model.components_):
            message = 'Topic #{}:'.format(topic_idx)
            message += ' '.join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])
            print(message)

    def run(self):
        self.model = lda(n_topics=100, learning_method='online')
        self.model.fit(self.doc_terms)

        feature_names = self.cv.get_feature_names()
        topic_matrix = []
        for topic_idx, topic in enumerate(self.model.components_):
            row = [feature_names[i] for i in topic.argsort()[:-10 - 1:-1]]
            topic_matrix.append(row)

        self.topics = np.array(topic_matrix)

    def write_output(self, iteration_counter=None):
        output_file = os.path.join(self.output_path, 'topics.txt') \
            if iteration_counter is None \
            else os.path.join(self.output_path, 'topics_' + str(iteration_counter) + '.txt')

        with open(output_file, 'w', encoding='utf-8') as output:
            for row in self.topics:
                output.write('[')
                output.write("'" + row[0] + "'")
                for i in range(1, len(row)):
                    output.write(" '")
                    output.write(row[i])
                    output.write("'")
                output.write(']\n')

        # self.print_top_words(self.model, feature_names, 10)

    def eval_methods(self):
        return ['coherence', 'accuracy']

    def evaluate(self, method, score_file, iteration_counter=None):
        if method == 'coherence':
            # coherence method stolen from ankura, just to make sure
            # the results are comparable to the topic modeling we did with ankura
            aw = anchor_words.AnchorWords(self.input_path, None, None, self.filter_string)
            aw.load_input()

            words_to_index = dict()
            for i in range(len(aw.corpus.vocabulary)):
                word = aw.corpus.vocabulary[i]
                words_to_index[word] = i

            topic_summary = np.array([[words_to_index[w] for w in topic] for topic in self.topics])
            scores = validate.coherence(aw.corpus, topic_summary)

            scores_file = os.path.join(self.output_path, 'coherence.txt') \
                if iteration_counter is None \
                else os.path.join(self.output_path, 'coherence_' + str(iteration_counter) + '.txt')

            with open(scores_file, mode='w', encoding='utf-8') as output:
                output.write(str(scores))
                output.write('\n')

            if score_file is not None:
                score_file.write(str(np.average(scores)))
                score_file.write('\n')

            print("Scores:")
            print(scores)

        elif method == 'accuracy':
            w2v_model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            w2v_vectors = w2v_model.wv
            del w2v_model

            stopwords = [w.strip() for w in open(r'..\data\english_stopwords.txt', 'r', encoding='utf-8')]
            tokenizer = self.get_tokenizer(remove_labels=False)

            train_data = []
            train_lbls = []
            test_data = []
            test_lbls = []

            cntr = 0
            with open(self.reference_path, 'r', encoding='utf-8') as lbl_file:
                lines = [l for l in lbl_file]

            random.shuffle(lines)
            for line in lines:
                line,lbl = line.split('\t')

                tokens = tokenizer(line)
                sentence = [t for t in tokens if not t in stopwords and t in w2v_vectors.vocab]

                if len(sentence) == 0:
                    continue

                count_vectorized = self.cv.transform([' '.join(sentence)])
                topic_dist = self.model.transform(count_vectorized)
                best_topic_idxs = []
                for _ in range(0):
                    best_topic_idx = topic_dist.argmax(axis=1)
                    best_topic_idxs.append(best_topic_idx)
                    #print(best_topic_idx, " ", topic_dist[0][best_topic_idx])
                    topic_dist[0][best_topic_idx] = 0

                # just use average of word vectors to represent sentence
                # it would be better to use tf-idf too, but I'm not sure how to get those numbers
                # avg_vector = w2v_vectors[sentence[0]]
                # for i in range(1,len(sentence)):
                #     avg_vector = avg_vector + w2v_vectors[sentence[i]]
                #
                # avg_vector = avg_vector / len(sentence)
                #
                # augmented = np.zeros([len(avg_vector)+1])
                # augmented[:-1] = avg_vector
                # augmented[-1:] = best_topic_idxs

                #vector_to_use = augmented
                #vector_to_use = avg_vector
                vector_to_use = topic_dist[0]

                if cntr % 5 == 4:
                    test_data.append(vector_to_use)
                    test_lbls.append(lbl)
                else:
                    train_data.append(vector_to_use)
                    train_lbls.append(lbl)

                cntr += 1

            classifier = lr()
            classifier.fit(train_data, train_lbls)
            predictions = classifier.predict(test_data)
            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == test_lbls[i]:
                    correct += 1

            acc = correct / len(predictions)

            if score_file is None:
                output_file = os.path.join(self.output_path, 'acc.txt') \
                    if iteration_counter is None \
                    else os.path.join(self.output_path, 'acc_' + str(iteration_counter) + '.txt')

                with open(output_file, 'w', encoding='utf-8') as output:
                    output.write(str(acc))
                    output.write('\n')
            else:
                score_file.write(str(acc))
                score_file.write('\n')
                score_file.flush()

            print(acc)
