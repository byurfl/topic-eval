from src.algorithms.sam.model import SAM
from src.algorithms.algorithm import Algorithm
from src.algorithms import anchor_words
import src.algorithms.ankura.validate as validate
from sklearn.linear_model import LogisticRegression as lr
import os, random
import numpy as np
import gensim
import csv

class SphericalAdmixture(Algorithm):
    def load_input(self, num_of_topics = 10):
        # input is loaded as part of model initialization
        self.model = SAM(self.input_path, num_of_topics, output = self.output_path)
        self.num_of_topics = num_of_topics
    def run(self):
        self.model.run()
        self.topics = self.model.get_topics()

    def write_output_old(self, iteration_counter=None, as_row = True):
        output_file = os.path.join(self.output_path, 'topics.txt') \
            if iteration_counter is None \
            else os.path.join(self.output_path, 'topics_' + str(iteration_counter) + '.txt')

        with open(output_file, 'w', encoding='utf-8') as output:

            for row in self.topics:
                output.write('[')
                output.write("'" + str(row[0]) + "'")
                for i in range(1, len(row)):
                    output.write(" '")
                    output.write(str(row[i]))
                    output.write("'")
                output.write(']\n')

    def write_output(self, iteration_counter=None):
        self.write_topics(iteration_counter=iteration_counter)

    def write_topics(self, iteration_counter=None, pivot_table = False):
        #print(self.topics)
        output_file = os.path.join(self.output_path, 'topics.csv') \
            if iteration_counter is None \
            else os.path.join(self.output_path, 'topics_' + str(iteration_counter) + '.csv')

        with open(output_file, 'w', newline='') as out:
            csv_out = csv.writer(out)
            if pivot_table:
                csv_out.writerow(['topic', 'name', 'num'])
                for t, topic in enumerate(self.topics):
                    for word in topic:
                        csv_out.writerow(["TOPIC " + str(t)] + list(word))
            else:
                csv_out.writerow(['name', 'num'] * self.num_of_topics)
                output_topics = self.topics[:]

                # Verify sort
                for topic in output_topics:
                    topic.sort(key=lambda x: -x[1])

                output = zip(*output_topics)
                for word_row in output:
                    csv_out.writerow([i for sub in word_row for i in sub])

    def eval_methods(self):
        return ['coherence', 'accuracy']

    def evaluate(self, method, score_file, iteration_counter=None, top_topic_words_ct = 10):
        if method == 'coherence':
            # coherence method stolen from ankura, just to make sure
            # the results are comparable to the topic modeling we did with ankura
            aw = anchor_words.AnchorWords(self.input_path, None, None, self.filter_string)
            aw.load_input()

            words_to_index = dict()
            for i in range(len(aw.corpus.vocabulary)):
                word = aw.corpus.vocabulary[i]
                words_to_index[word] = i

            topic_summary = np.array([[words_to_index[w] for w,v in topic] for topic in self.topics])
            topic_summary = topic_summary[:,:top_topic_words_ct]
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


def delete():
    for d,s,fs in os.walk(r"D:\PyCharm Projects\py-sam-master\topic-eval\data\corpus\very_mini_news"):
        for f in fs:
            path = os.path.join(d,f)
            os.rename(path, path+".txt")
            #os.remove(path)
#delete()


