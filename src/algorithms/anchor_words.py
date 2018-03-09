import src.algorithms.ankura.pipeline as pipeline
import src.algorithms.ankura.anchor as anchor
import src.algorithms.ankura.corpus as corpus
import src.algorithms.ankura.validate as validate
import src.algorithms.ankura.topic as topics

from sklearn.linear_model import LogisticRegression as lr
import os,regex,gensim,random, operator
import numpy as np

from src.algorithms.algorithm import Algorithm


class AnchorWords(Algorithm):
    def load_input(self):
        pipe = None
        input = self.get_files(self.input_path)

        stopwords = [w.strip() for w in open(r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\english_stopwords.txt', 'r', encoding='utf-8')]

        inputter = pipeline.file_inputer(*input)
        extractor = pipeline.whole_extractor() if os.path.isdir(self.input_path) else pipeline.line_extractor(delim='\t')
        tokenizer = pipeline.stopword_tokenizer(pipeline.default_tokenizer(), stopwords=stopwords)
        labeler = pipeline.noop_labeler()
        filterer = pipeline.keep_filterer()

        pipe = pipeline.Pipeline(
            inputter,
            extractor,
            tokenizer,
            labeler,
            filterer
        )
        # pipe.tokenizer = pipeline.frequency_tokenizer(p, rare=100, common=1000)

        # use pickle_path and docs_path to reduce memory load
        # unless no output path was given
        if self.output_path == None:
            self.corpus = pipe.run()
        else:
            self.corpus = pipe.run(pickle_path=os.path.join(self.output_path, 'enron.pickle'), docs_path=os.path.join(self.output_path, 'enron_docs.str'))

        if self.reference_path == None:
            self.reference = self.corpus
        else:
            reference = self.get_files(self.reference_path)
            p2 = pipeline.Pipeline(
                pipeline.file_inputer(*reference),
                pipeline.whole_extractor(),
                pipeline.stopword_tokenizer(pipeline.default_tokenizer(), stopwords=stopwords),
                pipeline.noop_labeler(),
                pipeline.keep_filterer()
            )
            self.reference = p2.run(pickle_path=r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\corpora\ohsumed-all\ohsumed.pickle', docs_path=r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\corpora\ohsumed-all\ohsumed.str')

    def run(self):
        self.topics = anchor.anchor_algorithm(self.corpus, 100, doc_threshold=10)

    def write_output(self, iteration_counter=None):
        output_file = os.path.join(self.output_path, 'topics.txt') \
            if iteration_counter == None \
            else os.path.join(self.output_path, 'topics_' + str(iteration_counter) + '.txt')

        with open(output_file, 'w', encoding='utf-8') as output:
            output.write(str(self.topics))
            output.write('\n')

    def eval_methods(self):
        return ['coherence', 'accuracy']

    def evaluate(self, method, score_file, iteration_counter=None):
        if method == 'coherence':
            scores = validate.coherence(self.reference, topics.topic_summary(self.topics))

            scores_file = os.path.join(self.output_path, 'coherence.txt') \
                if iteration_counter == None \
                else os.path.join(self.output_path, 'coherence_' + str(iteration_counter) + '.txt')

            with open(scores_file, mode='w', encoding='utf-8') as output:
                output.write(str(scores))
                output.write('\n')

            if score_file != None:
                score_file.write(str(np.average(scores)))
                score_file.write('\n')
                score_file.flush()

            print("Scores:")
            print(scores)

        elif method == 'accuracy':
            w2v_model = gensim.models.Word2Vec.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)
            w2v_vectors = w2v_model.wv
            del w2v_model

            train_data = []
            train_lbls = []
            test_data = []
            test_lbls = []

            all_lbls = []
            for line in list(open(self.reference_path, 'r', encoding='utf-8')):
                all_lbls.append(line.strip().split('\t')[1])

            i = 0
            doc_list = []
            for doc in self.corpus.documents:
                doc.metadata['label'] = all_lbls[i]
                doc_list.append(doc)
                i += 1

            cntr = 0
            random.shuffle(doc_list)
            for doc in doc_list:
                doc_topics = []
                for t in doc.tokens:
                    # if t.token in self.corpus.vocabulary:
                    topic_dist = self.topics[t.token]
                    best_topic_idx = np.argmax(topic_dist)
                    doc_topics.append(best_topic_idx)

                if len(doc_topics) == 0:
                    continue

                topic_counts = {}
                for topic in doc_topics:
                    if topic in topic_counts:
                        topic_counts[topic] += 1
                    else:
                        topic_counts[topic] = 1

                normalized_topics = {}
                for topic in topic_counts:
                    normalized_topics[topic] = topic_counts[topic] / len(doc_topics)

                average_topic_vector = self.topics[doc_topics[0]]
                for t in range(1, len(doc_topics)):
                    average_topic_vector += self.topics[doc_topics[t]] / len(doc_topics)

                sorted_topics = sorted(normalized_topics.items(), key=operator.itemgetter(1))

                tokens = [self.corpus.vocabulary[t.token] for t in doc.tokens if self.corpus.vocabulary[t.token] in w2v_vectors.vocab]
                if len(tokens) == 0:
                    continue

                average_word_vector = w2v_vectors[tokens[0]]
                for i in range(1, len(tokens)):
                    average_word_vector = average_word_vector + w2v_vectors[tokens[i]]
                average_word_vector = average_word_vector / len(tokens)

                #vector.append(sorted_topics[0][0]) # add most-likely topic
                #vector.append(sorted_topics[1][0]) # add second-most-likely topic
                vector = average_topic_vector # whole distribution
                #vector = average_word_vector # word embedding

                # topics + embeddings
                # vector = np.zeros(len(average_word_vector) + len(average_topic_vector))
                # vector[:-len(average_topic_vector)] = average_word_vector
                # vector[-len(average_topic_vector):] = average_topic_vector

                if cntr % 5 == 4:
                    test_data.append(vector)
                    test_lbls.append(doc.metadata['label'])
                else:
                    train_data.append(vector)
                    train_lbls.append(doc.metadata['label'])

                cntr += 1

            classifier = lr()
            classifier.fit(train_data, train_lbls)
            predictions = classifier.predict(test_data)

            correct = 0
            for i in range(len(predictions)):
                if predictions[i] == test_lbls[i]:
                    correct += 1

            acc = correct / len(predictions)

            if (score_file == None):
                output_file = os.path.join(self.output_path, 'acc.txt') \
                    if iteration_counter == None \
                    else os.path.join(self.output_path, 'acc_' + str(iteration_counter) + '.txt')

                with open(output_file, 'w', encoding='utf-8') as output:
                    output.write(str(acc))
                    output.write('\n')
            else:
                score_file.write(str(acc))
                score_file.write('\n')
                score_file.flush()

            print(acc)

