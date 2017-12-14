import algorithms.ankura.ankura.pipeline as pipeline
import algorithms.ankura.ankura.anchor as anchor
import algorithms.ankura.ankura.corpus as corpus
import algorithms.ankura.ankura.validate as validate
import os

from algorithms.algorithm import Algorithm

class SupervisedAnchorWords(Algorithm):
    def load_input(self):
        input_files = []
        for root,dirs,files in os.walk(self.input_path):
            input_files.extend([os.path.join(root,f) for f in files if f.endswith('_sentences.txt')])

        stopwords = list(open(r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\english_stopwords.txt', 'r', encoding='utf-8'))

        p = pipeline.Pipeline(
            pipeline.file_inputer(*input_files),
            pipeline.whole_extractor(),
            pipeline.stopword_tokenizer(pipeline.default_tokenizer(), stopwords=stopwords),
            pipeline.noop_labeler(),
            pipeline.keep_filterer()
        )

        # p.tokenizer = pipeline.frequency_tokenizer(p, rare=100, common=1000)

        # use pickle_path and docs_path to reduce memory load
        self.corpus = p.run(pickle_path=r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\crowdflower_results\batch_10\20171202\aw_test\enron.pickle', docs_path=r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\crowdflower_results\batch_10\20171202\aw_test\enron_docs.str')
        self.reference = corpus.newsgroups()

    def run(self):
        self.topics = anchor.anchor_algorithm(self.corpus, 100)

    def write_output(self):
        with open(os.path.join(self.output_path, 'anchor_words_output.txt'), mode='w', encoding='utf-8') as output:
            for t in self.topics:
                output.write(str(t))
                output.write('\n')

    def evaluate(self):
        scores = validate.coherence(self.reference, self.topics)
        print("Scores:")
        print(scores)