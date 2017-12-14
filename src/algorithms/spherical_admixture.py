import algorithms.pysam.sam
from algorithms.pysam.sam.corpus.corpus import CorpusReader

from algorithms.algorithm import Algorithm
# never got very far with this one, obviously
class SAM(Algorithm):
    def __init__(self, input):
        Algorithm.__init__(self, input)

    def run(self):
        reader = CorpusReader(filename=r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\corpora\enron\maildir\allen-p\all_documents\1_sentences.txt')
        data = reader.read_doc(0)
        doc_label = reader.labels(0)
        doc_name = reader.names(0)
        class_names_of_doc_1 = reader.class_names[reader.labels[1]]
