from __future__ import print_function
import evaluate
from evaluate import *


import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

# other modules:
# regex, gensim

data_folder = r"D:\OneDrive\Documents\Graduate School\2018 Winter\CS 678\Project\topic-eval-master\topic-eval-master\data"

_algorithm = "sam"
_algorithm = "lda"
#_input = os.path.join(data_folder, "enron_sentences_with_labels.txt")
_input = os.path.join(data_folder, "enron_sentences_short.txt")
_output = "../output"
_reference = os.path.join(data_folder, "enron_with_labels\enron_sentences_with_labels_short.txt")
_filter = ""
_evaluation = 'accuracy' # choices= 'accuracy', 'fmeasure', 'coherence'

#eval = AlgorithmEvaluator(args.algorithm, args.input, args.output, args.reference, args.filter)
eval = AlgorithmEvaluator(_algorithm, _input, _output, _reference)
eval.load_inputs()

print('done.')

# args.evaluation
# n_topics => n_components

with open(os.path.join(_output, 'avg_coherences.txt'), 'w', encoding='utf-8') as score_file:
    for i in range(1):
        print('Iteration ' + str(i) + ': Recovering topics...', end=' ')
        eval.run_algorithm()
        print('done.')

        print('Iteration ' + str(i) + ': Writing topics to file...', end='')
        eval.write_outputs(iteration_counter=i)
        print('done.')

        print('Iteration ' + str(i) + ': Evaluating topic assignments...', end='')
        eval.evaluate_results(_evaluation, iteration_counter=i, score_file=score_file)
        print('done.')

