import argparse, os

ITERATIONS = 30

if os.environ["COMPUTERNAME"] == 'DALAILAMA':
    import sys
    path = r"D:\PyCharm Projects\py-sam-master\topic-eval"
    os.environ["HOME"] = r"D:\PyCharm Projects\py-sam-master\topic-eval\data\corpus;"
    sys.path.append(path)
    os.chdir(path)
    ITERATIONS = 1

class AlgorithmEvaluator:
    def __init__(self, algorithm, input_path, output_path, reference_path=None, filter_string=r'_sentences\.txt$'):
        self.input_path = input_path
        self.output_path = output_path
        self.reference_path = reference_path
        self.filter_string = filter_string
        self.select_algorithm(algorithm)

    def select_algorithm(self, alg):
        if alg == 'lda':
            from src.algorithms.lda import LDA
            self.algorithm = LDA(self.input_path, self.output_path, reference_path=self.reference_path)

        elif alg == 'lda_sup':
            from src.algorithms.sup_lda import SupervisedLDA
            self.algorithm = SupervisedLDA(self.input_path, self.output_path)

        elif alg == 'anchors':
            from src.algorithms.anchor_words import AnchorWords
            self.algorithm = AnchorWords(self.input_path, self.output_path, reference_path=self.reference_path, filter_string=self.filter_string)

        elif alg == 'anchors_sup':
            from src.algorithms.anchor_words import SupervisedAnchorWords
            self.algorithm = SupervisedAnchorWords(self.input_path, self.output_path)

        elif alg == 'sam':
            from src.algorithms.spherical_admixture import SphericalAdmixture
            self.algorithm = SphericalAdmixture(self.input_path, self.output_path, reference_path=self.reference_path)

    def run_algorithm(self):
        self.algorithm.run()

    def load_inputs(self):
        self.algorithm.load_input()

    def write_outputs(self, iteration_counter=None):
        self.algorithm.write_output(iteration_counter)

    def evaluate_results(self, method, iteration_counter=None, score_file=None):
        if method in self.algorithm.eval_methods():
            self.algorithm.evaluate(method, score_file, iteration_counter)
        else:
            raise Exception('Unsupported evaluation method ' + method + ' for selected algorithm.\nOptions are: ' + str(self.algorithm.eval_methods()))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parse arguments
    #####
    # algorithm: lda, lda_sup, anchors, anchors_sup, sam
    # input path
    # output folder
    # evaluation method: classification accuracy, F-measure, coherence (unsupervised approaches only)
    #####
    parser.add_argument('--algorithm', required=True, choices=['lda', 'lda_sup', 'anchors', 'anchors_sup', 'sam'], help='Algorithm to use. Options are lda, lda_sup, anchors, anchors_sup, sam. **Note that SAM is still unimplemented.')
    parser.add_argument('--input', required=True, help='Path to folder containing input documents, or to a single document containing an entire corpus.')
    parser.add_argument('--filter', required=False, help=r'Regular expression to filter input files with. By default, no filter is used.')
    parser.add_argument('--output', required=True, help='Path to a folder to receive output documents')
    parser.add_argument('--reference', required=False, help='Path to a folder containing reference documents, or to a single document containing an entire reference corpus.')
    parser.add_argument('--evaluation', required=True, choices=['accuracy', 'fmeasure', 'coherence'], help='Method for evaluating results. Options are accuracy, fmeasure, coherence. Accuracy and F-measure are for supervised approches, coherence is for unsupervised approaches.')
    #parser.add_argument('--corpus_codec', required=False, help='Codec for corpus.')
    args = parser.parse_args()

    eval = AlgorithmEvaluator(args.algorithm, args.input, args.output, args.reference, args.filter)
    print('Getting data...', end='')
    eval.load_inputs()
    print('done.')

    # Create output if needed
    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    with open(os.path.join(args.output, 'avg_coherences.txt'), 'w', encoding='utf-8') as score_file:
        for i in range(ITERATIONS):
            print('Iteration ' + str(i) + ': Recovering topics...', end='')
            eval.run_algorithm()
            print('done.')

            print('Iteration ' + str(i) + ': Writing topics to file...', end='')
            eval.write_outputs(iteration_counter=i)
            print('done.')

            print('Iteration ' + str(i) + ': Evaluating topic assignments...', end='')
            eval.evaluate_results(args.evaluation, iteration_counter=i, score_file=score_file)
            print('done.')