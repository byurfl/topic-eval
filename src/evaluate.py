import argparse, os

class AlgorithmEvaluator:
    def __init__(self, algorithm, input_path, output_path, reference_path=None, filter_string=r'_sentences\.txt$'):
        self.input_path = input_path
        self.output_path = output_path
        self.reference_path = reference_path
        self.filter_string = filter_string
        self.select_algorithm(algorithm)

    def select_algorithm(self, alg):
        if alg == 'lda':
            from algorithms.lda import LDA
            self.algorithm = LDA(self.input_path, self.output_path, reference_path=self.reference_path)

        elif alg == 'lda_sup':
            from algorithms.sup_lda import SupervisedLDA
            self.algorithm = SupervisedLDA(self.input_path, self.output_path)

        elif alg == 'anchors':
            from algorithms.anchor_words import AnchorWords
            self.algorithm = AnchorWords(self.input_path, self.output_path, self.reference_path, self.filter_string)

        elif alg == 'anchors_sup':
            from algorithms.anchor_words import SupervisedAnchorWords
            self.algorithm = SupervisedAnchorWords(self.input_path, self.output_path)

        elif alg == 'sam':
            from algorithms.spherical_admixture import SAM
            self.algorithm = SAM(self.input_path, self.output_path)

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
    parser.add_argument('--filter', required=False, default=r'_sentences\.txt', help=r'Regular expression to filter input files with. Defaults to "_sentences\.txt$".')
    parser.add_argument('--output', required=True, help='Path to a folder to receive output documents')
    parser.add_argument('--reference', required=False, help='Path to a folder containing reference documents, or to a single document containing an entire reference corpus.')
    parser.add_argument('--evaluation', required=True, choices=['accuracy', 'fmeasure', 'coherence'], help='Method for evaluating results. Options are accuracy, fmeasure, coherence. Accuracy and F-measure are for supervised approches, coherence is for unsupervised approaches.')
    args = parser.parse_args()

    eval = AlgorithmEvaluator(args.algorithm, args.input, args.output, args.reference, args.filter)
    print('Getting data...', end='')
    eval.load_inputs()
    print('done.')

    with open(os.path.join(args.output, 'avg_coherences.txt'), 'w', encoding='utf-8') as score_file:
        for i in range(30):
            print('Iteration ' + str(i) + ': Recovering topics...', end='')
            eval.run_algorithm()
            print('done.')

            print('Iteration ' + str(i) + ': Writing topics to file...', end='')
            eval.write_outputs(iteration_counter=i)
            print('done.')

            print('Iteration ' + str(i) + ': Evaluating topic assignments...', end='')
            eval.evaluate_results(args.evaluation, iteration_counter=i, score_file=score_file)
            print('done.')