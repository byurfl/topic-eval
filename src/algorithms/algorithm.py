import os, regex, string

class Algorithm:
    def __init__(self, input_path, output_path, reference_path=None, filter_string=r'_sentences\.txt$'):
        self.input_path = input_path
        self.output_path = output_path
        self.reference_path = reference_path
        self.filter_string = filter_string

    def load_input(self):
        pass

    def run(self):
        pass

    def write_output(self, iteration_counter=None):
        pass

    def eval_methods(self):
        return []

    def evaluate(self, method, score_file, iteration_counter=None):
        pass

    def get_files(self, path):
        if os.path.isdir(path):
            corpus_files = []
            filter_regex = regex.compile(self.filter_string, flags=regex.UNICODE)
            for root,dirs,files in os.walk(path):
                corpus_files.extend([os.path.join(root,f) for f in files if filter_regex.search(f)])
            return corpus_files

        elif os.path.isfile(path):
            return [path]

        else:
            return None

    def get_tokenizer(self, remove_labels=True):
        delimiter = string.whitespace
        translation = str.maketrans(string.ascii_letters,
                                      string.ascii_lowercase * 2,
                                      string.punctuation)

        def _tokenize(_string):
            if remove_labels:
                tabloc = _string.find('\t')
                _string = _string[tabloc+1:]
            tokens = _string.split()
            tokens_trans = [t.translate(translation) for t in tokens]
            return [t for t in tokens_trans if t != '' and t != None]

        return _tokenize