import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

vocab_dict = [word.decode() for word in list(np.load('../data/vocab20k.npy'))]
#test_file = r"D:\PyCharm Projects\py-sam-master\topic-eval\README.md"
path = r"..\data\corpus\enron_clean"
KEEP_IDS = True

def recurse_dir(file_path):
    if os.path.isdir(file_path):
        for d, paths, files in (os.walk(file_path)):
            for f in files:
                print(f)
                path = os.path.join(d, f)
                remove_non_words(path, keep_ids=KEEP_IDS)

def remove_non_words(file_path, encoding='utf-8', keep_ids=False):
    with open(file_path, mode='r', encoding=encoding) as f:
        if keep_ids:
            text = f.readlines()
            tokenized = []
            for l in text:
                tokens = word_tokenize(l.lower())
                id = tokens[0]
                tokens = [x for x in tokens[1:] if x in vocab_dict]
                tokens.insert(0, id)
                tokenized.append(tokens)

            write_lines(tokenized, file_path)

        else:
            text = f.read()
            tokens = word_tokenize(text.lower())
            tokens = [x for x in tokens if x in vocab_dict]
            write_out(tokens, file_path)

def write_out(l, file_path, encoding = 'utf-8'):
    with open(file_path, mode='w', encoding=encoding) as f:
        for i in l:
            f.write(i+ " \n")

def write_lines(lines, file_path, encoding='utf-8'):
    with open(file_path, mode='w', encoding=encoding) as f:
        for l in lines:
            f.write(l[0] + '\t')
            for t in l[1:]:
                f.write(t + ' ')
            f.write('\n')

#remove_non_words(test_file)
recurse_dir(path)
