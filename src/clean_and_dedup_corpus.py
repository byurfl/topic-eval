import os, os.path as path
import nltk.tokenize as tok
import regex
import argparse

import matplotlib.pyplot as plt

def write_batch(batch, batch_start, batch_end, f_sentences, csv):
    # write previous context
    csv.write('"')

    # write batch
    k = 0
    for b in batch:
        k += 1
        if k == 10:
            csv.write(b.replace('"', '""').strip() + '"')

        else:
            csv.write(b.replace('"', '""').strip() + '","')

    csv.write('\n')


parser = argparse.ArgumentParser()
parser.add_argument('root', help='Root directory in which to process files. The subtree rooted at this directory will be searched for files matching the filter.')
parser.add_argument('-s', '--smushlines', action='store_true', help='Option to "smush" all lines in the file down to a single line before extracting sentences.')
parser.add_argument('--threshold', default=3, type=int, help='Minimum length of a sentence (in words) to be kept. Sentences with fewer words than this threshold will be removed from the data.')
parser.add_argument('--keepheader', action='store_true', help='Option to keep the header (everything up to the first blank line. If this is NOT specified, everything in each file up to the first blank line will be discarded.')
parser.add_argument('-f', '--filter', default=r'.txt$', help=r'Regular expression used to filter out files to process. Files whose names match the regex will be processed. Defaults to ".txt$"')
parser.add_argument('-b', '--batchsize', type=int, default=10, help='Size of batches to keep. Documents with fewer than this number of sentences will be filtered out.')

args = parser.parse_args()

ROOT = args.root
CSV_HEADER = 'current_unit_0,current_unit_1,current_unit_2,current_unit_3,current_unit_4,current_unit_5,current_unit_6,current_unit_7,current_unit_8,current_unit_9\n'

singlecsv = open(path.join(ROOT, 'single.csv'), mode='w', encoding='utf-8')
singlecsv.write(CSV_HEADER)

unique_sentences = set()
filesToSentenceLists = {}
emailSentenceCounts = {}
emailUniqueSentenceCounts = {}

'''
read in all files
do some cleaning on each file
create a map of filenames to lists of sentences
'''
for (root, dirs, files) in os.walk(ROOT, topdown=False):
    matching_files = [path.join(root, f) for f in files if regex.search(args.filter, f, regex.UNICODE)]
    for file in matching_files:

        # don't process category labels or prior outputs
        if file.lower().endswith('.cats') or \
            file.lower().endswith('_sentences.txt') or \
            file.lower().endswith('_phrases.txt') or \
            file.lower().endswith('.csv'):

            continue

        else:
            print("Processing " + file)
            # find quoted message text, if any
            with open(file, mode='r', encoding='utf-8') as input:
                # skip everything up to the first blank line
                # (this should get rid of E-mail headers, etc.)
                # unless the user specified the keepheader option
                while (not args.keepheader) and (input.readline().strip() != ""):
                    pass

                text = input.read() + input.read()
                qIdx = -1
                qIdxMatch = regex.search('-{5,}', text, regex.UNICODE)
                if qIdxMatch:
                    qIdx = qIdxMatch.start()

                if qIdx > -1:
                    # remove quoted text of previous E-mails
                    text = text[:qIdx]

                # normalize line endings
                text = text.replace('\r\n', '\n').replace('\r', '\n')

                # remove URLs
                text = regex.sub(r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))',
                                 '', text, flags=regex.UNICODE|regex.IGNORECASE)

                # remove XML tags
                text = regex.sub(r'<[^>]+>', '', text, flags=regex.UNICODE|regex.IGNORECASE)

                # remove weird encoding artifacts
                text = text.replace('=\n', '')
                text = regex.sub(r'=[0-9a-f]{2}', '', text, flags=regex.UNICODE|regex.IGNORECASE)

                # remove > symbols and replace with spaces (whitespace will be normalized later)
                text = text.replace('>', ' ')

                # remove blank lines
                text = regex.sub(r'\n\s*\n', r'\n', text, flags=regex.UNICODE)

                # break up paragraphs into sentences
                sentences = tok.sent_tokenize(text)

                # keep track of sentences per email (may create a histogram or something later)
                sentenceCount = len(sentences)
                if sentenceCount in emailSentenceCounts:
                    emailSentenceCounts[sentenceCount] += 1
                else:
                    emailSentenceCounts[sentenceCount] = 1

                outputFile = file.replace('.txt', '_sentences.txt') if file.endswith('.txt') else file + '_sentences.txt'
                with open(outputFile, mode='w', encoding='utf-8') as output:
                    for s in sentences:
                        if regex.match(r'[\p{Z}\p{P}\p{S}\p{N}]+$', s, regex.UNICODE):
                            continue

                        # remove punctuation and symbols so they aren't counted as tokens
                        nopunc = regex.sub(r'[\p{P}\p{S}]', '', s.strip())
                        tokens = tok.word_tokenize(nopunc)
                        if len(tokens) > args.threshold:
                            # normalize spacing before printing
                            s = regex.sub('\s+', ' ', s)
                            output.write(s.strip() + '\n')

                            if file in filesToSentenceLists:
                                filesToSentenceLists[file].append(s)
                            else:
                                filesToSentenceLists[file] = [s]

'''
now write batches to "master" CSV file
a batch consists of sentences from a single E-mail
size of the batch is determined by the batchsize argument
E-mails with too few sentences will drop out
'''
for f in filesToSentenceLists:
    f_sentences = filesToSentenceLists[f]
    batch = []
    batch_start = -1

    unique_f_sentences = []
    if (len(f_sentences) >= args.batchsize):
    # if True:
        i = 0
        while len(batch) < args.batchsize and i < len(f_sentences):
        # while i < len(f_sentences):
            s = f_sentences[i].strip()
            if s not in unique_sentences:
                unique_f_sentences.append(s)
                batch.append(s)
                unique_sentences.add(s)
                if batch_start == -1:
                    batch_start = i
            i += 1

        if len(batch) >= args.batchsize:
            write_batch(batch, batch_start, i, f_sentences, singlecsv)
            batch = []
            batch_start = -1

    if len(unique_f_sentences) > 0:
        if len(unique_f_sentences) in emailUniqueSentenceCounts:
            emailUniqueSentenceCounts[len(unique_f_sentences)] += 1
        else:
            emailUniqueSentenceCounts[len(unique_f_sentences)] = 1

# close that giant CSV we wrote everything to
if singlecsv != None:
    singlecsv.close()

plt.figure()
plt.xlabel('# of sentences')
plt.ylabel('# of emails')
sentenceCounts = list(emailSentenceCounts.keys())
emails = [emailSentenceCounts[s] for s in sentenceCounts]

with open('enronSentenceCounts.txt', 'w', encoding='utf-8') as data:
    for i in range(len(sentenceCounts)):
        s = sentenceCounts[i]
        e = emails[i]
        data.write(str(s) + '\t' + str(e) + '\n')

plt.plot(sentenceCounts, emails)
plt.show()

plt.figure()
plt.xlabel('# of unique sentences')
plt.ylabel('# of emails')
uniqueSentenceCounts = list(emailUniqueSentenceCounts.keys())
uEmails = [emailUniqueSentenceCounts[s] for s in uniqueSentenceCounts]

with open('enronUniqueSentenceCounts.txt', 'w', encoding='utf-8') as data:
    for i in range(len(uniqueSentenceCounts)):
        s = uniqueSentenceCounts[i]
        e = uEmails[i]
        data.write(str(s) + '\t' + str(e) + '\n')

plt.plot(uniqueSentenceCounts, uEmails)
plt.show()
