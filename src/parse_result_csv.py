import argparse
import csv

'''
This script takes in an aggregated CSV file produced by CrowdFlower.
It produces two output files:
  1. a CSV file consisting of only the rows classified as confidential
  2. a text file consisting of sentences and their labels (both confidential and non-confidential), 
        one pair per line, separated by a tab character
'''

parser = argparse.ArgumentParser()
parser.add_argument('--input', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--agreements', type=int, default=2)
args = parser.parse_args()

csvReader = csv.reader(open(args.input))
csvWriter = csv.writer(open(args.input.replace('.csv', '_out.csv'), 'w', encoding='utf-8'))
agreements = {}

with open(args.output, 'w', encoding='utf-8') as output:

    for row in csvReader:
        id = row[0]
        if id == '_unit_id':
            # this is the header
            csvWriter.writerow(row)
            continue

        user_answers = row[5].splitlines()
        answers = {}
        for answer_string in user_answers:
            answer_list = answer_string.split('|')
            for a in answer_list:
                if a == "none":
                    continue

                if a in answers:
                    answers[a] += 1
                else:
                    answers[a] = 1

        write = False
        idxs = set()
        for a in sorted(answers.keys()):
            sent_idx = int(a.split('_')[1]) + 5
            idxs.add(sent_idx)
            sentence = row[sent_idx]
            output.write('"' + sentence.replace('"', '""') + '"\t"')

            if answers[a] >= args.agreements:
                output.write('CONFIDENTIAL"\n')

                if id in agreements:
                    agreements[id]['count'] += 1
                else:
                    agreements[id] = {'count':1,'is_none':False}

                if a == "none":
                    agreements[id]['is_none'] = True
                else:
                    write = True

            else:
                output.write('PUBLIC"\n')

        for i in range(6, 16):
            if i not in idxs:
                output.write('"' + row[i].replace('"', '""') + '"\t"PUBLIC"\n')

        if write:
            csvWriter.writerow(row)

    for a in agreements:
        print(a, '\t', agreements[a]['count'], '\t', agreements[a]['is_none'])
