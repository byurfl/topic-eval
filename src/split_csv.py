import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('input', help='Input CSV file to split.')
parser.add_argument('-c', '--count', type=int, help='Number of files to split <input> into. Must specify either this or maxsize. If both are specified, count takes precedence.')
parser.add_argument('-m', '--maxsize', type=int, help='Maximum size (in lines) allowed for a single file. Must specify either this or count. If both are specified, count takes precedence.')
args = parser.parse_args()

with open(args.input, 'r', encoding='utf-8') as input:
    lines = input.readlines()
    header = lines.pop(0).strip()

if args.count:
    lineCnt = len(lines)
    linesPerFile = math.ceil(lineCnt / args.count)
    for i in range(args.count):
        chunk = lines[i*linesPerFile:(i+1)*linesPerFile]
        with open(args.input.replace('.csv', '_{}.csv'.format(i)), 'w', encoding='utf-8') as chunk_out:
            chunk_out.write(header + '\n')
            for c in chunk:
                chunk_out.write(c.strip() + '\n')


elif args.maxsize:
    pass

else:
    raise Exception('Must specify either count or maxsize option.')

