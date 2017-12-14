import argparse
import os, os.path as path

parser = argparse.ArgumentParser()
parser.add_argument('directory', help='Directory in which to convert file encodings.')
parser.add_argument('-e', '--newencoding', default='utf-8', help='"Destination" encoding. When the script finishes, all files in the ' +
                                                'specified directory should have this encoding.')
args = parser.parse_args()

if path.isdir(args.directory):
    # objs = os.listdir(args.directory)
    # files = [path.join(args.directory, f) for f in objs if not path.isdir(f)]
    for (root, dirs, files) in os.walk(args.directory):
        for f in files:
            f = path.join(root, f)
            print('Checking encoding in ' + f)
            try:
                with open(f, mode='rt', encoding=args.newencoding) as input:
                    input.read()

            except:
                print('Encoding does not match expected encoding, attempting to convert...')
                # so far all the problems have been ANSI, so assume ANSI encoding
                with open(f, mode='rt', encoding='mbcs') as input:
                    text = input.read()

                with open(f, mode='wt', encoding=args.newencoding) as output:
                    output.write(text)
                    print("Done.")

else:
    raise 'Invalid directory: ' + args.directory