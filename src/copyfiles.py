import shutil
import os

ROOT = r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\corpora\enron\maildir'
DEST = r'C:\Users\leer1\Documents\aaPERSONAL\School\CS698R (F2017)\data\eval_test'
copylimit = 400

copy = False
counter = 0
for root, dirs, files in os.walk(ROOT):
    cleanroot = root.replace('\\', '')
    cleanroot = cleanroot.replace('/', '')
    cleanroot = cleanroot.replace(' ', '')
    cleanroot = cleanroot.replace(':', '')

    copydict = {}

    for f in files:
        if f.endswith('_sentences.txt') and os.stat(os.path.join(root, f)).st_size > 0:
            copydict[os.path.join(root,f)] = cleanroot+f

    for c in copydict:
        if copy:
            shutil.copy2(c, os.path.join(DEST, copydict[c]))
            copy = False
            counter = 0

        else:
            counter += 1

        if counter >= copylimit:
            copy = True