import pickle
import json
import os

ROOT = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main'
split_file = '/mnt1/myeongah/Test/Implicit3DUnderstanding-main/data/sunrgbd/splits/train.json'
with open(split_file) as file:
    split = json.load(file)

a = []
b = []
c = []
d = []
f = open(os.path.join(ROOT, split[0][2:]), 'rb')
sequence = pickle.load(f)
for key in sequence:
    a.append(key)
    if key == 'boxes':
        for subkey in sequence[key]:
            b.append(subkey)
    elif key == 'camera':
        for subkey in sequence[key]:
            c.append(subkey)
    elif key == 'layout':
        for subkey in sequence[key]:
            d.append(subkey)
for ff in split:
    f = open(os.path.join(ROOT, ff[2:]), 'rb')
    sequence = pickle.load(f)
    for aa in a:
        if aa not in sequence:
            print('%s: %s'%(ff, aa))
    for bb in b:
        if bb not in sequence['boxes']:
            print('%s: %s' % (ff, bb))
    for cc in c:
        if cc not in sequence['camera']:
            print('%s: %s' % (ff, cc))
    for dd in d:
        if dd not in sequence['layout']:
            print('%s: %s' % (ff, dd))