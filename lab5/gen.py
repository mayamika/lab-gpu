#!/usr/bin/python3

from random import shuffle

r = 1024
l = list(range(-r, r))
shuffle(l)
n = len(l)
with open('in',  'w') as f:
    f.write(str(n) + ' ')
    for x in l:
        f.write(str(x) + ' ')
