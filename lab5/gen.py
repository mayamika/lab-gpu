#!/usr/bin/python3

l = list(range(-2500, 2500))
n = len(l)
with open('in',  'w') as f:
    f.write(str(n) + ' ')
    for x in l:
        f.write(str(x) + ' ')
