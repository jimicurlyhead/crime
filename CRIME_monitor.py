#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from glob import glob
import os
import time


def digitformatter(x, n_dig):
    '''rounds a given float to n_dig digits after the comma and returns the
    resulting number in string format including trailing zeros'''
    x_r = round(x, n_dig)
    x_rs = str(x_r)
    n_digr = len(x_rs.split('.')[1])
    return x_rs + '0'*(n_dig - n_digr)


#find all snapshot files of TIPTOE reconstruction runs:
files = glob('*.snp')

#parse chi squared values:
d_chi = {}
d_state = {}
now = time.time()
for f in files:
    
    #get time of last modification:
    mtime_f = os.path.getmtime(f)
    
    #store chi squared value for recently updated, running jobs:
    if now - mtime_f < 300:
        with open(f, 'r') as txt:
            lines = txt.readlines()
            state = lines[0].split(':')[1].strip()
            chi = lines[2].split(':')[1].strip()
            if state == 'Optimization terminated successfully.':
                state = 'finished'
            name = f.split('.snp')[0]
            d_chi[name] = round(float(chi), 4)
            d_state[name] = state
                
#print summary:
names = list(d_chi.keys())
names.sort()
sep = ' '*3
if len(names):
    n_name = max([len(name) for name in names])
    out = '  ' + sep + 'name'.ljust(n_name) + sep + 'opt. target' + sep + '  status' + '\n\n'
    for i,name in enumerate(names):
        chi_ns = digitformatter(d_chi[name], 4)
        out += str(i+1).rjust(2) + sep + name.ljust(n_name) + chi_ns.rjust(14) + d_state[name].rjust(11) + '\n'
    print(out)
else:
    print('no active jobs')
