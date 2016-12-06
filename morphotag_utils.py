'''
Created on Dec 7, 2016

@author: Yuval Pinter
'''

def split_tagstring(s):
    '''
    Returns attribute-value mapping from UD-type CONLL field
    '''
    ret = {}
    for attval in s.split('|'):
        a,v = attval.strip().split('=')
        ret[a] = v
    return ret