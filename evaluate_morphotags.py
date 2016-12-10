'''
Created on Nov 30, 2016
Follows <gold,observed> pairs, each <attribute, value> and produces macro or micro f1 scores,
either by attribute alone (pooled over values) or by attribute-value combination.

@author: Yuval Pinter
'''
from __future__ import division
from numpy import average

def f1(corr, gold, obs):
    if gold <= 0 or obs <= 0:
        return 0
    rec = corr / gold
    pre = corr / obs
    return (2 * rec * pre) / (rec + pre)

class Evaluator(object):
    '''
    classdocs
    '''   
    
    def __init__(self, m='att'):
        '''
        values for m (mode):
        att - pool scores by attribute over values
        att_val - separate scores for each <attribute, value> pair
        exact - only compute accuracy for full tag (all attributes in instance)
        '''
        self.instance_count = 0
        self.exact_match = 0
        self.correct = {}
        self.gold = {}
        self.observed = {}
        self.mode = m
        
    def add_instance(self, g, o):
        '''
        g - gold annotation for instance
        o - observed (inferred) annotation for instance
        '''
        self.instance_count = self.instance_count + 1
        if self.mode == 'exact':
            if g == o: # order-insensitive
                self.exact_match = self.exact_match + 1
            return
            
        for (k, v) in g.items():
            key = self._key(k, v)
            if o.get(k, 'NOT A VALUE') == v:
                self.correct[key] = self.correct.get(key, 0) + 1  # for macro-micro
            self.gold[key] = self.gold.get(key, 0) + 1  # mac-mic
            
        for (k, v) in o.items():
            key = self._key(k, v)
            self.observed[key] = self.observed.get(key, 0) + 1  # mac-mic
            
    def _key(self, k, v):
        if self.mode == 'att':
            return k
        if self.mode == 'att_val':
            return (k,v)
    
    def mic_f1(self):
        return f1(sum(self.correct.values()), sum(self.gold.values()), sum(self.observed.values()))
    
    def mac_f1(self):
        all_keys = set().union(self.gold.keys(), self.observed.keys())
        return average(map(lambda k : f1(self.correct.get(k, 0), self.gold.get(k, 0), self.observed.get(k, 0)), all_keys))
    
    def acc(self):
        if self.instance_count <= 0:
            return 0.0
        return self.exact_match / self.instance_count
