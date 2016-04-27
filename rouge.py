#! /usr/bin/env python2.7
# -*- coding: utf-8 -*-

import pyrouge
import json

if __name__ == "__main__":
    # You need to modify the following directories 
    # to make pyrouge run properly on your machine
    rouge_dir = './RELEASE-1.5.5'
    rouge_args = '-e ./RELEASE-1.5.5/data -n 4 -m -2 4 -u -c 95 -r 1000 -f A -p 0.5 -t 0 -a -x -l 100'
     
    rouge = pyrouge.Rouge155(rouge_dir, rouge_args)
   
    # 'model' refers to the human summaries 
    rouge.model_dir = './Human_Summaries/eval/'
    rouge.model_filename_pattern = 'D3#ID#.M.100.T.[A-Z]'
    
    # 'system' or 'peer' refers to the system summaries
    # Options are: Centroid, DPP, ICSISumm, LexRank, Submodular
    rouge.system_dir = './System_Summaries/'
    rouge.system_filename_pattern = 'd3(\d+)t'
    
    rouge_output = rouge.evaluate()    
    output_dict = rouge.output_to_dict(rouge_output)
    
    print json.dumps(output_dict, indent=2, sort_keys=True)
