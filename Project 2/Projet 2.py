# -*- coding: utf-8 -*-
"""
Created on Fri May  7 13:36:44 2021

@author: aurel
"""

#Generates  a  training  and  a  test  set  of  1,000  points  sampled  uniformly  in  [0,1]2,  
# each  with  alabel 0 if outside the disk centered at (0.5,0.5) of radius 1/√2π, and 1 inside

def generate_disc_set(nb):
    