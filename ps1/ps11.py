#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:46:34 2019

@author: krishna
"""

import nltk
import re
from urllib import request
import sys
def listCompression(rawData):
    """
    Used to design and expore two list comphrension statements
    """
    
    
def regexManipulation(rawData):
    """
    regex manipulation
    """
    
    

if __name__== "__main__":
    """
    Main function to call list and regex 
    """
    url='https://www.gutenberg.org/files/219/219-0.txt' #sys.argv[1]
    response= request.urlopen(url)
    raw = response.read().decode('utf8')
    #listCompression(raw)
    #regexManipulation(raw)
