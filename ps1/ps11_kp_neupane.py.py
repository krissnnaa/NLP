#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 13:46:34 2019

@author: krishna

Ref:https://www.nltk.org/book/
"""

import nltk
import re
from urllib import request
import sys
def listCompression(rawData):
    """
    Used to design and expore two list comphrension statements
    compute modal verbs in the corpus
    """
    wordData=rawData.split(' ')

    #First list compression: Lower first ten words and List first using original plain text
    wordList1=[w.lower() for w in wordData[:10]]
    print(wordList1)
    #Second list compression: To grab the modal verbs in the given plain text
    modal = ['can', 'may', 'must', 'might', 'will', 'would', 'should']
    wordList2=[w for w in wordData if w in modal]
    print(wordList2)
    
def regexManipulation(rawData):
    """
    regex manipulation
    """
    # match : To do the exact match i.e. form the beginning of the string
    exactMatch=re.match(r'book',rawData)
    print(exactMatch)
    # search: To match everywhere (from start, in-between of the string)as much as possible
    matchList=[]
    for w in rawData.split(' '):
        everyMatch=re.search(r'book',w)
        if everyMatch !=None:
            matchList.append(w)
    print(matchList)

if __name__== "__main__":
    """
    Main function to call list and regex 
    """
    url= sys.argv[1] #'http://www.gutenberg.org/files/219/219.txt'
    response= request.urlopen(url)
    raw = response.read().decode('utf8')
    listCompression(raw)
    regexManipulation(raw)
