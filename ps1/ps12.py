#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 13:27:37 2019

@author: krishna
"""

from nltk.metrics.distance import edit_distance
import re
from nltk.stem import WordNetLemmatizer
wnl=WordNetLemmatizer()
def calEditDistance(barK, truP):
    dist=edit_distance(barK,truP)
    return dist


def shortCorpus(barK,truP):
    appStart='that you work'
    appEnd='respect'
    barK=  re.search(r'(?<= that you work)(.*)(?=respect)',barK).group(1)
    barK= appStart+ barK + appEnd 
    truP=  re.search(r'(?<= that you work)(.*)(?=respect)',truP).group(1)
    truP= appStart+ truP + appEnd 
    dist =edit_distance(barK,truP)
    
    #barK=re.split(r' ', barK)
    #barK=re.split(r'[ \t\n]+', barK) 
    #Tokenization
    
    barK=barK.split(' ')
    barkStr=[wnl.lemmatize(w) for w in barK]
    setbarkStr=set(barkStr)
    print(setbarkStr)
    
    truP=truP.split(' ')
    trupStr=[wnl.lemmatize(w) for w in truP]
    settrupStr=set(trupStr)
    print(settrupStr)
    diffString= settrupStr & setbarkStr
    print(diffString)
    
    return dist
    
    
if __name__== "__main__":
    """
    Main function
    """
    barack=open('/home/krishna/Desktop/barack.txt','r').read()
    trump=open('/home/krishna/Desktop/trump.txt','r').read()
    returnDistance= calEditDistance(barack,trump)
    print("Full Paragraph Differece= {}".format(returnDistance))
    shortDistance=shortCorpus(barack,trump)
    print("Substring Differece= {}".format(shortDistance))
    
    