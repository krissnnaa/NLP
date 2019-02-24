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
    barK = re.sub(r'[^\w]', ' ', barK)
    truP= re.sub(r'[^\w]', ' ', truP)
    dist =edit_distance(barK,truP)

    barK=barK.split(' ')
    barkStr=[wnl.lemmatize(w) for w in barK]
    setbarkStr=list(set(barkStr))
    setbarkStr=setbarkStr[1:]
    print(setbarkStr)

    truP=truP.split(' ')
    trupStr=[wnl.lemmatize(w) for w in truP]
    settrupStr=list(set(trupStr))
    settrupStr=settrupStr[1:]
    lenBarack=setbarkStr.__len__()
    lenTrump=settrupStr.__len__()
    setbarkStr=set(setbarkStr)
    settrupStr=set(settrupStr)
    print(settrupStr)
    print('Total words in Barack=%d and in Trump=%d'%(lenBarack,lenTrump))
    sameString= settrupStr & setbarkStr
    print(sameString)
    print("Length of same words=%d" %(sameString.__len__()))
    return dist
    
    
if __name__== "__main__":
    """
    Main function
    """
    barack=open('barack.txt','r').read()
    trump=open('trump.txt','r').read()
    returnDistance= calEditDistance(barack,trump)
    print("Full Paragraph Differece= {}".format(returnDistance))
    shortDistance=shortCorpus(barack,trump)
    print("Substring Differece= {}".format(shortDistance))
    
    