import nltk
import sys
from matplotlib import pyplot as pt
from bs4 import BeautifulSoup
from urllib import request
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
import statistics
from nltk.corpus import words

def removeTag(raw):
    """
    Remove tag from the text
    :param raw: raw
    :return: refined raw text
    """
    rawRefine = BeautifulSoup(raw,'html.parser')
    rawText = rawRefine.get_text()
    #rawText= re.sub(r'[^\w]', ' ', rawText)
    return rawText

def wordTokenize(raw):
    tokenWords= word_tokenize(raw)
    tokenWords= [w.lower() for w in tokenWords]
    # print(tokenWords)
    print('Size of the Tokens=%d'%tokenWords.__len__())
    return tokenWords

def frequencyDistribution(tokenWords):
    freqWords= FreqDist(tokenWords)
    print(freqWords.most_common(10))
    print(freqWords.hapaxes())
    pt.plot()
    freqWords.plot(30, cumulative=True)

def longestWordforms(tokenWords):
    typeWords=list(set(tokenWords))
    wordForms=nltk.pos_tag(tokenWords)
    # print(wordForms)
    lenType=len(typeWords)
    print('Length of Types=%d' %lenType)
    longList=[]
    for i in range(10):
        temp=max(typeWords, key=len)
        longList.append(temp)
        typeWords.remove(temp)

    print(longList)
    return lenType

def uniqueWord(tokenWords,typeLen):
    wordCount=Counter(tokenWords)
    uniqWords=[word for word, count in wordCount.items() if count == 1]
    print('Percentage of unique words =%f'%(((typeLen-uniqWords.__len__())/typeLen)*100))

def meanMedainStd(tokenWords):
    lenWords=[len(w) for w in tokenWords]
    mean= statistics.mean(lenWords)
    median=statistics.median(lenWords)
    stdDev= statistics.stdev(lenWords)
    print( 'Mean =%f Median=%f Standard Deviation=%f'%(mean,median,stdDev))

def unknownWords(tokenWords):
    wordList=list(w.lower() for w in words.words())
    unkList=[]
    for w in tokenWords:
        indicator=0
        for word in wordList:
            if w==word:
                indicator=1
                break
        if indicator==0:
            unkList.append(w)

    print(unkList)



if __name__=='__main__':

    url = sys.argv[1] # 'http://www.gutenberg.org/files/25990/25990-h/25990-h.htm'
    response = request.urlopen(url)
    raw = response.read().decode('utf8')
    rawReturn = removeTag(raw)
    tokenWords = wordTokenize(rawReturn)
    frequencyDistribution(tokenWords)
    typeLen=longestWordforms(tokenWords)
    uniqueWord(tokenWords,typeLen)
    meanMedainStd(tokenWords)
    unknownWords(tokenWords)






