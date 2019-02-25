import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.collocations import *

def readingCsvFile():

    df=pd.read_csv('~/Desktop/NLP/ps1/amazon-fine-food-reviews/Reviews.csv',encoding='utf-8')
    textColumn=[]
    df=df[:10000]
    for row in df.itertuples():
        textColumn.append(row.Text)

    textFinal=''.join(textColumn)

    textFinal=BeautifulSoup(textFinal,'html.parser')
    textFinal=textFinal.get_text()
    return textFinal

def wordTokenization(finalText):

    tokenText=nltk.word_tokenize(finalText)
    print(tokenText.__len__())
    freqWords = nltk.FreqDist(tokenText)
    # for w , k in freqWords.items():
    #     print(w,k)
    #     if k=='the':
    #         break
    print(freqWords.most_common(10))
    #print(freqWords.hapaxes())

    return tokenText

def collocationCalculation(tokenWords):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    word_fd=nltk.FreqDist(tokenWords)
    bigram_fd= nltk.FreqDist(nltk.bigrams(tokenWords))
    finder=nltk.BigramCollocationFinder(word_fd,bigram_fd,window_size=4)

    collText=nltk.collocations(tokenWords)
    print(collText)



if __name__=='__main__':

    finalText = readingCsvFile()
    tokenWords=wordTokenization(finalText)
    collocationCalculation(tokenWords)
