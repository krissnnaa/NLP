import re
import nltk
import pandas as pd
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import sys

def readingCsvFile(csvPath):

    df=pd.read_csv(csvPath,encoding='utf-8')
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
    freqWords = nltk.FreqDist(tokenText)
    print('Unigram Words= %d'%freqWords.N())
    print('Length of unique words=%d'%len(freqWords.hapaxes()))
    return tokenText

def collocationCalculation(tokenWords):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    word_fd=nltk.FreqDist(tokenWords)
    bigram_fd= nltk.FreqDist(nltk.bigrams(tokenWords))
    adjacent=nltk.BigramCollocationFinder(word_fd,bigram_fd,window_size=4)
    print('Count of 2-word adjacents and non-adjacents = %d'%len(adjacent.score_ngrams(bigram_measures.raw_freq)))


def stopWordsToFilter(tokenWords):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    stoplist = stopwords.words('english')
    finder = nltk.BigramCollocationFinder.from_words(tokenWords)
    print('Length before filtering=%d'%len(finder.score_ngrams(bigram_measures.raw_freq)))
    finder.apply_word_filter(lambda w: w in stoplist)
    print('Length after filtering=%d' % len(finder.score_ngrams(bigram_measures.raw_freq)))

def pointWiseMutualInformation(tokenWords):
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    finder = nltk.BigramCollocationFinder.from_words(tokenWords)
    collList=[]
    for find in finder.score_ngrams(bigram_measures.pmi):

        if find[1]>=2:
            collList.append(find)

    print('Five good collocations=')
    print(collList[:5])

    print('Two not so good collocations=')
    print(collList[-2:])



if __name__=='__main__':
    csvPath = sys.argv[1]  # '~/NLP/ps1/Reviews.csv'
    finalText = readingCsvFile()
    tokenWords=wordTokenization(finalText)
    collocationCalculation(tokenWords)
    stopWordsToFilter(tokenWords)
    pointWiseMutualInformation(tokenWords)
