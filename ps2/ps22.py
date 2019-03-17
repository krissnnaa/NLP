#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 11:31:19 2019

@author: krishna
"""

from stanfordcorenlp import StanfordCoreNLP
import re
import spacy
import gensim


def textProcessing(text):
    stringLower = ''
    smoothPart = re.sub(r'[^\w\s]', '', text)
    smoothTrunc = smoothPart.lower()
    stringLower = stringLower + ' ' + smoothTrunc
    return stringLower


class StanfordNLP:
    def __init__(self, host='http://localhost', port=9000):
        self.nlp = StanfordCoreNLP(host, port=port,
                                   timeout=30000)
        self.props = {
            'annotators': 'tokenize,ssplit,pos,lemma,ner,parse,depparse,dcoref,relation'}

    def word_tokenize(self, sentence):
        return self.nlp.word_tokenize(sentence)

    def pos(self, sentence):
        return self.nlp.pos_tag(sentence)
        self.nlp.le

    def ner(self, sentence):
        return self.nlp.ner(sentence)

    def parse(self, sentence):
        return self.nlp.parse(sentence)


def spacyFunction(text):
    nlp = spacy.load('en')
    tokenList = []
    tokenLemaa = []
    tokenPos = []
    tokenNer = []
    setenceSplit = []

    textFile = nlp(text)
    for token in textFile:
        tokenList.append(token.text)
        tokenLemaa.append(token.lemma_)
        tokenPos.append(tuple((token.text, token.pos_)))
    for token in textFile.ents:
        tokenNer.append(tuple((token.text, token.label_)))

    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    for token in textFile.sents:
        setenceSplit.append(token.text)

    print(tokenLemaa)
    print(tokenPos)
    print(tokenNer)
    print(setenceSplit)


def gensimOperation(text):
    tokenList = []
    tokenLemma = []
    tokenPos = []
    tokenNer = []

    tokenList = list(gensim.utils.tokenize(text))
    tokenLemma = list(gensim.utils.lemmatize(text))
    tokenPos = list(gensim.utils.lemmatize(text))
    print(tokenList)
    print(tokenLemma)


if __name__ == '__main__':
    option = input("\n Enter s for stanford core nlp \n Enter p for SpaCy and \n Enter g for gensim\n ")
    textFile = open('/home/krishna/Downloads/NLP/ps2/mysentences.txt', 'r').read()
    textFile = textProcessing(textFile)
    if option == 's':
        sNLP = StanfordNLP()
        print("POS:", sNLP.pos(textFile))
        print("Tokens:", sNLP.word_tokenize(textFile))
        print("NER:", sNLP.ner(textFile))
    if option == 'p':
        spacyFunction(textFile)

    if option == 'g':
        gensimOperation(textFile)

