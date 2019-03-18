from gensim.models import Word2Vec
import nltk
firstExample='breakfast cereal dinner lunch'
secondExample='man women king kingdom'
thirdExample='coffee milk tea popcorn'
firToken=nltk.tokenize.word_tokenize(firstExample)
secondToken=nltk.tokenize.word_tokenize(secondExample)
thirdToken=nltk.tokenize.word_tokenize(thirdExample)
finalList=[]
finalList.append(firToken)
finalList.append(secondToken)
finalList.append(thirdToken)

# Training the word2vec model
model=Word2Vec(finalList,min_count=1)
# Vocabulary in the sentences
vocabWords=list(model.wv.vocab)
print(vocabWords)
# Doesnt_match method
doesnotMatch=model.wv.doesnt_match(firstExample.split())
print(doesnotMatch)
doesnotMatch=model.wv.doesnt_match(secondExample.split())
print(doesnotMatch)
doesnotMatch=model.wv.doesnt_match(thirdExample.split())
print(doesnotMatch)