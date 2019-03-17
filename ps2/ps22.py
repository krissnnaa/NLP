# import corenlp
#
# inputText=open('mysentences.txt','r')
# with corenlp.CoreNLPClient(annotators="tokenize ssplit pos lemma ner depparse".split()) as client:
#  ann = client.annotate(inputText)
# sentence = ann.sentence[0]
# tokenWords= [token.word for token in sentence.token]
# print(tokenWords)
#
#
# #import subprocess
# #subprocess.call(['java', '-mx4g,' '-cp', "*" ,'edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators', "tokenize,ssplit,pos,lemma,parse,sentiment" ,'-port 9000', '-timeout 30000'])
# # # Simple usage
# # from stanfordcorenlp import StanfordCoreNLP
# #
# # nlp = StanfordCoreNLP(r'C:/Users/CoCo Lab/Downloads/stanford-corenlp-full-2018-10-05')
# #
# # sentence = 'Guangdong University of Foreign Studies is located in Guangzhou.'
# # print ('Tokenize:', nlp.word_tokenize(sentence))
# # print ('Part of Speech:', nlp.pos_tag(sentence))
# # print ('Named Entities:', nlp.ner(sentence))
# # print ('Constituency Parsing:', nlp.parse(sentence))
# # print ('Dependency Parsing:', nlp.dependency_parse(sentence))
# #
# # nlp.close()

import