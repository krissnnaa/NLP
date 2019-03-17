import nltk
import re
import math
from nltk.probability import FreqDist
from nltk.wsd import lesk
import sys
import operator


# one word collocation distribution count
# if fish occurred in +-k word context
# if play bass
# if bass player
# if river  occurred +-k words
# sea bass
# Guitar in +-k words
# on bass
# striped bass
# electric bass
# stringApp =stringApp + ' '+ lowerTrunc
# words=nltk.tokenize.word_tokenize(stringApp)
# fdist=FreqDist(words)
# print(stringApp)
# print(fdist.most_common(50))

def filePreprocessing(readTrain, test, name):
    trainFile = [x.strip() for x in readTrain]
    trainDict = {}
    trainList = []
    scoreList=[]
    if name == 'bass':
        for line in trainFile:
            truncPart = line.split(':\t')
            if truncPart[0] == 'bass':
                temp = 'music'
                scoreList.append('music')
            else:
                temp = 'fish'
                scoreList.append('fish')
            truncPart[1] = re.sub(r'[^\w\s]', '', truncPart[1])
            lowerTrunc = truncPart[1].lower()
            trainDict[lowerTrunc] = temp
            if test == 1:
                trainList.append(lowerTrunc)

    else:
        for line in trainFile:
            truncPart = line.split(':\t')
            if truncPart[0] == 'sake':
                temp = 'cause'
                scoreList.append('cause')
            else:
                temp = 'beer'
                scoreList.append('beer')
            truncPart[1] = re.sub(r'[^\w\s]', '', truncPart[1])
            lowerTrunc = truncPart[1].lower()
            trainDict[lowerTrunc] = temp
            if test == 1:
                trainList.append(lowerTrunc)

    if test == 1:
        return trainDict, trainList,scoreList
    else:
        return trainDict,scoreList


def loglikelihoodComputation(trainDict):
    logLikeli = {}
    for key, value in trainDict.items():
        keyTokens = nltk.word_tokenize(key)
        riv = 0
        fish = 0
        guitar = 0
        piano = 0
        for val in keyTokens:
            if val == 'fish':
                fish = 1
                break
            if val == 'river':
                riv = 1
                break
            if val == 'guitar':
                guitar = 1
                break
            if val == 'piano':
                piano = 1
                break
            if val == 'bass':
                targetIndex = keyTokens.index(val)

        # fish word
        if fish == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        # river word
        if riv == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        # guitar word
        if guitar == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        if piano == 1:
            # music
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # previous and after one words and pos
        targetWindow = keyTokens[targetIndex - 1] + ' ' + 'bass'
        targetWind = 'bass' + ' ' + keyTokens[targetIndex + 1]
        posItem = nltk.pos_tag(keyTokens)
        posItem = [x[1] for x in posItem]
        beforePos = posItem[targetIndex - 1]
        afterPos = posItem[targetIndex + 1]
        indx = 0
        if targetWindow == 'play bass':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1
        elif targetWindow == 'sea bass':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1
        elif targetWindow == 'on bass':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1
        elif targetWindow == 'striped bass':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1
        elif targetWindow == 'electric bass':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1
        elif targetWind == 'bass player':
            for k, v in logLikeli.items():
                if k == targetWind:
                    indx = 1

                    logLikeli[targetWind] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWind] = 1

        elif beforePos == 'NN' or beforePos == 'DT' or beforePos == 'CC':
            # Music
            targetWord = beforePos + ' ' + 'bass'
            for k, v in logLikeli.items():
                if k == targetWord:
                    indx = 1
                    logLikeli[targetWord] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWord] = 1

        elif afterPos == 'IN' or afterPos == 'VBG':
            # fish
            targetWord = 'bass' + ' ' + afterPos
            for k, v in logLikeli.items():
                if k == targetWord:
                    indx = 1
                    logLikeli[targetWord] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWord] = 1

        else:
            continue

    return logLikeli


def sakeLogLiklihood(trainDict):
    logLikeli = {}
    for key, value in trainDict.items():
        keyTokens = nltk.word_tokenize(key)
        # these are for cause
        national = 0
        nation = 0
        children = 0
        unity = 0
        country = 0
        stability = 0
        peace = 0
        people = 0
        # sake + IN =>INDERTERMINANAT
        # these are for the beer
        tablespoons = 0
        wine = 0
        cup = 0
        drink = 0
        bottle = 0
        sauce = 0
        # japanese +sake
        # sake+NN => n+2 pos

        for val in keyTokens:
            if val == 'national':
                national = 1
                break
            if val == 'nation':
                nation = 1
                break
            if val == 'children':
                children = 1
                break
            if val == 'unity':
                unity = 1
                break
            if val == 'country':
                country = 1
                break
            if val == 'stability':
                stability = 1
                break
            if val == 'peace':
                peace = 1
                break
            if val == 'people':
                people = 1
                break
            if val == 'wine':
                wine = 1
                break
            if val == 'cup':
                cup = 1
                break
            if val == 'tablespoons':
                tablespoons = 1
                break
            if val == 'bottle':
                bottle = 1
                break
            if val == 'drink':
                drink = 1
                break
            if val == 'sake':
                targetIndex = keyTokens.index(val)


        # national word
        if national == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # nation word
        if nation == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # country word
        if country == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # unity word
        if unity == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # stability word
        if stability == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # people word
        if people == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # children word
        if children == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # wine word
        if peace == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        # wine word
        if wine == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        # bottle word
        if bottle == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        # tablespoons word
        if tablespoons == 1:
            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1

        if drink == 1:

            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        if sauce == 1:

            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        if cup == 1:

            indx = 0
            for k, v in logLikeli.items():
                if k == val:
                    logLikeli[val] = v + 1
                    indx = 1
                    break
            if indx == 0:
                logLikeli[val] = 1
        # previous and after one words and pos
        targetWindow = keyTokens[targetIndex - 1] + ' ' + 'sake'
        posItem = nltk.pos_tag(keyTokens)
        posItem = [x[1] for x in posItem]
        afterPos = posItem[targetIndex + 1] + posItem[targetIndex + 2]  # CCNN
        afterOnePos = posItem[targetIndex + 1]
        indx = 0

        if afterOnePos == 'IN':
            # cause
            targetWord = afterOnePos
            for k, v in logLikeli.items():
                if k == targetWord:
                    indx = 1
                    logLikeli[targetWord] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWord] = 1

        elif targetWindow == 'japanese sake':
            for k, v in logLikeli.items():
                if k == targetWindow:
                    indx = 1
                    logLikeli[targetWindow] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWindow] = 1

        elif afterPos == 'CCNN':
            # beer
            targetWord = afterPos
            for k, v in logLikeli.items():
                if k == targetWord:
                    indx = 1
                    logLikeli[targetWord] = v + 1
                    break
            if indx == 0:
                logLikeli[targetWord] = 1

        else:
            continue


    return logLikeli


def decisionListComputaion(logLikeli):
    # log likelihood for decision list

    decisionList = {}
    for key, value in logLikeli.items():
        decisionList[key] = math.log10(value)

    decisionList = sorted(decisionList.items(), key=lambda kv: kv[1], reverse=True)
    finalList = [x[0] for x in decisionList]
    finalDecisionList = []
    finalDecisionList.append((finalList[0], 'music'))
    finalDecisionList.append((finalList[1], 'music'))
    finalDecisionList.append((finalList[2], 'fish'))
    finalDecisionList.append((finalList[3], 'fish'))
    finalDecisionList.append((finalList[4], 'music'))
    finalDecisionList.append((finalList[5], 'music'))
    finalDecisionList.append((finalList[6], 'fish'))
    finalDecisionList.append((finalList[7], 'music'))
    finalDecisionList.append((finalList[8], 'fish'))
    finalDecisionList.append((finalList[9], 'music'))
    finalDecisionList.append((finalList[10], 'fish'))
    finalDecisionList.append((finalList[11], 'music'))
    finalDecisionList.append((finalList[12], 'music'))
    finalDecisionList.append((finalList[13], 'fish'))
    finalDecisionList.append((finalList[14], 'music'))

    return finalDecisionList

def sakedecisionListComputaion(logLikeli):
    # log likelihood for decision list

    decisionList = {}
    for key, value in logLikeli.items():
        decisionList[key] = math.log10(value)

    decisionList = sorted(decisionList.items(), key=lambda kv: kv[1], reverse=True)
    finalList = [x[0] for x in decisionList]
    finalDecisionList = []
    finalDecisionList.append((finalList[0], 'cause'))
    finalDecisionList.append((finalList[1], 'cause'))
    finalDecisionList.append((finalList[2], 'cause'))
    finalDecisionList.append((finalList[3], 'cause'))
    finalDecisionList.append((finalList[4], 'cause'))
    finalDecisionList.append((finalList[5], 'cause'))
    finalDecisionList.append((finalList[6], 'cause'))
    finalDecisionList.append((finalList[7], 'cause'))
    finalDecisionList.append((finalList[8], 'cause'))
    finalDecisionList.append((finalList[9], 'beer'))
    finalDecisionList.append((finalList[10], 'beer'))
    finalDecisionList.append((finalList[11], 'beer'))
    finalDecisionList.append((finalList[12], 'beer'))
    finalDecisionList.append((finalList[13], 'beer'))
    finalDecisionList.append((finalList[14], 'beer'))
    finalDecisionList.append((finalList[15], 'beer'))
    return finalDecisionList



if __name__ == '__main__':
    trainFile=sys.argv[1]
    trainFileS=str(trainFile)
    testFile=sys.argv[2]

    if trainFileS.startswith('bass')==True:
        name='bass'
    else:
        name='sake'

    # name='bass'
    # trainFile='C:/Users/CoCo Lab/PycharmProjects/NLP/NLP/ps2/bassTrain.txt'
    # testFile='C:/Users/CoCo Lab/PycharmProjects/NLP/NLP/ps2/bassTest.txt'
    with open(trainFile, 'r') as f:
        readTrain = f.readlines()
    preProcess,scoreListT = filePreprocessing(readTrain, 0, name)
    # Testing set
    with open(testFile, 'r') as f:
        readTest = f.readlines()
    testDict, preProcessTest,scoreList = filePreprocessing(readTest, 1, name)

    baseLine=input("\n If you want to run baseline (Lesk algorithm for WSD)then enter 1 else enter 0 to run Decision List =")


    if baseLine == '1':
        i = 1
        outputList = []
        outputLabel = []
        if name == 'bass':
            for item in preProcessTest:
                tokenItem = nltk.tokenize.word_tokenize(item)
                leskComp = lesk(tokenItem, 'bass')
                leskName = leskComp._name
                if leskName == 'bass.n.01' or leskName == 'bass.n.02' or leskName == 'bass.n.03' or leskName == 'bass.n.06' or leskName == 'bass.n.07' or leskName == 'bass.s.01':
                    print('\n%d %s------->music ' % (i, item))
                    outputList.append('music')
                    i = i + 1
                else:
                    print('\n%d %s------->fish ' % (i, item))
                    outputList.append('fish')
                    i = i + 1

            for value in testDict.values():
                outputLabel.append(value)
        else:
            for item in preProcessTest:
                tokenItem = nltk.tokenize.word_tokenize(item)
                leskComp = lesk(tokenItem, 'sake')
                leskName = leskComp._name
                if leskName == 'sake.n.01' or leskName == 'sake.n.03':
                    print('\n%d %s------->cause ' % (i, item))
                    outputList.append('cause')
                    i = i + 1
                else:
                    print('\n%d %s------->beer ' % (i, item))
                    outputList.append('beer')
                    i = i + 1

            for value in testDict.values():
                outputLabel.append(value)

        count = 0
        ifmusic=0
        ifcause=0
        iffish=0
        ifbeer=0
        for val, value in zip(outputLabel, outputList):
            if val == value:
                count = count + 1
                if val=='music':
                    ifmusic=ifmusic+1
                if val=='cause':
                    ifcause=ifcause+1
            if value=='fish':
                iffish=iffish+1
            if value=='beer':
                ifbeer=ifbeer+1
        accuracy = count / 100

        print("\nAccuracy of the Baseline in percentage = {:.1%}".format(accuracy))
        print("\nmusic true match=%d \t cause true match=%d"%(ifmusic,ifcause))
        print("\n total fish in testfile =%d \t total beer in testfile=%d"%(iffish,ifbeer))
    else:
        if name == 'bass':
            logLiklihood = loglikelihoodComputation(preProcess)
            finalDecisionList = decisionListComputaion(logLiklihood)
            print("\n\n========Decision List=======\n\n")
            print(finalDecisionList)
            print("\n\n========Sense Classification=======")
            print("Context" + '\t\t---->\t\t' + 'Sense\n\n')

            finalScore=[]
            for item in preProcessTest:
                tokenItem = nltk.tokenize.word_tokenize(item)
                posItem = nltk.pos_tag(tokenItem)
                posItem = [x[1] for x in posItem]
                for word in tokenItem:
                    if word == 'bass':
                        wordIndex = tokenItem.index(word)
                        break
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                targetPos = posItem[wordIndex - 1] + ' ' + word
                if targetPos == finalDecisionList[0][0]:
                    print(item + '--->' + finalDecisionList[0][1])
                    finalScore.append('music')
                    continue
                if targetPos == finalDecisionList[1][0]:
                    print(item + '--->' + finalDecisionList[1][1])
                    finalScore.append('music')
                    continue

                if targetWord == finalDecisionList[2][0]:
                    print(item + '--->' + finalDecisionList[2][1])
                    finalScore.append('fish')
                    continue
                targetWord = word + ' ' + posItem[wordIndex + 1]
                if targetWord == finalDecisionList[3][0]:
                    print(item + '--->' + finalDecisionList[3][1])
                    finalScore.append('fish')
                    continue
                targetWord =  word + ' '+ tokenItem[wordIndex+1]
                if targetWord == finalDecisionList[4][0]:
                    print(item + '--->' + finalDecisionList[4][1])
                    finalScore.append('music')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[5][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[5][1])
                    finalScore.append('music')
                    continue

                targetPos = tokenItem[wordIndex - 1] + ' ' + word
                if targetPos == finalDecisionList[6][0]:
                    print(item + '--->' + finalDecisionList[6][1])
                    finalScore.append('fish')
                    continue
                targetWord = posItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[7][0]:
                    print(item + '--->' + finalDecisionList[7][1])
                    finalScore.append('music')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[8][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[8][1])
                    finalScore.append('fish')
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[9][0]:
                    print(item + '--->' + finalDecisionList[9][1])
                    finalScore.append('music')
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[10][0]:
                    print(item + '--->' + finalDecisionList[10][1])
                    finalScore.append('fish')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[11][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[11][1])
                    finalScore.append('music')
                    continue

                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[12][0]:
                    print(item + '--->' + finalDecisionList[12][1])
                    finalScore.append('music')
                    continue

                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[13][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[13][1])
                    finalScore.append('fish')
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[14][0]:
                    print(item + '--->' + finalDecisionList[14][1])
                    finalScore.append('music')
                    continue
            count=0
            for val,value in zip(finalScore,scoreList):
                if val==value:
                    count=count+1
            accuracy = count / 100
            print("\nAccuracy of the Decision List for WSD of bass word in percentage = {:.1%}".format(accuracy))

        else:

            logLiklihood = sakeLogLiklihood(preProcess)
            finalDecisionList = sakedecisionListComputaion(logLiklihood)
            print("\n\n========Decision List=======\n\n")
            print(finalDecisionList)
            print("\n\n========Sense Classification=======")
            print("Context" + '\t\t---->\t\t' + 'Sense\n\n')

            classifiedList=[]
            for item in preProcessTest:
                tokenItem = nltk.tokenize.word_tokenize(item)
                wordIndex=0
                for word in tokenItem:
                    if word == 'bass':
                        wordIndex = tokenItem.index(word)
                        break
                posTag=nltk.pos_tag(tokenItem)
                posTag=[x[1] for x in posTag]
                targetWord = posTag[wordIndex + 1]

                if targetWord == finalDecisionList[0][0]:
                    print(item + '--->' + finalDecisionList[0][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[1][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[1][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[2][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[2][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[3][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[3][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[4][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[4][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[5][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[5][1])
                    classifiedList.append('cause')
                    continue

                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[6][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[6][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[7][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[7][1])
                    classifiedList.append('cause')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[8][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[8][1])
                    classifiedList.append('cause')
                    continue

                targetWord = posTag[wordIndex + 1]+posTag[wordIndex+2]
                if targetWord == finalDecisionList[9][0]:
                    print(item + '--->' + finalDecisionList[9][1])
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[10][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[10][1])
                    classifiedList.append('beer')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[11][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[11][1])
                    classifiedList.append('beer')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[12][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[12][1])
                    classifiedList.append('beer')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[13][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[13][1])
                    classifiedList.append('beer')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[14][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[14][1])
                    classifiedList.append('beer')
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[15][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[15][1])
                    classifiedList.append('beer')
                    continue
                print(item + '--->' + 'cause')
                classifiedList.append('cause')
            countFinal=0
            for val,value in zip(classifiedList,scoreList):
                    if val==value:
                        countFinal=countFinal+1

            accuracy = countFinal / 100
            print("\nAccuracy of the Decision List for WSD of sake word in percentage = {:.1%}".format(accuracy))
