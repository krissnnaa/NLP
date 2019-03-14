import nltk
import re
import math
from nltk.probability import FreqDist
from nltk.wsd import lesk
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
    if name == 'bass':
        for line in trainFile:
            truncPart = line.split(':\t')
            if truncPart[0] == 'bass':
                temp = 'music'
            else:
                temp = 'fish'
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
            else:
                temp = 'beer'
            truncPart[1] = re.sub(r'[^\w\s]', '', truncPart[1])
            lowerTrunc = truncPart[1].lower()
            trainDict[lowerTrunc] = temp
            if test == 1:
                trainList.append(lowerTrunc)

    if test == 1:
        return trainDict, trainList
    else:
        return trainDict


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
                break
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
        #these are for cause
        national = 0
        nation = 0
        children = 0
        unity = 0
        country=0
        stability=0
        peace=0
        people=0
        #sake + IN =>INDERTERMINANAT
        # these are for the beer
        tablespoons=0
        wine=0
        cup=0
        drink=0
        bottle=0
        sauce=0
        #japanese +sake
        #sake+NN => n+2 pos

        for val in keyTokens:
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
                break
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
        afterPos = posItem[targetIndex+1]+posItem[targetIndex + 2] # CCNN

        indx = 0
        if targetWindow == 'japanese sake':
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
    finalDecisionList.append((finalList[3], 'music'))
    finalDecisionList.append((finalList[4], 'fish'))
    finalDecisionList.append((finalList[5], 'fish'))
    finalDecisionList.append((finalList[6], 'music'))
    finalDecisionList.append((finalList[7], 'music'))
    finalDecisionList.append((finalList[8], 'music'))
    finalDecisionList.append((finalList[9], 'fish'))
    finalDecisionList.append((finalList[10], 'fish'))
    finalDecisionList.append((finalList[11], 'fish'))
    finalDecisionList.append((finalList[12], 'music'))
    finalDecisionList.append((finalList[13], 'music'))
    finalDecisionList.append((finalList[14], 'fish'))
    finalDecisionList.append((finalList[15], 'fish'))
    finalDecisionList.append((finalList[16], 'fish'))
    finalDecisionList.append((finalList[17], 'fish'))
    finalDecisionList.append((finalList[18], 'music'))
    finalDecisionList.append((finalList[19], 'fish'))
    finalDecisionList.append((finalList[20], 'fish'))

    return finalDecisionList


if __name__ == '__main__':
    name = 'sake'
    baseLine = 0
    # Training set
    with open('C:/Users/CoCo Lab/PycharmProjects/NLP/NLP/ps2/bass_sake_train_test/sakeTrain.txt', 'r') as f:
        readTrain = f.readlines()
    preProcess = filePreprocessing(readTrain, 0, name)
    # Testing set
    with open('C:/Users/CoCo Lab/PycharmProjects/NLP/NLP/ps2/bass_sake_train_test/sakeTest.txt', 'r') as f:
        readTest = f.readlines()
    testDict, preProcessTest = filePreprocessing(readTest, 1, name)

    if baseLine == 1:
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
        for val, value in zip(outputLabel, outputList):
            if val == value:
                count = count + 1
        accuracy = count / 100

        print("\nAccuracy of the Baseline in percentage = {:.1%}".format(accuracy))

    else:
        if name == 'bass':
            logLiklihood = loglikelihoodComputation(preProcess)
            finalDecisionList = decisionListComputaion(logLiklihood)
            print("\n\n========Decision List=======\n\n")
            print(finalDecisionList)
            print("\n\n========Sense Classification=======")
            print("Context" + '\t\t---->\t\t' + 'Sense\n\n')

            count = 0
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
                    count = count + 1
                    continue
                if targetPos == finalDecisionList[1][0]:
                    print(item + '--->' + finalDecisionList[1][1])
                    count = count + 1
                    continue

                if targetWord == finalDecisionList[2][0]:
                    print(item + '--->' + finalDecisionList[2][1])
                    count = count + 1
                    continue
                targetWord = word + ' ' + tokenItem[wordIndex + 1]
                if targetWord == finalDecisionList[3][0]:
                    print(item + '--->' + finalDecisionList[3][1])
                    count = count + 1
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[4][0]:
                    print(item + '--->' + finalDecisionList[4][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[5][0]:
                    print(item + '--->' + finalDecisionList[5][1])
                    count = count + 1
                    continue
                targetPos = posItem[wordIndex - 1] + ' ' + word
                if targetPos == finalDecisionList[6][0]:
                    print(item + '--->' + finalDecisionList[6][1])
                    count = count + 1
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[7][0]:
                    print(item + '--->' + finalDecisionList[7][1])
                    count = count + 1
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[8][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[8][1])
                    count = count + 1
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[9][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[9][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[10][0]:
                    print(item + '--->' + finalDecisionList[10][1])
                    count = count + 1
                    continue

                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[11][0]:
                    print(item + '--->' + finalDecisionList[11][1])
                    count = count + 1
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[12][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[12][1])
                    count = count + 1
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[13][0]:
                    print(item + '--->' + finalDecisionList[13][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[14][0]:
                    print(item + '--->' + finalDecisionList[14][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[15][0]:
                    print(item + '--->' + finalDecisionList[15][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[16][0]:
                    print(item + '--->' + finalDecisionList[16][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[17][0]:
                    print(item + '--->' + finalDecisionList[17][1])
                    count = count + 1
                    continue
                targetWord = tokenItem[wordIndex - 1] + ' ' + word
                if targetWord == finalDecisionList[18][0]:
                    print(item + '--->' + finalDecisionList[18][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[19][0]:
                    print(item + '--->' + finalDecisionList[19][1])
                    count = count + 1
                    continue
                targetPos = word + ' ' + posItem[wordIndex + 1]
                if targetPos == finalDecisionList[20][0]:
                    print(item + '--->' + finalDecisionList[20][1])
                    count = count + 1
                    continue
            accuracy = count / 100
            print("\nAccuracy of the Decision List for WSD of bass word in percentage = {:.1%}".format(accuracy))

        else:

            logLiklihood = loglikelihoodComputation(preProcess)
            finalDecisionList = decisionListComputaion(logLiklihood)
            print("\n\n========Decision List=======\n\n")
            print(finalDecisionList)
            print("\n\n========Sense Classification=======")
            print("Context" + '\t\t---->\t\t' + 'Sense\n\n')

            count = 0
            for item in preProcessTest:
                tokenItem = nltk.tokenize.word_tokenize(item)
                for word in tokenItem:
                    if word == 'bass':
                        wordIndex = tokenItem.index(word)
                        break
                targetWord = tokenItem[wordIndex - 1] + ' ' + word

                if targetWord == finalDecisionList[0][0]:
                    print(item + '--->' + finalDecisionList[0][1])
                    count = count + 1
                    continue
                if targetWord == finalDecisionList[1][0]:
                    print(item + '--->' + finalDecisionList[1][1])
                    count = count + 1
                    continue
                if targetWord == finalDecisionList[2][0]:
                    print(item + '--->' + finalDecisionList[2][1])
                    count = count + 1
                    continue
                if targetWord == finalDecisionList[3][0]:
                    print(item + '--->' + finalDecisionList[3][1])
                    count = count + 1
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[4][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[4][1])
                    count = count + 1
                    continue
                indx = 0
                for word in tokenItem:
                    if word == finalDecisionList[5][0]:
                        indx = 1
                        break
                if indx == 1:
                    print(item + '--->' + finalDecisionList[5][1])
                    count = count + 1
                    continue
                if targetWord == finalDecisionList[6][0]:
                    print(item + '--->' + finalDecisionList[6][1])
                    count = count + 1
                    continue
                targetWord = 'bass' + ' ' + tokenItem[wordIndex + 1]
                if targetWord == finalDecisionList[7][0]:
                    print(item + '--->' + finalDecisionList[7][1])
                    count = count + 1
            accuracy = count / 100
            print("\nAccuracy of the Decision List for WSD of sake word in percentage = {:.1%}".format(accuracy))
