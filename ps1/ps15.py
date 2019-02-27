from nltk.corpus import words
import re

indicator =0
def maxMatch(sentence,dictionary,result):
    global indicator
    if sentence is None:
        return None
    else:
        for i in range(len(sentence),1,-1):
            firstword=sentence[:i]
            remainder=sentence[i:]
            for w in dictionary:
                if firstword==w:
                    indicator=1
                    result.append(firstword)
                    return maxMatch(remainder,dictionary,result)

        if indicator==0:
            firstword=sentence[0]
            remainder=sentence[1:]
            result.append(firstword)
            return maxMatch(remainder, dictionary, result)


def soundexAlgo(name):
    dictLetter={'1': ['b','f','p','v'],
                '2': ['c', 'g', 'k', 'q', 's', 'x', 'z'],
                '3': ['d', 't'],
                '4': ['l'],
                '5': ['m', 'n'],
                '6': ['r']
                }
    vowel=['a','e','i','o','u','y','h','w']
    noVowelName=[]

    if len(name)==1:
        return name[0].upper()+'000'
    elif len(name)==2:
        indicator = 0

        for key, value in dictLetter.items():
            for v in value:
                if name[1] == v:
                    indicator = 1
                    break
        if indicator == 0:
                key = '0'

        return name[0].upper()+key+'00'
    else:
    #step 3
        #which category of dictionary each letter belongs to
        matchLen =[]
        for char in name:
            indicator=0
            for key,value in dictLetter.items():
                for v in value:
                    if char==v:
                        matchLen.append(key)
                        indicator=1
                        break
            if indicator==0:
                if char=='h' or char =='w':
                    key='9'
                else:
                    key='0'
                matchLen.append(key)



        # same character number includes h and w character's inbetween
        indMatch = []

        for indx in range(1, len(matchLen) - 2):

            if matchLen[indx] == '0' or matchLen[indx] == '9':
                continue

            else:

                if matchLen[indx] == matchLen[indx + 2] and matchLen[indx] =='9':
                    indMatch.append(indx + 2)

        matchLen = [v for i, v in enumerate(matchLen) if i not in frozenset(indMatch)]



        #same adjacent character's position calculation
        indMatch = []

        for indx in range (0,len (matchLen)-2):

            if matchLen[indx]=='0' or matchLen[indx]=='9':
                continue

            else:

                if matchLen[indx]==matchLen[indx+1]:
                    indMatch.append(indx+1)

        matchLen = [v for i, v in enumerate(matchLen) if i not in frozenset(indMatch)]

        finalVal=[]
        appendList=[]
        for val in matchLen:
            if val!='0' and val !='9':
                finalVal.append(val)

        if name[0] not in vowel:
            finalVal = finalVal[1:]
        if finalVal.__len__() >=3:
            appendList=finalVal[0:3]
            addList= ''.join(appendList)
            soundexWord=name[0].upper()+addList
        else:
            appendList=finalVal
            while len(appendList)<3:
                appendList.append('0')
            addList = ''.join(appendList)

            soundexWord=name[0].upper()+addList

        return soundexWord


if __name__== '__main__':
    # wordlist=words.words()
    # sentence = open('barack.txt', 'r').read()
    # appStart = 'that you work'
    # appEnd = 'respect'
    # barK = re.search(r'(?<= that you work)(.*)(?=respect)', sentence).group(1)
    # sentence = appStart + barK + appEnd
    # sentence= re.sub(r'[^\w]', ' ', sentence)
    # sentence=sentence.replace(" ","")
    # result=[]
    # maxMatch(sentence,wordlist,result)
    # strResult = ''.join(str(w +' ') for w in result)
    # print(strResult)
    amePresident=['ab','kamala','hillary']
    preSoundex=[]
    for item in amePresident:
        temp=soundexAlgo(item)
        preSoundex.append(temp)

    print(preSoundex)