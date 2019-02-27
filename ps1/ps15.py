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
        string='000'
        return name[0]+string

    for ch in name:
        indi = 0
        for vow in vowel:
            if ch==vow:
                indi =1
                break
        if indi==0:
            noVowelName.append(ch)
    noConso=[]
    noConso.append(noVowelName[0])
    for letter in noVowelName[1:]:
        for key,value in dictLetter.items():
            for v in value:
                if letter==v:
                    noConso.append(key)
                    break

    print(noConso)


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
    amePresident=['donald','kamala','hillary']
    for item in amePresident:
        soundexAlgo(item)