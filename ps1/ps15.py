from nltk.corpus import words

indicator =0
def maxMatch(sentence,dictionary):
    global indicator
    if sentence is None:
        return None
    else:
        for i in range(len(sentence),1,-1):
            firstword=sentence[:i]
            remainder=list(set(sentence)-set(firstword))
            for w in dictionary:
                if firstword==w:
                    indicator=1
                    return (firstword,maxMatch(remainder,dictionary))
        if indicator==0:
            finalList=[]
            for i in range(0,sentence.__len__()):
                for w in dictionary:
                    if firstword == w:
                        finalList.append(firstword)
            return finalList

if __name__== '__main__':
    wordlist=words.words()
    sentence = open('barack.txt', 'r').read()
    sentence=sentence.replace(" ","")
    result= list(maxMatch(sentence,wordlist))
    print(result)
