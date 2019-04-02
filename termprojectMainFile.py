import glob2
import json
import nltk
from sklearn.linear_model import LogisticRegression

def creatingFeatures():

    flag=0
    if flag==0:
        # Creating a list of path for each dataset
        sourcePath = glob2.glob('/home/krishna/Downloads/termProject/dataset/rumoureval-data/**/**/source-tweet/*.json')
        sourcePath=list(set(sourcePath))
        with open('filePath.txt','w') as fd:
            for item in sourcePath:
                fd.write('%s\n'%item)

    with open ('filePath.txt','r') as fd:
        sourcePath = [l[:-1] for l in fd.readlines()]
    featureList=[]
    sourceTweet={}
    for file in sourcePath:
        innerFeature=[]
        innerSource={}
        with open(file,encoding='utf-8') as fd:
            jsonFile=json.load(fd)
            tweetPost=jsonFile['text']
            tweetID=jsonFile['id']
            tokenWords=nltk.tokenize.word_tokenize(tweetPost)
            tokenCount=len(tokenWords)
            upperCount=0
            for words in tokenWords:
                if words[0].isupper():
                    upperCount=upperCount+1
            perUpper=upperCount/tokenCount
            follower=jsonFile['user']['followers_count']
            innerFeature.append(follower)
            innerFeature.append(perUpper)
            retweet=jsonFile['retweet_count']
            innerFeature.append(retweet)
            if tokenWords[-1]=='?':
                questionMark=1
            else:
                questionMark=0
            innerFeature.append(questionMark)
            favorite=jsonFile['favorite_count']
            innerFeature.append(favorite)
            innerSource[tweetID]=innerFeature
            sourceTweet.update(innerSource)
        featureList.append(innerFeature)
    with open('tweetSourceFile.json','w',encoding='utf-8') as fd:
        json.dump(sourceTweet,fd)
    return  sourceTweet


def logisticRegressionClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]

    clf=LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial').fit(x_train, y_train)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)

    print('Logistic accuracy score for dev set=%0.2f'%accuracyScore)




if __name__=='__main__':

    #Preprocss and create features
    creatingFeatures()

    with open('tweetSourceFile.json','r',encoding='utf-8') as fd:
        sourceTweet=json.load(fd)

    with open('/home/krishna/Downloads/termProject/dataset/traindev/rumoureval-subtaskB-train.json','r') as infile:
        sourceLable=json.load(infile)
    with open('/home/krishna/Downloads/termProject/dataset/traindev/rumoureval-subtaskB-dev.json','r') as infile:
        devLable=json.load(infile)

    trainDict={}
    devDict={}
    for key,value in sourceTweet.items():
        xandy=[]
        for k,v in sourceLable.items():
            if key==k:
                if v=='true':
                    lable=1
                elif v=='false':
                    lable=0
                else:
                    lable=2

                xandy.append(value)
                xandy.append(lable)
                trainDict[key]=xandy
        for k, v in devLable.items():
            if key == k:
                if v == 'true':
                    lable = 1
                elif v == 'false':
                    lable = 0
                else:
                    lable = 2

                xandy.append(value)
                xandy.append(lable)
                devDict[key] = xandy

    x_train=[l[0] for l in trainDict.values()]
    y_train=[l[1] for l in trainDict.values()]
    x_dev = [l[0] for l in devDict.values()]
    y_dev = [l[1] for l in devDict.values()]
    X_data=(x_train,x_dev)
    y_data=(y_train,y_dev)

    logisticRegressionClassifier(X_data,y_data)

    print()
