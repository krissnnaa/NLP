import glob2
import json
import nltk
from sklearn.linear_model import LogisticRegressionCV
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


def creatingFeatures(testFile):

    if testFile is not None:
        sourcePath = glob2.glob('/home/krishna/Downloads/termProject/test_data/**/source-tweet/*.json')
        sourcePath = list(set(sourcePath))
        with open('testFilePath.txt','w') as fd:
            for item in sourcePath:
                fd.write('%s\n'%item)
        with open('testfilePath.txt', 'r') as fd:
            sourcePath=[l[:-1] for l in fd.readlines()]

        testfeatureList = []
        testsourceTweet = {}
        for file in sourcePath:
            innerFeature = []
            innerSource = {}
            with open(file, 'r') as fd:
                jsonFile = json.load(fd)
                tweetPost = jsonFile['text']
                tweetID = jsonFile['id']
                tokenWords = nltk.tokenize.word_tokenize(tweetPost)
                tokenCount = len(tokenWords)
                upperCount = 0
                hashtag=0
                url=0
                for words in tokenWords:
                    if words[0].isupper():
                        upperCount = upperCount + 1
                    if words=='#':
                        hashtag=1
                    if words.startswith('http'):
                        url=1
                innerFeature.append(hashtag)
                innerFeature.append(url)
                perUpper = upperCount / tokenCount
                innerFeature.append(perUpper)
                retweet = jsonFile['retweet_count']/1000
                innerFeature.append(retweet)
                if tokenWords[-1] == '?':
                    questionMark = 1
                else:
                    questionMark = 0
                innerFeature.append(questionMark)
                favorite = jsonFile['favorite_count']/1000
                innerFeature.append(favorite)
                innerSource[tweetID] = innerFeature
                testsourceTweet.update(innerSource)
            testfeatureList.append(innerFeature)
        return testsourceTweet
    else:
        sourcePath = glob2.glob('/home/krishna/Downloads/termProject/dataset/rumoureval-data/**/**/source-tweet/*.json')
        sourcePath = list(set(sourcePath))
        with open('trainFilePath.txt', 'w') as fd:
            for item in sourcePath:
                fd.write('%s\n' %item)
        with open('trainFilePath.txt', 'r') as fd:
            sourcePath = [l[:-1] for l in fd.readlines()]
        featureList=[]
        sourceTweet={}
        for file in sourcePath:
            innerFeature=[]
            innerSource={}
            with open(file,'r') as fd:
                jsonFile=json.load(fd)
                tweetPost=jsonFile['text']
                tweetID=jsonFile['id']
                tokenWords=nltk.tokenize.word_tokenize(tweetPost)
                tokenCount=len(tokenWords)
                upperCount=0
                hashtag=0
                url=0
                for words in tokenWords:
                    if words[0].isupper():
                        upperCount=upperCount+1
                    if words=='#':
                        hashtag=1
                    if words.startswith('http'):
                        url=1
                innerFeature.append(hashtag)
                innerFeature.append(url)
                perUpper=upperCount/tokenCount
                innerFeature.append(perUpper)
                retweet=jsonFile['retweet_count']/1000
                innerFeature.append(retweet)
                if tokenWords[-1]=='?':
                    questionMark=1
                else:
                    questionMark=0
                innerFeature.append(questionMark)
                favorite=jsonFile['favorite_count']/1000
                innerFeature.append(favorite)
                innerSource[tweetID]=innerFeature
                sourceTweet.update(innerSource)
            featureList.append(innerFeature)
        return sourceTweet


def logisticRegressionClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    print(sorted(Counter(y_train).items()))
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    print(sorted(Counter(y_resampled).items()))
    clf=LogisticRegressionCV(cv=5,random_state=0, solver='lbfgs', multi_class='multinomial').fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Logistic accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Logistic accuracy score for test set=%0.2f' % accuracyScore)

def naiveBayesClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=MultinomialNB().fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Naive Bayes accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Naive Bayes accuracy score for test set=%0.2f' % accuracyScore)

def LinearSVMClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=LinearSVC(random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Linear SVC accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Linear SVC accuracy score for test set=%0.2f' % accuracyScore)

def ensembleClassifier(X,y,x_test,y_test):

    x_train=X[0]
    x_dev=X[1]
    y_train=y[0]
    y_dev=y[1]
    ros = RandomOverSampler(random_state=0)
    X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    clf=RandomForestClassifier(n_estimators=10,random_state=0).fit(X_resampled, y_resampled)
    clf.predict(x_dev)
    accuracyScore=clf.score(x_dev,y_dev)
    print('Ensemble Random Forest accuracy score for dev set=%0.2f'% accuracyScore)

    clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)

if __name__=='__main__':

    #Preprocss and create features
    sourceTweet=creatingFeatures(None)
    testsourceTweet=creatingFeatures('testfile')
    with open('/home/krishna/Downloads/termProject/dataset/traindev/rumoureval-subtaskB-train.json','r') as infile:
        sourceLable=json.load(infile)
    with open('/home/krishna/Downloads/termProject/dataset/traindev/rumoureval-subtaskB-dev.json','r') as infile:
        devLable=json.load(infile)
    with open('/home/krishna/Downloads/termProject/test_data/test_label.json', 'r') as infile:
        testLabel = json.load(infile)

    devDict={}
    trainDict={}
    for key,value in sourceTweet.items():
        xandy=[]
        for k,v in sourceLable.items():
            if str(key)==k:
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
            if str(key) == k:
                if v == 'true':
                    lable = 1
                elif v == 'false':
                    lable = 0
                else:
                    lable = 2

                xandy.append(value)
                xandy.append(lable)
                devDict[key] = xandy

    testDict = {}
    for key, value in testsourceTweet.items():
        xandy = []
        for k, v in testLabel.items():
            if str(key) == k:
                if v == 'true':
                    lable = 1
                elif v == 'false':
                    lable = 0
                else:
                    lable = 2

                xandy.append(value)
                xandy.append(lable)
                testDict[key] = xandy

    x_train= [l[0] for l in trainDict.values()]
    y_train= [l[1] for l in trainDict.values()]
    x_dev = [l[0] for l in devDict.values()]
    y_dev = [l[1] for l in devDict.values()]
    x_test = [l[0] for l in testDict.values()]
    y_test = [l[1] for l in testDict.values()]
    X_data= (x_train,x_dev)
    y_data= (y_train,y_dev)

    #Normalizing with min_max scalar
    min_max_scaler = preprocessing.MinMaxScaler()
    x_train=min_max_scaler.fit_transform(x_train)
    x_dev=min_max_scaler.fit_transform(x_dev)
    x_test=min_max_scaler.fit_transform(x_test)

    #Logistic Regression
    logisticRegressionClassifier(X_data,y_data,x_test,y_test)
    #Naive Bayse
    naiveBayesClassifier(X_data, y_data, x_test, y_test)
    # Linear SVC
    LinearSVMClassifier(X_data, y_data, x_test, y_test)
    #Ensemble Random forest
    ensembleClassifier(X_data, y_data, x_test, y_test)

