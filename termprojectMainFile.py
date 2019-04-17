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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils.multiclass import unique_labels

def creatingFeatures(testFile):
    if testFile is not None:
        # sourcePath = glob2.glob('/home/krishna/Downloads/termProject/test_data/**/source-tweet/*.json')
        # sourcePath = list(set(sourcePath))
        # with open('testFilePath.txt','w') as fd:
        #     for item in sourcePath:
        #         fd.write('%s\n'%item)
        with open('testFilePath.txt', 'r') as fd:
            sourcePath = [l[:-1] for l in fd.readlines()]
    else:
        # sourcePath = glob2.glob('/home/krishna/Downloads/termProject/dataset/rumoureval-data/**/**/source-tweet/*.json')
        # sourcePath = list(set(sourcePath))
        # with open('trainFilePath.txt', 'w') as fd:
        #     for item in sourcePath:
        #         fd.write('%s\n' %item)
        with open('trainFilePath.txt', 'r') as fd:
            sourcePath = [l[:-1] for l in fd.readlines()]

    featureList = []
    sourceTweet = {}

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
            hashtag = 0
            url = 0
            for words in tokenWords:
                if words[0].isupper(): # percentage of words that are upper
                    upperCount = upperCount + 1
                if words == '#': # if tweet has hastag
                    hashtag = 1
                if words.startswith('http'): # if tweet is referring to some url
                    url = 1
            innerFeature.append(hashtag)
            innerFeature.append(url)
            perUpper = upperCount / tokenCount
            innerFeature.append(perUpper)
            retweet = jsonFile['retweet_count'] / 1000 # retweet count
            innerFeature.append(retweet)
            if tokenWords[-1] == '?':   # question mark at the end
                questionMark = 1
            else:
                questionMark = 0
            innerFeature.append(questionMark)
            favorite = jsonFile['favorite_count'] / 1000 # favorite count
            innerFeature.append(favorite)
            innerSource[tweetID] = innerFeature
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

    y_pred=clf.predict(x_test)
    accuracyScore = clf.score(x_test, y_test)
    print('Ensemble Random Forest  accuracy score for test set=%0.2f' % accuracyScore)
    confusion_matrix_plot(y_test,y_pred)

def confusion_matrix_plot(y_true, y_pred,normalize=True,cmap=plt.cm.Blues):
    """
    source:
    https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    """
    title = 'Confusion matrix'
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # labels that appear in the data
    classes = np.array(['True','False','Unverified'])
    classes=classes[unique_labels(y_true, y_pred)]
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print(title)
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)

    # All ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


if __name__=='__main__':

    #Preprocess and creating features
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
    plt.show()
