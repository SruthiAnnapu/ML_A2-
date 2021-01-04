
import time
import scipy
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
import pandas as pa
from scipy.stats import friedmanchisquare


# Import data
result = pa.DataFrame(columns=['Fold', 'Logistic', 'Naive_bayes', 'SGD'])
result['Fold'] = range(1, 11)

title_list = []
# headings for the coloumns
for x in range(0, 58):
    title_list.append(str(x))
spamdata = pa.read_csv("spambase.data", names=title_list)

data = spamdata.iloc[:, 0:57]
X = data.values
y = spamdata.iloc[:, 57].values

skf = StratifiedKFold(n_splits=10)
skf.split(X, y)
count = 1
list_acc = []

xa, ya, za, xf, yf, zf, xt, yt, zt  = [],[],[],[],[],[],[],[],[]


for train_index, test_index in skf.split(X, y):

    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # logistic regression
    logreg = LogisticRegression(solver='liblinear')
    start = time.time()
    logreg.fit(X_train, y_train)
    end = time.time()
    eval_time = end - start
    pred = logreg.predict(X_test)
    xa.append(accuracy_score(pred, y_test))
    xf.append(f1_score(pred, y_test))
    xt.append(eval_time)


    # Naive Bayes
    spam_bayes = MultinomialNB()
    start = time.time()
    spam_bayes.fit(X_train, y_train)
    end = time.time()
    eval_time = end - start
    pred = spam_bayes.predict(X_test)
    ya.append(accuracy_score(pred, y_test))
    yf.append(f1_score(pred, y_test))
    yt.append(eval_time)

    # stocastic gradient descent
    clf = SGDClassifier(loss="hinge", penalty="l2", tol=0.001)
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    eval_time = end - start
    pred = clf.predict(X_test)
    za.append(accuracy_score(pred, y_test))
    zf.append(f1_score(pred, y_test))
    zt.append(eval_time)

print("accuracy of LR in each fold is :")
print(xa)
print("")

print("accuracy of NB in each fold is :")
print(ya)
print("")

print("accuracy of SGD in each fold is :")
print(za)
print("")


print("f1 score of LR in each fold is :")
print(xf)
print("")

print("f1 score of NB in each fold is :")
print(yf)
print("")

print("f1 score of SGD in each fold is :")
print(zf)
print("")


print("trining time of LR in each fold is :")
print(xt)
print("")

print("trining time of NB in each fold is :")
print(yt)
print("")

print("trining time of SGD in each fold is :")
print(zt)
print("")

print("average accuracy of LR is :"+str(sum(xa)/10))
print("average accuracy of NB is :"+str(sum(ya)/10))
print("average accuracy of SGD is :"+str(sum(za)/10))
print("")

print("average f1-score of LR is :"+str(sum(xf)/10))
print("average f1-score of NB is :"+str(sum(yf)/10))
print("average f1-score of SGD is :"+str(sum(zf)/10))
print("")

print("average trining time of LR is :"+str(sum(xt)/10))
print("average trining time of NB is :"+str(sum(yt)/10))
print("average trining time of SGD is :"+str(sum(zt)/10))
print("")

fa, p = friedmanchisquare(xa, ya, za)
print("friedmans statistic for accuracy is :"+str(fa))

ff, p = friedmanchisquare(xf, yf, zf)
print("friedmans statistic for f1-score is :"+str(ff))

ft, p = friedmanchisquare(xt, yt, zt)
print("friedmans statistic for accuracy is :"+str(fa))









