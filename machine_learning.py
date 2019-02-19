# coding=utf-8
import csv
from sklearn import svm
from sklearn.model_selection import learning_curve# import validation_curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import naive_bayes
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from matplotlib.font_manager import *
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.multiclass import OneVsRestClassifier
myfont = FontProperties(fname='/usr/share/fonts/truetype/wqy/wqy-microhei.ttc')
import warnings
warnings.filterwarnings('ignore')

weather_contain = ['SO2', 'NO2', 'PM10', 'CO', 'O38h', 'PM2-5']
data_dir = '../数据处理'
interval = 1
interval0 = 20
breath_max_num = 1


def show_data(Y_pred, Y_test, title=''):
    plt.title(title)
    plt.xlabel(u'时间', fontproperties=myfont)
    plt.ylabel(u'呼吸科门诊量', fontproperties=myfont)
    l1, = plt.plot(np.arange(0, len(Y_test)), Y_test, color='r')
    l2, = plt.plot(np.arange(0, len(Y_pred)), Y_pred, color='b')
    plt.legend([l1, l2], ['label', 'predict'], loc='upper right')
    plt.show()


def LinearRegression(X_train, Y_train, X_test, Y_test):
    print ('LinearRegression...')
    reg = linear_model.LinearRegression()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    # show_data(Y_pred, Y_train, title='train')
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)

    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    show_data(Y_pred, Y_test, title='test')

    # pca = PCA(n_components=1).fit(X_train)
    # X_train_1 = pca.transform(X_train)
    #
    # plt.scatter(X_train_1, Y_train)
    # plt.plot(X_train_1, reg.predict(X_train).astype(int))
    # plt.show()
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def Ridge(X_train, Y_train, X_test, Y_test):
    print ('Ridge...')
    reg = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    # show_data(Y_pred, Y_train, title='train')
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    # show_data(Y_pred, Y_test, title='test')
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def Lasso(X_train, Y_train, X_test, Y_test):
    print ('Lasso...')
    reg = linear_model.Lasso(alpha=0.1)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def BayesianRidge(X_train, Y_train, X_test, Y_test):
    print ('BayesianRidge...')
    reg = linear_model.BayesianRidge()
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def LogisticRegression(X_train, Y_train, X_test, Y_test):
    print ('LogisticRegression...')
    reg = linear_model.LogisticRegression(multi_class='multinomial', solver='sag', max_iter=100)
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def SGDRegressor(X_train, Y_train, X_test, Y_test):
    print ('SGDRegressor...')
    reg = linear_model.SGDClassifier(loss='hinge', penalty='l1')
    reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def Polynomial(X_train, Y_train, X_test, Y_test):
    print ('Polynomial...')
    reg = Pipeline(
        [('poly', PolynomialFeatures(degree=1)),
         ('linear', linear_model.LinearRegression(fit_intercept=False))]
    )
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def SVC(X_train, Y_train, X_test, Y_test):
    print ('SVC...')
    reg = svm.SVR()
    # c_range = np.arange(1, 20, 2) / 10.0
    # gamma_range = np.logspace(-10, 10, 20, base=10)
    # param_grid = [{'kernel': ['rbf'], 'C': c_range, 'gamma': gamma_range}]

    param_grid = [
        {
            'C': np.arange(1, 20)/10.0,
            'gamma': np.logspace(-10, 10, 21, base=10),
            'kernel': ['rbf']
        },
        {
            'C': np.arange(1, 20)/10.0,
            'kernel': ['linear']
        }
    ]

    grid = GridSearchCV(reg, param_grid, cv=5, n_jobs=4)
    grid.fit(X_train, Y_train)
    Y_pred = grid.predict(X_train)

    Y_pred = Y_pred.astype(int)
    Y_train = (Y_train / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = grid.predict(X_test)
    Y_pred = Y_pred.astype(int)

    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracy_score', accuracy_score(Y_test, Y_pred))
    print


def LinearSVC(X_train, Y_train, X_test, Y_test):
    print ('LinearSVC...')
    reg = svm.LinearSVC()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def RandomForest(X_train, Y_train, X_test, Y_test):
    print ('RandomForest...')
    reg = RandomForestClassifier()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def GaussianNB(X_train, Y_train, X_test, Y_test):
    print ('GaussianNB...')
    reg = naive_bayes.GaussianNB()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def MultinomialNB(X_train, Y_train, X_test, Y_test):
    print ('MultinomialNB...')
    reg = naive_bayes.MultinomialNB()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def BernoulliNB(X_train, Y_train, X_test, Y_test):
    print ('BernoulliNB...')
    reg = naive_bayes.BernoulliNB()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def DecisionTreeClassifier(X_train, Y_train, X_test, Y_test):
    print ('DecisionTreeClassifier...')
    reg = tree.DecisionTreeClassifier()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def AdaBoost(X_train, Y_train, X_test, Y_test):
    print ('AdaBoostClassifier...')
    reg = AdaBoostClassifier()
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def MLP(X_train, Y_train, X_test, Y_test):
    print ('MLPClassifier...')
    reg = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(20, 10, 20), random_state=1)
    reg = reg.fit(X_train, Y_train)
    Y_pred = reg.predict(X_train)
    Y_pred = Y_pred.astype(int)
    print ('train accuracy_score:', accuracy_score(Y_train, Y_pred))

    Y_pred = reg.predict(X_test)
    Y_pred = Y_pred.astype(int)
    Y_test = (Y_test / interval).astype(int)
    Y_pred = (Y_pred / interval).astype(int)
    print ('test accuracu_score', accuracy_score(Y_test, Y_pred))
    print


def original_data():
    '''
    去除节假日的门诊量，具体做法是把总门诊量小于1000的日期去掉
    '''
    file_weather = '../数据处理/weather.csv'
    file_outpatient = '../数据处理/outpatient.csv'
    weather_list = []
    outpatient_list = []
    with open(file_weather, 'r') as f:
        weather = csv.reader(f)
        for row in weather:
            tmp = []
            for i in row:
                tmp.append(float(i))
            weather_list.append(tmp)
    with open(file_outpatient, 'r') as f:
        outpatient = csv.reader(f)
        for row in outpatient:
            tmp = []
            for i in row:
                tmp.append(float(i))
            outpatient_list.append(tmp)
    weather_list = np.array(weather_list)
    outpatient_list = np.array(outpatient_list)
    index = np.where(outpatient_list[:, 0] > 1000)
    weather_list = weather_list[index]
    outpatient_list = outpatient_list[index]

    plt.plot(np.arange(len(outpatient_list[:, 1])), outpatient_list[:, 1])
    plt.show()
    Y = (outpatient_list[:, 1]/interval0).astype(int)
    X = weather_list

    length = len(Y)
    X_train, Y_train = X[:length * 2 / 3], Y[:length * 2 / 3]
    X_test, Y_test = X[length * 2 / 3:], Y[length * 2 / 3:]
    # 分类器
    LinearRegression(X_train, Y_train, X_test, Y_test)
    Ridge(X_train, Y_train, X_test, Y_test)
    Lasso(X_train, Y_train, X_test, Y_test)
    BayesianRidge(X_train, Y_train, X_test, Y_test)
    LogisticRegression(X_train, Y_train, X_test, Y_test)
    SGDRegressor(X_train, Y_train, X_test, Y_test)
    Polynomial(X_train, Y_train, X_test, Y_test)
    SVC(X_train, Y_train, X_test, Y_test)
    RandomForest(X_train, Y_train, X_test, Y_test)
    GaussianNB(X_train, Y_train, X_test, Y_test)
    MultinomialNB(X_train, Y_train, X_test, Y_test)
    BernoulliNB(X_train, Y_train, X_test, Y_test)
    DecisionTreeClassifier(X_train, Y_train, X_test, Y_test)
    AdaBoost(X_train, Y_train, X_test, Y_test)
    MLP(X_train, Y_train, X_test, Y_test)


def main():
    breath_num = np.load(data_dir + '/呼吸科门诊量.npy')
    SO2 = np.load(data_dir + '/SO2浓度.npy')
    NO2 = np.load(data_dir + '/NO2浓度.npy')
    PM10 = np.load(data_dir + '/PM10浓度.npy')
    CO = np.load(data_dir + '/CO浓度.npy')
    O38h = np.load(data_dir + '/O38h浓度.npy')
    PM2_5 = np.load(data_dir + '/PM2-5浓度.npy')
    #时间序列
    time = []
    for i in np.arange(1, 50):
        time.append([i, i])
    time = np.array(time)
    time = time.reshape(-1)
    time = np.hstack((time, time, time))
    print (time)

    #对y进行区间划分
    Y = (breath_num / interval0).astype(int)
    #对X进行归一化
    SO2 = SO2 / float(SO2.max())
    NO2 = NO2 / float(NO2.max())
    PM10 = PM10 / float(PM10.max())
    CO = CO / float(CO.max())
    O38h = O38h / float(O38h.max())
    PM2_5 = PM2_5 / float(PM2_5.max())
    time = time / float(time.max())
    Y = Y / float(Y.max())

    # X = np.vstack([SO2, NO2, PM10, CO, O38h, PM2_5, time])
    X = np.vstack ( [SO2,time] )
    X = X.transpose((1, 0))
    print (X)
    # X -= X.mean(axis=0)

    # X = X[index]
    # Y = Y[index]
    # plt.plot(np.arange(len(Y)), Y, color='r')
    # plt.plot(np.arange(len(X)), X[:, 5], color='b')
    # plt.show()

    # plt.title(u'空气质量-时间', fontproperties=myfont)
    # plt.xlabel(u'时间', fontproperties=myfont)
    # plt.ylabel(u'空气质量', fontproperties=myfont)
    # l1, = plt.plot(np.arange(0, len(SO2)), SO2, color='r')
    # l2, = plt.plot(np.arange(0, len(NO2)), NO2, color='g')
    # l3, = plt.plot(np.arange(0, len(PM10)), PM10, color='b')
    # l4, = plt.plot(np.arange(0, len(CO)), CO, color='000')
    # l5, = plt.plot(np.arange(0, len(O38h)), O38h, color='056')
    # l6, = plt.plot(np.arange(0, len(PM2_5)), PM2_5, color='100')
    #
    # plt.legend([l1, l2, l3, l4, l5, l6], weather_contain, loc='upper right', prop=myfont)
    # plt.show()

    length = len(Y)
    X_train, Y_train = X[:int(length*2/3)], Y[:int(length*2/3)]
    X_test, Y_test = X[int(length*2/3):], Y[int(length*2/3):]
    #
    # plt.plot(np.arange(len(Y_train)), Y_train / float(Y_train.max()), color='r')
    # plt.plot(np.arange(len(X_train)), X_train[:, 0] / X_train[:, 0].max(), color='b')
    # plt.show()

    # plt.plot(np.arange(len(Y)), Y/float(Y.max()), color='r')
    # plt.plot(np.arange(len(X)), X[:, 5], color='b')
    # plt.show()
    # plt.scatter(PM2_5, Y)
    # plt.show()

    #pca降维分析
    # pca = PCA(n_components=2).fit(X_train)
    # X_train = pca.transform(X_train)
    # plt.scatter(np.arange(len(X_train)), X_train)
    # plt.show()
    # X_test = pca.transform(X_test)
    # plt.scatter(np.arange(len(X_test)), X_test)
    # plt.show()

    #分类器
    LinearRegression(X_train, Y_train, X_test, Y_test)
    # Ridge(X_train, Y_train, X_test, Y_test)
    # Lasso(X_train, Y_train, X_test, Y_test)
    # BayesianRidge(X_train, Y_train, X_test, Y_test)
    # LogisticRegression(X_train, Y_train, X_test, Y_test)
    # SGDRegressor(X_train, Y_train, X_test, Y_test)
    # Polynomial(X_train, Y_train, X_test, Y_test)
    # LinearSVC(X_train, Y_train, X_test, Y_test)
    # SVC(X_train, Y_train, X_test, Y_test)
    # RandomForest(X_train, Y_train, X_test, Y_test)
    # GaussianNB(X_train, Y_train, X_test, Y_test)
    # MultinomialNB(X_train, Y_train, X_test, Y_test)
    # BernoulliNB(X_train, Y_train, X_test, Y_test)
    # DecisionTreeClassifier(X_train, Y_train, X_test, Y_test)
    # AdaBoost(X_train, Y_train, X_test, Y_test)
    # MLP(X_train, Y_train, X_test, Y_test)

    # plt.plot(np.linspace(-10, 10, 100), 1.0 / (np.exp(-np.linspace(-10, 10, 100))+1))
    # plt.show()


if __name__ == '__main__':
    # original_data()
    main()