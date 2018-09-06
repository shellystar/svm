# python: 3.5.2
# encoding: utf-8

import numpy as np
import cvxopt 
import matplotlib.pyplot as plt
import math

def load_data(fname):
    with open(fname, 'r') as f:
        data = []
        line = f.readline()
        for line in f:
            line = line.strip().split()
            x1 = float(line[0])
            x2 = float(line[1])
            t = int(line[2])
            data.append([x1, x2, t])
        return np.array(data)

def eval_acc(label, pred):
    return np.sum(label == pred) / len(pred)

# calculate kernel, flag indicates which kernel method is used
# when using polynomial kernel, d indicates the times of polynomial kernel
# when using gauss kernel, sigma indicates the standard of gauss kernel
# all input should be matrix
def kernel(pi, pj, flag, d=2, sigma=1):
    if flag == 'linear':
        return np.dot(pi, np.transpose(pj))
    if flag == 'polynomial':
        return pow(np.dot(pi, np.transpose(pj)) + 1, d)
    if flag == 'gauss':
        return np.exp(-1 * (np.power(np.linalg.norm(pi - pj, axis=1),2) / (2 * pow(sigma, 2))))

def drawContour(data1, data2, predictFunc, color1='ro', color2='b*', title='SVM'):
    # points
    plt.plot(data1[:, 0], data1[:, 1], 'ro')
    plt.plot(data2[:, 0], data2[:, 1], 'b*')

    # contour
    maxX1 = math.ceil(max(max(data1[:,0]), max(data2[:,0])))
    maxX2 = math.ceil(max(max(data1[:,1]), max(data2[:,1])))
    x1, x2 = np.meshgrid(np.linspace(0, maxX1, 200), np.linspace(0, maxX2, 200))
    y = predictFunc(x1.ravel(), x2.ravel()).reshape(x1.shape)
    plt.contour(x1, x2, y, 1, linewidths=1, linestyles='solid')

    # labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.show()

def drawLine(dataRed, dataBlue, w, b, title='Linear Classifier'):
    # points
    plt.plot(dataRed[:, 0], dataRed[:, 1], 'ro')
    plt.plot(dataBlue[:, 0], dataBlue[:, 1], 'b*')

    # line
    maxX1 = math.ceil(max(max(dataRed[:,0]), max(dataBlue[:,0])))
    minX1 = math.floor(min(min(dataRed[:, 0]), min(dataBlue[:, 0])))
    x0 = np.linspace(minX1, maxX1, 200)
    w0 = w[0][0]
    w1 = w[0][1]
    x1 = (-b - w0 * x0)/w1

    # labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.plot(x0, x1, color="pink")
    plt.show()
 
def drawMultiClass(data1, data2, data3, svmList, title='Multi SVM'):
    # points
    plt.plot(data1[:, 0], data1[:, 1], 'ro')
    plt.plot(data2[:, 0], data2[:, 1], 'b*')
    plt.plot(data3[:, 0], data3[:, 1], 'g+')

    # contour
    max11=max(data1[:,0])
    max21=max(data2[:,0])
    max31=max(data3[:,0])
    max12=max(data1[:,1])
    max22=max(data2[:,1])
    max32=max(data3[:,1])
    maxX1 = math.ceil(max([max11, max21, max31]))
    maxX2 = math.ceil(max([max12, max22, max32]))
    x1, x2 = np.meshgrid(np.linspace(0, maxX1, 500), np.linspace(0, maxX2, 500))
    y = multiSVMPredict(x1.ravel(), x2.ravel(), svmList).reshape(x1.shape)
    plt.contour(x1, x2, y, colors='magenta', linewidths=1, linestyles='solid')

    # labels and title
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title(title)

    plt.show()

class SVM():

    def __init__(self, learningRate=1e-5, normLambda=0.01):
        self.sgdW = np.random.rand(1,2)
        self.sgdb = np.random.rand(1,1)
        self.learningRate = learningRate
        self.normLambda = normLambda
    
    def sgd(self, dataTrain):
        predY = np.dot(dataTrain[:, :2], np.transpose(self.sgdW)) + self.sgdb

        # loss
        oneMinusyt = 1 - predY * dataTrain[:, 2].reshape(len(dataTrain), 1)
        oneMinusyt[oneMinusyt < 0] = 0
        self.loss = np.sum(oneMinusyt, axis=0)

        # calculate gradient and sgd training
        oneMinusyt[oneMinusyt > 0] = 1
        tmp = np.transpose(oneMinusyt * dataTrain[:, 2].reshape(len(dataTrain),1))
        wgrad = - np.dot(tmp, dataTrain[:,:2])
        bgrad = - np.dot(np.transpose(oneMinusyt), dataTrain[:,2].reshape(len(dataTrain),1))
        self.sgdW = self.sgdW - self.learningRate * wgrad
        self.sgdb = self.sgdb - self.learningRate * bgrad
    
    def sgdTrain(self, dataTrain, epoch=100):
        for i in range(epoch):
            self.sgd(dataTrain)
            print ("The ", i, " th epoch, Loss is ", self.loss)

    def predict(self, xi, xj):
        if xi.ndim == 1 or xi.shape[1] == 1:
            x = np.concatenate((xi.reshape(len(xi), 1), xj.reshape(len(xj), 1)), axis=1)
        
        predY = np.dot(x, np.transpose(self.sgdW)) + self.sgdb
        predY[predY > 0] = 1
        predY[predY < 0] = -1
        return np.ravel(predY)

    def train(self, data_train, kernelMethod='gauss'):
        self.kernelMethod = kernelMethod
        # get data number
        N = data_train.shape[0]
        # Q represents the coefficient of quardradic part
        Q = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                kernelValue = kernel(np.reshape(data_train[i, :2], (1,2)),np.reshape(data_train[j, :2], (1,2)), flag=self.kernelMethod, d=3, sigma=5.0)
                Q[i][j] = data_train[i, 2] * data_train[j, 2] * kernelValue 
        Q = cvxopt.matrix(Q, (N, N), 'd')#-1 * cvxopt.matrix(Q)
        # P represents the coefficient of linear part
        P = np.ones(N)
        P = -1 * P#P 
        P = cvxopt.matrix(P, (N, 1), 'd')
        # G represents inequality constraint: -ai <= 0
        G = -1 * np.eye(N, N)
        G = cvxopt.matrix(G, (N, N), 'd')
        h = cvxopt.matrix(np.zeros(N), (N, 1), 'd')
        # A,b represents equality constraint: a1*t1 + a2*t2 + ... + aN*tN = 0
        A = data_train[:,2]
        A = cvxopt.matrix(A, (1, N), 'd')
        b = cvxopt.matrix(0.0)
        
        solution = cvxopt.solvers.qp(Q, P, G, h, A, b)

        #print ("f: %.3g  x: %s  Df: %s" % (f[0], np.squeeze(x), np.squeeze(Df)))
        aMatrix = np.ravel(solution['x'])
        # store all the support vector, take all values which >= 1e-5 as nonzero
        supportVectorIndex = []
        self.A = []
        for i in range(aMatrix.size):
            if aMatrix[i] >= 1e-5:
                supportVectorIndex.append(i)
                self.A.append(aMatrix[i])
        self.supportVector = []
        self.supportVectorNumber = len(supportVectorIndex)
        for j in range(self.supportVectorNumber):
            self.supportVector.append(data_train[supportVectorIndex[j]])
        self.A = np.array(self.A)
        self.supportVector = np.array(self.supportVector)

        # take one support vector to calculate b
        tmpPj = self.supportVector[0, :2].reshape(1,2)
        tmpTj = self.supportVector[0, 2]
        kernelValue = kernel(self.supportVector[:, :2].reshape(self.supportVectorNumber, 2), tmpPj, flag=self.kernelMethod, d=3, sigma=5.0)  # phi_xm * phi_xj
        at = np.dot(self.A.reshape(1, self.supportVectorNumber), np.diag(self.supportVector[:,2]))
        atk = np.dot(at, kernelValue)
        self.b = 1 / tmpTj - atk

    def predictWithKernel(self, xi, xj):

        if xi.ndim == 1 or xi.shape[1] == 1:
            x = np.concatenate((xi.reshape(len(xi), 1), xj.reshape(len(xj), 1)), axis=1)

        K = []
        for i in range(self.supportVectorNumber):
            K.append(kernel(x, np.reshape(self.supportVector[i, :2], (1,2)), flag=self.kernelMethod, d=3, sigma=5.0))
        K = np.transpose(np.array(K))
        # each row contains a test point's all kernel values with support vector points
        pred_y = np.dot(K, np.dot(self.A, np.diag(self.supportVector[:,2]))) + self.b
        pred_y[pred_y > 0] = 1
        pred_y[pred_y < 0] = -1
        return np.ravel(pred_y)

class linearClassifier():
    def __init__(self, learningRate=1e-7, normLambda=0.01):
        self.w = np.random.rand(1,2)
        self.b = np.random.rand(1,1)
        self.lambdaV = normLambda
        self.learningRate = learningRate
    
    def sgd(self, dataTrain):
        predY = np.dot(dataTrain[:, :2], np.transpose(self.w)) + self.b

        # squared loss calculate
        ySubt = np.subtract(predY, dataTrain[:, 2].reshape(len(dataTrain), 1)) 
        self.loss = np.power(np.linalg.norm(ySubt, axis=0),2) + self.lambdaV * np.power(np.linalg.norm(self.w, axis=1),2)

        # calculate gradient and sgd training
        wgrad = np.dot(np.transpose(2 * ySubt), dataTrain[:, :2]) + 2 * self.lambdaV * self.w
        bgrad = np.sum(2 * ySubt, axis=0)
        self.w = self.w - self.learningRate * wgrad
        self.b = self.b - self.learningRate * bgrad

    def train(self, dataTrain, epoch=100):
        for i in range(epoch):
            self.sgd(dataTrain)
            print ("The ", i, " th epoch, Loss is ", self.loss)
    
    def predict(self, xi, xj):
        if xi.ndim == 1 or xi.shape[1] == 1:
            x = np.concatenate((xi.reshape(len(xi), 1), xj.reshape(len(xj), 1)), axis=1)
        
        predY = np.dot(x, np.transpose(self.w)) + self.b
        predY[predY > 0] = 1
        predY[predY < 0] = -1
        return np.ravel(predY)

class logisticClassifier():
    def __init__(self, learningRate=1e-4, normLambda=0.01):
        self.w = np.random.rand(1,2)
        self.b = np.random.rand(1,1)
        self.learningRate = learningRate
        self.normLambda = normLambda
    
    def sgd(self, dataTrain):
        predY = np.dot(dataTrain[:, :2], np.transpose(self.w)) + self.b

        # cross entropy loss
        yt = (predY.reshape(len(predY)) * dataTrain[:, 2]).reshape(len(predY), 1)
        self.loss = np.sum(np.log(1 + np.exp(-yt)), axis=0) + self.normLambda * np.power(np.linalg.norm(self.w, axis=1),2)

        # calculate gradient and sgd training
        tnSigmoidyt = (1/(1 + np.exp(-yt)) - 1) * dataTrain[:, 2].reshape(len(dataTrain), 1)
        wgrad = np.dot(np.transpose(tnSigmoidyt), dataTrain[:, :2]) + 2 * self.normLambda * self.w
        bgrad = np.sum(tnSigmoidyt, axis=0)
        self.w = self.w - self.learningRate * wgrad
        self.b = self.b - self.learningRate * bgrad
    
    def train(self, dataTrain, epoch=100):
        for i in range(epoch):
            self.sgd(dataTrain)
            print ("The ", i, " th epoch, Loss is ", self.loss)

    def predict(self, xi, xj):
        if xi.ndim == 1 or xi.shape[1] == 1:
            x = np.concatenate((xi.reshape(len(xi), 1), xj.reshape(len(xj), 1)), axis=1)

        predY = np.dot(x, np.transpose(self.w)) + self.b
        predY[predY > 0] = 1
        predY[predY < 0] = -1
        return np.ravel(predY)                   

def kernelSVMMain():
    train_file = 'data/train_kernel.txt'
    test_file = 'data/test_kernel.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    svm = SVM()
    svm.train(data_train, kernelMethod='gauss')

    x_train = data_train[:, :2]  # feature [x1, x2]
    t_train = data_train[:, 2]
    t_train_pred = svm.predictWithKernel(x_train[:, 0], x_train[:, 1])
    print ("t_train_pred: ", t_train_pred)
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    t_test_pred = svm.predictWithKernel(x_test[:, 0], x_test[:, 1])
    print ("t_test_pred: ", t_test_pred)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

    drawContour(x_test[:100], x_test[100:], svm.predictWithKernel, title='SVM Gauss Kernel')

def linearClassifierCmpMain():
    train_file = 'data/train_linear.txt'
    test_file = 'data/test_linear.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # linear classifier
    linear = linearClassifier(learningRate=1e-7)
    #linear.train(data_train, epoch=100)

    # logistic classifier
    logistic = logisticClassifier(learningRate=1e-4)
    #logistic.train(data_train, epoch=150)

    # svm classifier
    svm = SVM(learningRate=1e-4)
    svm.sgdTrain(data_train, epoch=200)

    # test and evaluate accuracy
    x_train = data_train[:, :2]
    t_train = data_train[:, 2]
    x_test = data_test[:, :2]
    t_test = data_test[:, 2]
    
    #t_train_pred = linear.predict(x_train[:, 0], x_train[:, 1])
    #t_test_pred = linear.predict(x_test[:, 0], x_test[:, 1])
    #t_train_pred = logistic.predict(x_train[:, 0], x_train[:, 1])
    #t_test_pred = logistic.predict(x_test[:, 0], x_test[:, 1])
    t_train_pred = svm.predict(x_train[:, 0], x_train[:, 1])
    t_test_pred = svm.predict(x_test[:, 0], x_test[:, 1])
    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

    # draw figure
    #drawLine(x_test[:100], x_test[100:], linear.w, linear.b[0][0], title='Linear Classifier')
    #drawLine(x_test[:100], x_test[100:], logistic.w, logistic.b[0][0], title='Logistic Classifier')
    drawLine(x_test[:100], x_test[100:], svm.sgdW, svm.sgdb[0][0], title='SVM Classifier With Hinge Loss')

def multiSVMPredict(dataTesti, dataTestj, svmList):

    if dataTesti.ndim == 1:
        dataTest = np.concatenate((dataTesti.reshape(len(dataTesti), 1), dataTestj.reshape(len(dataTestj), 1)), axis=1)

    t_test_pred_list = []
    for i in range(len(svmList)):
        svm = svmList[i]
        t_test_pred_list.append(svm.predictWithKernel(dataTest[:, 0], dataTest[:, 1]))
    t_test_pred_list = np.array(t_test_pred_list)
    t_test_pred = t_test_pred_list.argmax(axis=0)
    t_test_pred[t_test_pred == 0] = -1 # svm1 divide out label -1
    t_test_pred[t_test_pred == 1] = 0  # svm2 divide out label 0
    t_test_pred[t_test_pred == 2] = 1  # svm3 divide out label 1
    return np.ravel(t_test_pred)

def multiSVM():
    train_file = 'data/train_multi.txt'
    test_file = 'data/test_multi.txt'
    data_train = load_data(train_file)
    data_test = load_data(test_file)

    # use three svm instance to divide all sample points

    # train
    # first treat label 0 and label 1 as the same label
    # then treat label 1 and label -1 as the same label
    # then label -1 and label 0
    svm1 = SVM()
    svm2 = SVM()
    svm3 = SVM()
    svmList = [svm1, svm2, svm3]

    label1 = data_train[:, 2].copy()
    label1[label1 >= 0] = 1
    label1[label1 == -1] = 0
    label1[label1 == 1] = -1
    label1[label1 == 0] = 1
    label2 = data_train[:, 2].copy()
    label2[label2 == 1] = -1
    label2[label2 == 0] = 1
    label3 = data_train[:, 2].copy()
    label3[label3 == 0] = -1
    labelList = [label1, label2, label3]

    for i in range(3):
        svm = svmList[i]
        label = labelList[i]
        data = np.concatenate((data_train[:, :2], label.reshape(len(label), 1)), axis=1)
        svm.train(data, kernelMethod='gauss')

    # test
    t_train = data_train[:, 2]
    t_test = data_test[:, 2]    
    t_train_pred = multiSVMPredict(data_train[:, 0], data_train[:, 1], svmList)
    t_test_pred = multiSVMPredict(data_test[:, 0], data_test[:, 1], svmList)

    acc_train = eval_acc(t_train, t_train_pred)
    acc_test = eval_acc(t_test, t_test_pred)
    print("train accuracy: {:.1f}%".format(acc_train * 100))
    print("test accuracy: {:.1f}%".format(acc_test * 100))

    drawMultiClass(data_test[:100], data_test[100:200], data_test[200:], svmList)

def test():
    data_train = np.array([[1,1,0], [2,2,0], [3,3,0], [10,10,1], [11,11,1], [12,12,1], [20,20,-1], [21,21,-1], [22,22,-1]])
    
    svm1 = SVM()
    svm2 = SVM()
    svm3 = SVM()
    svmList = [svm1, svm2, svm3]

    label1 = data_train[:, 2].copy()
    label1[label1 >= 0] = 1
    label1[label1 == -1] = 0
    label1[label1 == 1] = -1
    label1[label1 == 0] = 1
    label2 = data_train[:, 2].copy()
    label2[label2 == 1] = -1
    label2[label2 == 0] = 1
    label3 = data_train[:, 2].copy()
    label3[label3 == 0] = -1
    labelList = [label1, label2, label3]

    for i in range(3):
        svm = svmList[i]
        label = labelList[i]
        data = np.concatenate((data_train[:, :2], label.reshape(len(label), 1)), axis=1)
        svm.train(data, kernelMethod='gauss')

    # test
    t_train = data_train[:, 2]
    
    t_train_pred_list = []
    for i in range(3):
        svm = svmList[i]
        t_train_pred_list.append(svm.predictWithKernel(data_train[:, 0], data_train[:, 1]))
    t_train_pred_list = np.array(t_train_pred_list)
    t_train_pred = t_train_pred_list.argmax(axis=0) # get the valid svm number
    t_train_pred[t_train_pred == 0] = -1 # svm1 divide out label -1
    t_train_pred[t_train_pred == 1] = 0  # svm2 divide out label 0
    t_train_pred[t_train_pred == 2] = 1  # svm3 divide out label 1

if __name__ == '__main__':
    #kernelSVMMain()
    #linearClassifierCmpMain()
    #test()
    multiSVM()
