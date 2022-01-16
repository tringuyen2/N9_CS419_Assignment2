'''
    Template for polynomial regression
    AUTHOR Eric Eaton, Xiaoxiang Hu
'''

import numpy as np


#-----------------------------------------------------------------
#  Class PolynomialRegression
#-----------------------------------------------------------------

class PolynomialRegression:

    def __init__(self, degree = 1, regLambda = 1E-8):
        '''
        Constructor
        '''
        #TODO
        self.degree = degree
        self.regLambda = regLambda
        self.theta = None 
        self.mean = None
        self.std = None


    def polyfeatures(self, X, degree):
        '''
        Expands the given X into an n * d array of polynomial features of
            degree d.

        Returns:
            A n-by-d numpy array, with each row comprising of
            X, X * X, X ** 3, ... up to the dth power of X.
            Note that the returned matrix will not inlude the zero-th power.

        Arguments:
            X is an n-by-1 column numpy array
            degree is a positive integer
        '''
        #TODO
        X=np.array(X)
        n=X.shape[0]
        store=np.zeros((n,degree))
        for i in range(n):
            x=X[i]
            for j in range(degree):
                store[i,j]=x**(j+1)
        return store
        

    def fit(self, X, y):
        '''
            Trains the model
            Arguments:
                X is a n-by-1 array
                y is an n-by-1 array
            Returns:
                No return value
            Note:
                You need to apply polynomial expansion and scaling
                at first
        '''
        #TODO
        X = self.polyfeatures(X, self.degree)
        my_data=X

        # Standardize
        n,d=X.shape

        # Add a row of ones for the bias term
        self.mean=np.mean(my_data,axis=0)
        self.std=np.std(my_data,axis=0)
        std_x=(my_data - np.mean(my_data,axis=0))/np.std(my_data,axis=0)

        # Add 1 to beginning of X 
        X = np.c_[np.ones((n,1)), std_x]

        # construct reg matrix
        regular = self.regLambda * np.ones((d+1,1))
        regular[0,0] = 0
        # lr_model = LinearRegression(init_theta = theta, alpha = alpha, n_iter = n_iter)
        # lr_model.fit(x,y,self.regLambda)
        # thetaClosedForm = np.linalg.inv(X.T*X)*X.T*y
        self.theta=np.linalg.pinv(X.T.dot(X)+regular).dot(X.T).dot(y) 
        print(self.theta)
        # self.theta=lr_model.theta
        
        
        
    def predict(self, X):
        '''
        Use the trained model to predict values for each instance in X
        Arguments:
            X is a n-by-1 numpy array
        Returns:
            an n-by-1 numpy array of the predictions
        '''
        # TODO
        x=self.polyfeatures(X, self.degree)
        my_data=x
        n,d=x.shape
        std_x=(my_data - self.mean)/self.std
        X = np.c_[np.ones((n,1)), std_x]
        print(X.dot(self.theta))
       
        return X.dot(self.theta)
        


#-----------------------------------------------------------------
#  End of Class PolynomialRegression
#-----------------------------------------------------------------


def learningCurve(Xtrain, Ytrain, Xtest, Ytest, regLambda, degree):
    '''
    Compute learning curve
        
    Arguments:
        Xtrain -- Training X, n-by-1 matrix
        Ytrain -- Training y, n-by-1 matrix
        Xtest -- Testing X, m-by-1 matrix
        Ytest -- Testing Y, m-by-1 matrix
        regLambda -- regularization factor
        degree -- polynomial degree
        
    Returns:
        errorTrains -- errorTrains[i] is the training accuracy using
        model trained by Xtrain[0:(i+1)]
        errorTests -- errorTrains[i] is the testing accuracy using
        model trained by Xtrain[0:(i+1)]
        
    Note:
        errorTrains[0:1] and errorTests[0:1] won't actually matter, since we start displaying the learning curve at n = 2 (or higher)
    '''
    
    n = len(Xtrain);
    
    errorTrain = np.zeros((n))
    errorTest = np.zeros((n))
    for i in xrange(2, n):
        Xtrain_subset = Xtrain[:(i+1)]
        Ytrain_subset = Ytrain[:(i+1)]
        model = PolynomialRegression(degree, regLambda)
        model.fit(Xtrain_subset,Ytrain_subset)
        
        predictTrain = model.predict(Xtrain_subset)
        err = predictTrain - Ytrain_subset;
        errorTrain[i] = np.multiply(err, err).mean();
        
        predictTest = model.predict(Xtest)
        err = predictTest - Ytest;
        errorTest[i] = np.multiply(err, err).mean();
    
    return (errorTrain, errorTest)