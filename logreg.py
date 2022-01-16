import numpy as np
from numpy import linalg 

class LogisticRegression:

    def __init__(self, alpha = 0.01, regLambda=0.01, regNorm=2, epsilon=0.0001, maxNumIters = 10000):
        '''
        Constructor
        Arguments:
            alpha is the learning rate
            regLambda is the regularization parameter
            regNorm is the type of regularization (either L1 or L2, denoted by a 1 or a 2)
            epsilon is the convergence parameter
            maxNumIters is the maximum number of iterations to run
        '''
        self.alpha=alpha
        self.regLambda=regLambda
        self.regNorm=regNorm
        self.epsilon=epsilon
        self.maxNumIters=maxNumIters
        self.theta=None
        
      

    def computeCost(self, theta, X, y, regLambda):
        '''
        Computes the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            a scalar value of the cost  ** make certain you're not returning a 1 x 1 matrix! **
        '''
        n,d=X.shape
        Sig_out=self.sigmoid(X*theta)
#        regular=regLambda * np.ones((d,1))
#        regular[0,0]=0
        if self.regNorm == 2:
            regular_theta=np.linalg.norm(theta,2)
        elif self.regNorm == 1  :
            regular_theta=np.linalg.norm(theta,1)           
        J =  -((y).T*np.log(Sig_out)+(1-y).T* np.log(1-Sig_out)) + 0.5*regLambda*regular_theta
        J_scalar = J.tolist()[0][0]  # convert matrix to scalar
        return J_scalar
    
    
    def computeGradient(self, theta, X, y, regLambda):
        '''
        Computes the gradient of the objective function
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            regLambda is the scalar regularization constant
        Returns:
            the gradient, an d-dimensional vector
        '''
        n,d = X.shape
        
        regular=regLambda * np.ones((d,1))
        regular[0,0]=0
        
        if self.regNorm == 1:
            gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda*np.sign(theta))
        else:
            gradient = (X.T * (self.sigmoid(X * theta) - y) + regLambda*theta) 
        
#        gradient=(X.T * (self.sigmoid(Z) - y) + regLambda*theta) 
        # l,f = theta.shape
        # print "n = %d" % (n)
        # print "d = %d" % (d)
        # print "l = %d" % (l)
        # print "f = %d" % (f)
        
        # don't regularize the theta_0 parameter
        gradient[0] = sum(self.sigmoid(X * theta) - y) 
#        print(gradient)
#        gradient = X.T*((self.sigmoid(np.matmul(X,theta)))-y) +(regular.T*(theta)) 
        return gradient

    def hasConverged(self, new_theta,old_theta):
        if np.linalg.norm(new_theta - old_theta) <= self.epsilon:
            return True
        else:
            return False
        
    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n =X.shape[0]
        X = np.c_[np.ones((n,1)), X]
        n,d = X.shape
        self.theta = np.matrix(np.random.randn((d))).T
        theta_new=self.theta
        theta_old=self.theta
        for i in range(self.maxNumIters):
            theta_new = self.theta - self.alpha*self.computeGradient(self.theta, X, y, self.regLambda)
            if self.hasConverged(theta_new,theta_old):
                self.theta=theta_new
                break
            else:
                theta_old=theta_new
                self.theta=theta_new  
                cost=self.computeCost(self.theta,X, y,self.regLambda) 
                
                #print("Iteration: ", i+1, " Cost: ", cost, " Theta: ", self.theta)
        

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        n,d=X.shape
        X = np.c_[np.ones((n,1)), X]
        y_final=self.sigmoid(np.matmul(X,self.theta))
        for i in range(y_final.shape[0]):
            if y_final[i] > 0.5:
                y_final[i]=1
            elif y_final[i]<= 0.5:
                y_final[i]=0
        return np.array(y_final)
    def sigmoid(self, Z):
        '''
        Computes the sigmoid function 1/(1+exp(-z))
        '''
        return 1/(1+np.exp(-Z))