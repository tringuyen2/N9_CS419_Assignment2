import numpy as np



def mapFeature(x1, x2):
    '''
    Maps the two input features to quadratic features.
        
    Returns a new feature array with d features, comprising of
        X1, X2, X1 ** 2, X2 ** 2, X1*X2, X1*X2 ** 2, ... up to the 6th power polynomial
        
    Arguments:
        X1 is an n-by-1 column matrix
        X2 is an n-by-1 column matrix
    Returns:
        an n-by-d matrix, where each row represents the new features of the corresponding instance
    '''
    m=0;
    degree=6
    n=x1.shape[0]
    final_out=np.ones((n,27))
    for i in range(1, degree + 1):
            for j in range(0,i + 1):
                final_out[:,m] = np.power(x1,(i - j)) *np.power(x2,j)
                m=m+1
         
                
            
    return np.matrix(final_out)

