import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize



# GENERATE TESTING DATA:
classA = np.concatenate(
    (np.random.randn(10, 2) * 0.2 + [1.5, 0.5],
    np.random.randn(10, 2) * 0.2 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]),
    np.ones(classB.shape[0])))

N = inputs.shape[0]

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, : ]
targets = targets[permute]


def linearKernel(x,y):
    '''
    Linear Kernel:
    This kernel simply returns the scalar product between the two points.
    This results in a linear separation.
    '''
    
    return np.dot(x, y)

def polynomialKernel(x,y):
    '''
    Polynomial Kernel:
    This kernel allows for curved decision boundaries. The exponent p (a
    positive integer) controls the degree of the polynomials. p = 2 will make
    quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher
    will result in more complex shapes.
    '''
    p = 2
    return (np.dot(x.transpose(),y) + 1)**p

def RBF(x,y):
    '''
    Radial Basis Function (RBF) kernel:
    This kernel uses the explicit euclidian distance between the two datapoints,
    and often results in very good boundaries. The parameter σ is used to
    control the smoothness of the boundary.
    '''
    sigma = 0.5
    return math.exp( -(np.linalg.norm(x, y)^2) / (2*sigma**2) )

global ALPHA_NONZERO, X_NONZERO, Y_NONZERO, T_NONZERO, B 

KERNEL = polynomialKernel
T = targets
X = inputs[0,:]
Y = inputs[1,:]
P_MATRIX = np.matrix(N,N)
for i in range(N):
    for j in range(N):
        P_MATRIX(i,j) = T[i]*T[j]*KERNEL(inputs[i], inputs[j])


def objectiveFunction(alpha):
    return 0.5 * np.dot( np.dot(alpha, alpha), P_MATRIX ) - np.sum(alpha)

def zerofun(alpha):
    return np.dot(alpha, T)

def indicator(x, y):
    global ALPHA_NONZERO, T_NONZERO, B
    return np.sum(np.dot(np.dot(ALPHA_NONZERO, T_NONZERO), KERNEL(x, y))) - B

def main():
    global ALPHA_NONZERO, T_NONZERO, B
    '''
    TODO: Suitable Kernel Function              [X]
    TODO: Implement function objective          [X]
    TODO: Implement function zerofun            [X]
    TODO: Call minimize                         [X]
    TODO: Extract the non-zero Alpha values     [X]
    TODO: Calculate the b value using eq. (7)   [X]
    TODO: Implement the indicator function      [X]

    TODO: Genereate test data for visualization and results. 
    '''

    # CALL MINIMIZE:
    C = 10

    start = np.zeros(N)
    B = [(0,C) for b in range(N)]
    XC = {'type':'eq', 'fun':zerofun}

    ret = minimize(objectiveFunction, start, bounds=B, constraints=XC)
    alpha = ret['x']

    # EXTRACT NON-ZERO ALPHA VALUES:
    index = np.where(alpha > 10**(-5))

    X_NONZERO = X[index]
    Y_NONZERO = Y[index]
    T_NONZERO = T[index]
    ALPHA_NONZERO = alpha[index]
    
    # CALCULATE THE b VALUE USING EQ. (7):
    # B = 0
    # for i in range(len(ALPHA_NONZERO)):
    #     B += ALPHA_NONZERO[i]* T_NONZERO[i]*KERNEL(X_NONZERO,Y_NONZERO)
    # B = B - T_NONZERO[0]
    B = np.sum( np.dot(ALPHA_NONZERO, np.dot(T_NONZERO, KERNEL(X_NONZERO,Y_NONZERO) ) ) ) - T_NONZERO[0]
    print(B)

    plt.plot([p[0] for p in classA],
             [p[1] for p in classA], 
             'b.')
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB], 
             'r.')
    plt.axis('equal')
    plt.savefig('svmplot.pdf')

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array( [[indicator(x, y)
                        for x in xgrid]
                        for y in ygrid])
    plt.contour(xgrid,ygrid,grid, (-1.0, 0.0, 1.0),
                colors = ('red','black', 'blue'),
                linewidths = (1,3,1))
 
    plt.show()



if __name__ == "__main__":
    main()
