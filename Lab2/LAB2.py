import numpy as np
import random, math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

np.random.seed(100)
random.seed(100)

global ALPHA_NONZERO, targets_NONZERO, b, inputs_NONZERO, poly_degree, sigma
classA = np.concatenate(
    (np.random.randn(10, 2) * 1.5 + [1.5, 0.5],
    np.random.randn(10, 2) * 1.5 + [-1.5, 0.5]))
classB = np.random.randn(20, 2) * 1.5 + [0.0, -0.5]



inputs = np.concatenate((classA, classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])))

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
    global poly_degree
    '''
    Polynomial Kernel:
    This kernel allows for curved decision boundaries. The exponent p (a
    positive integer) controls the degree of the polynomials. p = 2 will make
    quadratic shapes (ellipses, parabolas, hyperbolas). Setting p = 3 or higher
    will result in more complex shapes.
    '''
    poly_degree = 2
    return np.power((np.dot(x,y) + 1), poly_degree)

def RBF(x,y):
    global sigma
    '''
    Radial Basis Function (RBF) kernel:
    This kernel uses the explicit euclidian distance between the two datapoints,
    and often results in very good boundaries. The parameter σ is used to
    control the smoothness of the boundary.
    '''
    # sigma = 0.8
    return math.exp( - np.divide(np.power(np.linalg.norm(np.subtract(x,y)),2) , (2*np.power(sigma,2))) )


KERNEL = RBF
P_MATRIX = np.zeros((N,N))
for i in range(N):
    for j in range(N):
        P_MATRIX[i,j] = targets[i] * targets[j] * linearKernel(inputs[i], inputs[j])


def objectiveFunction(alpha):
    return (1/2) * np.dot(alpha, np.dot(alpha, P_MATRIX ) ) - np.sum(alpha)

def zerofun(alpha):
    return np.dot(alpha, targets)

def indicator(x, y):
    global ALPHA_NONZERO, targets_NONZERO, b, inputs_NONZERO
    ind = 0
    for i in range(np.size(ALPHA_NONZERO)):
        ind += ALPHA_NONZERO[i] * targets_NONZERO[i] * KERNEL((x,y), inputs_NONZERO[i])
    return ind - b

def main():
    global ALPHA_NONZERO, targets_NONZERO, b, inputs_NONZERO, sigma, poly_degree

    # CALL MINIMIZE:

    idx = 0
    C = None
    for sigma in [ 0.25, 0.5, 0.75, 1]:
        idx += 1
        start = np.zeros(N)
        B = [(0,C) for b in range(N)]
        XC = {'type':'eq', 'fun':zerofun}

        ret = minimize(objectiveFunction, start, bounds=B, constraints=XC)
        alpha = ret['x']

        # EXTRACT NON-ZERO ALPHA VALUES:
        index = np.where(alpha > 10**(-5))


        inputs_NONZERO = inputs[index]
        targets_NONZERO = targets[index]
        ALPHA_NONZERO = alpha[index]
        
        # CALCULATE THE b VALUE USING EQ. (7):
        b = 0
        for i in range(len(ALPHA_NONZERO)):
            b += ALPHA_NONZERO[i]* targets_NONZERO[i]*KERNEL(inputs_NONZERO[0],inputs_NONZERO[i])
        b = b - targets_NONZERO[0]
        

        # PLOTTING
        plt.subplot(2,2,idx)
        plt.plot([p[0] for p in classA],
                [p[1] for p in classA], 
                'b.')
        plt.plot([p[0] for p in classB],
                [p[1] for p in classB], 
                'r.')
        # plt.title('Polynomial degree = {}'.format(poly_degree))
        plt.title('sigma = {}'.format(sigma))
    
        plt.axis('equal')
        plt.savefig('svmplot.pdf')

        xgrid = np.linspace(-5, 5)
        ygrid = np.linspace(-4, 4)
        grid = np.array( [[indicator(x, y)
                            for x in xgrid]
                            for y in ygrid])
        plt.contour(xgrid,ygrid,grid, 
                    (-1.0, 0.0, 1.0),
                    colors = ('red','black', 'blue'),
                    linewidths = (1,3,1))
    
    plt.show()



if __name__ == "__main__":
    main()
