import sys
import numpy as np
sys.path.append(".")
from Random import Gaussian


# main function for our coin toss Python code
if __name__ == "__main__":

    # default number of samples (per experiment)
    N = 10

    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [options]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print ("   -N number of intervals for integration")
        print
        sys.exit(1)

    if '-N' in sys.argv:
        p = sys.argv.index('-N')
        Ns = int(sys.argv[p+1])
        if Ns > 1:
            N = N
            
    I_0 = 0.6826894921370859
    G = Gaussian()
    def f(x):
        return G.Gaussian_pdf(x)
    def trap(N):
        h = 2 / float(N)
        I = 0.5 * h * (f(-1) + f(1))
        for i in range(1, int(N)):
            I += h * f(i * h-1)
        return I
    
    I_T = trap(N)
    x,w = np.polynomial.legendre.leggauss(10)
    I_G = np.dot(f(x),w)
    
    
    print('With '+str(N)+' evaluation of the function' )
    print('Gauss–Legendre quadrature method - trapezoidal integration =  '+str(I_G-I_T))
    print('Gauss–Legendre quadrature method error =  '+str(I_0 - I_G))
    print('Trapezoidal integration error =  '+str(I_0 - I_T))
    
    
    
    
    
    
    
