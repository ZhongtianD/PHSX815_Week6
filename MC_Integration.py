import sys
import numpy as np
from scipy.special import erf
sys.path.append(".")
from Random import Gaussian


# main function for our coin toss Python code
if __name__ == "__main__":

    # default number of samples (per experiment)
    N = 100
    a = -1
    b = 1

    if '-h' in sys.argv or '--help' in sys.argv:
        print ("Usage: %s [options]" % sys.argv[0])
        print ("  options:")
        print ("   --help(-h)          print options")
        print ("   -N number of random samples for integration")
        print ("   -a lower bound of integration")
        print ("   -b upper bound of integration")
        print
        sys.exit(1)

    if '-N' in sys.argv:
        p = sys.argv.index('-N')
        Ns = int(sys.argv[p+1])
        if Ns > 10:
            N = N
            
    if '-a' in sys.argv:
        p = sys.argv.index('-a')
        a0 = int(sys.argv[p+1])
        if a0 > -100:
            a = a0
    
    if '-b' in sys.argv:
        p = sys.argv.index('-b')
        b0 = int(sys.argv[p+1])
        if b0 > a:
            b = b0
            
    I_0 = (erf(b*np.sqrt(2)/2)-erf(a*np.sqrt(2)/2))/2
    G = Gaussian()
    def f(x):
        return G.Gaussian_pdf(x)
    Sample = np.random.uniform(low=a, high=b, size=N)
    F_val = f(Sample)
    I_M = (b-a)*np.mean(F_val)
    
    def trap(N):
        h = (b-a) / float(N)
        I = 0.5 * h * (f(a) + f(b))
        for i in range(1, int(N)):
            I += h * f(i * h+a)
        return I
    
    I_T = trap(N)
    x,w = np.polynomial.legendre.leggauss(N)
    x = (x+1)*(b-a)/2+a
    w *= (b-a)/2
    I_G = np.dot(f(x),w)
    
    
    print('With '+str(N)+' evaluation of the function' )
    print('True value of the integral is '+str(I_0) )
    print('Gauss–Legendre quadrature method value is '+str(I_G) )
    print('Trapezoidal integration value is '+str(I_T) )
    print('MC integration value is '+str(I_M) )
    print('Gauss–Legendre quadrature method error =  '+str(I_0 - I_G))
    print('Trapezoidal integration error =  '+str(I_0 - I_T))
    print('MC integration error =  '+str(I_0 - I_M))
    print('Gauss–Legendre quadrature method - MC integrationn =  '+str(I_G-I_M))
    print('MC integration - trapezoidal integration =  '+str(I_M-I_T))
    
    
    
    
    
    
    
