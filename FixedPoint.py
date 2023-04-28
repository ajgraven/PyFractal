
import numpy as np





def f_root(f,z0,N=50,tol=1e-8):
    '''
    complex function root finder, via the
    variable-step secant method
    f(z) = input function
    z0   = initial guess
    N    = maximum # of iterations
    tol  = default error tolerance
    '''
    h = 2*tol # step for first derivative approximation
    try:
        z1 = z0-(1/2)*f(z0)/((f(z0+h)-f(z0))/h) # get second iterate via truncated Newton's
        SM = lambda z_0,z_1,a : z_1-a*f(z_1)*(z_1-z_0)/(f(z_1)-f(z_0)) # secant method iteration
        for i in range(N):
            a = 1
            z_step = SM(z0,z1,a)
            while abs(f(z_step)) > abs(f(z1)):
                if a < np.sqrt(tol):
                    break
                a = a/2
                z_step = SM(z0,z1,a)
            z0 = z1
            z1 = z_step
            if abs(f(z1))<tol:
                return z1
    except (RuntimeWarning):
        pass
    return None

def fixed_point(f,z0,N=50,tol=1e-8):
    '''
    complex function fixed point finder, via the
    variable-step secant method
    f(z) = input function
    z0   = initial guess
    N    = maximum # of iterations
    tol  = default error tolerance
    '''
    return f_root(lambda z:f(z)-z,z0,N,tol)

def fn(f, z, n):
    ''' returns the nth iterate of f starting at z''' 
    for i in range(n):
        z = f(z)
    return z

def param_fn(f, z0, n):
    ''' returns the nth iterate of f starting at z'''
    z=z0
    for i in range(n):
        z = f(z,z0)
    return z

def per_point(f,z0,per=1,N=50,tol=1e-8):
    '''
    complex function periodic point finder,
    via the variable-step secant method
    f(z) = input function
    z0   = initial guess
    per  = period of point
    N    = maximum # of iterations
    tol  = default error tolerance
    '''
    return fixed_point(lambda z : fn(f,z,per),z0,N,tol)

def param_per_point(f,z0,per=1,N=50,tol=1e-8):
    '''
    complex function periodic point finder,
    via the variable-step secant method
    f(z) = input function
    z0   = initial guess
    per  = period of point
    N    = maximum # of iterations
    tol  = default error tolerance
    '''
    return fixed_point(lambda z : param_fn(f,z,per),z0,N,tol)


        


