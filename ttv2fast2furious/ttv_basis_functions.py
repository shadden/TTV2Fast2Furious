"""Routines for computing the basis functions used in the TTV linear model."""

import numpy as np
from scipy.special import gammainc,gammaincinv
from scipy.special import ellipk,ellipkinc,ellipe,ellipeinc
import scipy.integrate as integrate
import os

##################################
########## dl terms #############
##################################
# Equations A21 and A22 
def ellipkOsc(x,m):
    """
    Compute the oscillating part of elliptic function ?th kind,  K

    Args:
        x (real): elliptic function arugment
        m (real): elliptic function modulus

    Returns:
        real
    """
    return ellipkinc(x , m) - 2*x*ellipk(m)/(np.pi)
def ellipeOsc(x,m):
    """
    Compute the oscillating part of elliptic function ?th kind, E

    Args:
        x (real): elliptic function arugment
        m (real): elliptic function modulus

    Returns:
        real
    """
    return ellipeinc(x , m) - 2*x*ellipe(m)/(np.pi)

# oscillating part of integral (1+a^2-2a*cos(psi))^(-1/2)
def DinvOsc(alpha,psi):
    denom = 1-alpha
    m = -4*alpha / denom / denom
    psi=np.mod(psi,2*np.pi)
    return 2 * ellipkOsc(0.5 * psi , m) / denom

# oscillating part of integral d/da[(1+a^2-2a*cos(psi))^(-1/2)]
def dDdalphaInvOsc(alpha,psi):
    m = -4*alpha / (1-alpha) / (1-alpha)
    X = (1-alpha) * ellipeOsc(0.5 * psi,m)
    X = X - (1+alpha)* ellipkOsc(0.5 * psi,m)
    X = X + 2* alpha * np.sin(psi) / np.sqrt(1+alpha*alpha-2*alpha*np.cos(psi))
    X = X / alpha / (1-alpha) / (1+alpha)
    return X

# Equation A19 and A20
def A(alpha,psi):
    indirect= -3 * np.sin(psi) / alpha
    direct = 3 * DinvOsc(alpha,psi)  / alpha / alpha
    return direct + indirect
def B(alpha,psi):
    indirect = -2 * np.sqrt(alpha) * np.sin(psi)
    direct = 2 * np.sqrt(alpha) * dDdalphaInvOsc(alpha,psi)
    return  direct + indirect

# Equation A29
def A1(alpha,psi):
    indirect = 3 * np.sin(psi) / alpha / alpha
    direct = -3 * DinvOsc(alpha,psi) 
    return direct + indirect

def B1(alpha,psi):
    direct = -2 * (DinvOsc(alpha,psi) + alpha*dDdalphaInvOsc(alpha,psi))
    indirect = -2 * np.sin(psi) / alpha / alpha
    return direct + indirect


# inner planet lambda
def dlFn(alpha,psi):
    dnInv = 1.0/(alpha**(-1.5)-1.0)
    return dnInv*dnInv*A(alpha,psi) + dnInv*B(alpha,psi)
# outer planet lambda
def dl1Fn(alpha,psi):
    dnInv = 1.0/(alpha**(-1.5)-1.0)
    return dnInv*dnInv*A1(alpha,psi) + dnInv * B1(alpha,psi)


##################################
########## dz terms #############
##################################
# R = R0(alpha,psi) + (e*exp[i(l-w)])*R1(alpha,psi) + ...
def R1_re(alpha,psi):
    denom = (1+alpha*alpha-2*alpha*np.cos(psi))**(1.5)
    direct =  0.5 * (alpha * alpha - alpha * np.cos(psi) )  / denom
    indirect = 0.5 * alpha * np.cos(psi)
    return direct + indirect
def R1_im(alpha,psi):
    denom = (1+alpha*alpha-2*alpha*np.cos(psi))**(1.5)
    direct =  -1 * alpha * np.sin(psi)  / denom
    indirect = alpha * np.sin(psi)
    return direct + indirect
    
# dz_re / dt = integrand_re
def integrand_re(nt,alpha,psi0,l0):
    psi = psi0 + nt * (alpha**(1.5) - 1.0 )
    l = nt + l0
    return -2 * alpha * (np.sin(l) * R1_re(alpha,psi) + np.cos(l) * R1_im(alpha,psi))

# dz_im / dt = integrand_re
def integrand_im(nt,alpha,psi0,l0):
    psi = psi0 + nt * (alpha**(1.5) - 1.0 )
    l = nt + l0
    return 2 * alpha * (np.cos(l) * R1_re(alpha,psi) - np.sin(l) * R1_im(alpha,psi))

def dzRe(alpha,t,t0=0,psi0=0,l0=0):
    I =  integrate.quad( integrand_re, t0 , t , args=(alpha,psi0,l0) )[0] 
    return I 
def dzIm(alpha,t,t0=0,psi0=0,l0=0):
    I = integrate.quad( integrand_im , t0 , t , args=(alpha,psi0,l0) )[0]
    return  I

# Rp = Rp0(alpha,psi) + (e1*exp[i(l1-w1)])*Rp1(alpha,psi) + ...
def Rp1_re(alpha,psi):
    denom = (1+alpha*alpha-2*alpha*np.cos(psi))**(1.5)

    direct =  0.5 * (1 - alpha * np.cos(psi) )  / denom
    
    indirect = 0.5 * np.cos(psi) / alpha / alpha 

    return direct + indirect

def Rp1_im(alpha,psi):
    denom = (1+alpha*alpha-2*alpha*np.cos(psi))**(1.5)
 
    direct = alpha * np.sin(psi)  / denom
    
    indirect =  -1 * np.sin(psi) / alpha / alpha
    
    return direct + indirect
    
# dz1_re / dt = integrand_re
def integrand1_re(n1t,alpha,psi0,l10):
    psi = psi0 + n1t * (1.0 - alpha**(-1.5) )
    l = n1t + l10
    return -2 * (np.sin(l) * Rp1_re(alpha,psi) + np.cos(l) * Rp1_im(alpha,psi))

# dz_im / dt = integrand_re
def integrand1_im(n1t,alpha,psi0,l10):
    psi = psi0 + n1t * (1.0 - alpha**(-1.5) )
    l = n1t + l10
    return 2 * (np.cos(l) * Rp1_re(alpha,psi) - np.sin(l) * Rp1_im(alpha,psi))

def dz1Re(alpha,n1t,n1t0=0,psi0=0,l10=0):
    I =  integrate.quad( integrand1_re, n1t0 , n1t , args=(alpha,psi0,l10) )[0] 
    return I 
def dz1Im(alpha,n1t,n1t0=0,psi0=0,l10=0):
    I = integrate.quad( integrand1_im , n1t0 , n1t , args=(alpha,psi0,l10) )[0]
    return  I

########################################
########## ttv basis functions #########
########################################
def getTimesOfTransit(P,l0,N):
    return np.arange(1,N+1)*P - np.mod(l0,2*np.pi) * P / (2*np.pi)
def dt0_InnerPlanet(P,P1,T0,T10,Ntrans):
    """
    Compute the 0th order (in eccentricity) TTV basis function for an 
    inner planet with exterior perturber.

    Arguments
    ---------
        P : real
            The period of the planet
        P1 : real
            The period of the perturber
        T0 :  real
            Planet's time of initial transit
        T1 : real
            Perturbers time of initial transit
        Ntrans : int
            Number of transits to compute

    Returns
    -------

        ndarray :
            The basis function value at sequential transit times
    """

    # compute mean longitudes and synodic angle
    l = 2*np.pi * -T0 / P
    l1= 2*np.pi * -T10 / P1

    psi0=np.mod(l1-l ,2*np.pi)
    
    # get the unperturbed transit times
    TransitTimes0=getTimesOfTransit(P,l,Ntrans)
    TransitPhases0=2*np.pi * TransitTimes0 / P

    # get psi at the unperturbed transit times (i.e., when l=0)
    psi = np.mod(l1 + 2 * np.pi * TransitTimes0 / P1,2*np.pi)

    #get semi-major axis ratio
    alpha = np.power(P/P1,2./3.)
    
    # get list of delta-lambdas
    dl = dlFn(alpha,psi) 
    
    # get list of delta-z
    dzReArr = np.zeros(Ntrans)
    dzImArr = np.zeros(Ntrans)
    zlastRe=0.
    zlastIm=0.
    
    for i,t in enumerate(TransitPhases0):
        if i>0:
            zlastRe=zlastRe + dzRe(alpha,t,t0=TransitPhases0[i-1],psi0=psi0,l0=l)
            zlastIm=zlastIm + dzIm(alpha,t,t0=TransitPhases0[i-1],psi0=psi0,l0=l)
        else:
            zlastRe =  dzRe(alpha,t,psi0=psi0,l0=l)
            zlastIm =  dzIm(alpha,t,psi0=psi0,l0=l)
        dzReArr[i] = zlastRe
        dzImArr[i] = zlastIm
    dl = dl-np.mean(dl)
    dzImArr = dzImArr-np.mean(dzImArr)
    dzReArr = dzReArr-np.mean(dzReArr)
    dz = dzImArr
    ttv0 = -1 * P *  (dl - 2 * dz) / (2*np.pi)
    return ttv0
########################################
def dt0_OuterPlanet(P,P1,T0,T10,Ntrans):

    # compute mean longitudes and synodic angle
    l = 2*np.pi * -T0 / P
    l1= 2*np.pi * -T10 / P1

    psi0=np.mod(l1-l ,2*np.pi)
    
    # get the unperturbed transit times
    TransitTimes0=getTimesOfTransit(P1,l1,Ntrans)
    TransitPhases0=2*np.pi * TransitTimes0 / P1

    # get psi at the unperturbed transit times (i.e., when l1=0)
    psi = np.mod(l + 2 * np.pi * TransitTimes0 / P,2*np.pi)

    #get semi-major axis ratio
    alpha = np.power(P/P1,2./3.)
    
    # get list of delta-lambdas
    dl1A = dl1Fn(alpha,psi)

    
    # get list of delta-z
    dz1ReArr = np.zeros(Ntrans)
    dz1ImArr = np.zeros(Ntrans)
    z1lastRe=0.
    z1lastIm=0.
    
    for i,n1t in enumerate(TransitPhases0):
        if i>0:
            z1lastRe=z1lastRe + dz1Re(alpha,n1t,n1t0=TransitPhases0[i-1],psi0=psi0,l10=l1)
            z1lastIm=z1lastIm + dz1Im(alpha,n1t,n1t0=TransitPhases0[i-1],psi0=psi0,l10=l1)
        else:
            z1lastRe =  dz1Re(alpha,n1t,psi0=psi0,l10=l)
            z1lastIm =  dz1Im(alpha,n1t,psi0=psi0,l10=l)
        dz1ReArr[i] = z1lastRe
        dz1ImArr[i] = z1lastIm
    dl1A = dl1A-np.mean(dl1A)
    dz1ImArr = dz1ImArr-np.mean(dz1ImArr)
    dz1ReArr = dz1ReArr-np.mean(dz1ReArr)
    dz1 = dz1ImArr
    ttv = -1 * P1 *  (-dl1A - 2 * dz1) / (2*np.pi)
    return ttv
########################################

def get_nearest_firstorder(periodratio):
    return int(np.round((1-periodratio)**(-1)))
def get_superperiod(P,P1):
    j=get_nearest_firstorder(P/P1)
    SuperP = (j/P1 - (j-1)/P)**(-1)
    return SuperP

###########################################################
########## sympy for laplace coefficients #################
##########################################################
on_rtd = os.environ.get("READTHEDOCS", None) == "True"
if not on_rtd:
    from sympy import hyper,hyperexpand,S,binomial,diff,Subs,N
    _j = S("j")
    _alpha = S("alpha")
    _laplaceB= 2 * (-1)**_j *_alpha**_j * binomial(-1/S(2),_j)*hyper([1/S(2),_j+1/S(2)],[1+_j],_alpha*_alpha)
    _aDlaplaceB = _alpha * diff(_laplaceB,_alpha)
    _a2D2laplaceB = _alpha * _alpha * diff(_laplaceB,_alpha,_alpha)
    fCoeffInExprn = -_j * _laplaceB - (1/S(2)) * _aDlaplaceB
    fCoeffOutExprn = (_j + 1/S(2)) * _laplaceB + (1/S(2)) * _aDlaplaceB
    f49Coeff = -1 * (2 * _j * ( 1 + 2 *_j) * _laplaceB + (2 + 4 * _j) * _aDlaplaceB +  _a2D2laplaceB) / 4

def get_fCoeffs(j,alpha):
    fCoeffIn = N(Subs(fCoeffInExprn,[_j,_alpha],[j,alpha])) 
    fCoeffOut = N(Subs(fCoeffOutExprn,[_j,_alpha],[j-1,alpha])) 
    # Add indirect term
    if j==2:
     # Evaluated exactly on resonance, indirect contribution to 
     # fCoeffOut is the same for inner and outer planet. It is 
     # different for different values of aIn/aOut but I'm ignoring
     # this for now.
        fCoeffOut = fCoeffOut - 2.0**(1./3.)
    return float(fCoeffIn),float(fCoeffOut)

########################################
########### 2nd order res ##############
########################################
def get_gamma(k,alpha):
    f49 =  N(Subs(f49Coeff, [_j,_alpha] , [k-1,alpha]))
    j = int(np.ceil(k/2))
    f27,f31 = get_fCoeffs(j,alpha)
    # Eq. 58, Hadden & Lithwick '16
    gamma = f49 * (f27*f27 + f31*f31) / f27 / f31 / 2
    return gamma
def get_nearest_secondorder(Pin_by_Pout):
    return int(np.round(2 / (1 - Pin_by_Pout))) 
def get_second_order_superperiod(Pin,Pout):
    k = get_nearest_secondorder(Pin/Pout) 
    nIn = 2*np.pi / Pin
    nOut= 2*np.pi / Pout
    nRes = k*nOut - (k-2)*nIn
    return 2*np.pi / nRes
def dt2_InnerPlanet(P,P1,T0,T10,Ntrans):
    k=get_nearest_secondorder(P/P1)
    superP = get_second_order_superperiod(P,P1)
    superPhase = 2*np.pi * (k * (-T10/P1) - (k-2) * (-T0/P) )
    Times = T0 + np.arange(Ntrans) * P

    alpha = (P/P1)**(2./3.)
    alpha_2 = 1.0 / alpha / alpha
    gamma = get_gamma(k,alpha)
    n_K_2minusK = (1 / P1) / (k /P1 + (2-k) / P )

    # Eq. 60 of HL16
    S =  1.5 * (2-k) * alpha_2 * n_K_2minusK**2 * gamma * P / (np.pi)
    C = -1.5 * (2-k) * alpha_2 * n_K_2minusK**2 * gamma * P / (np.pi)
    superSin = S * np.sin(2*np.pi * Times / superP + superPhase)
    superCos = C * np.cos(2*np.pi * Times / superP + superPhase)
    basis_function_matrix=np.vstack((superSin,superCos)).T
    return basis_function_matrix

def dt2_OuterPlanet(P,P1,T0,T10,Ntrans):
    k=get_nearest_secondorder(P/P1)
    superP = get_second_order_superperiod(P,P1)
    superPhase = 2*np.pi * (k * (-T10/P1) - (k-2) * (-T0/P) )
    Times = T10 + np.arange(Ntrans) * P1
    alpha = (P/P1)**(2./3.)
    gamma = get_gamma(k,alpha)
    n_K_2minusK = (1 / P1) / (k /P1 + (2-k) / P )
    # Eq. 60 of HL16
    S =  1.5 * (k) * n_K_2minusK**2 * gamma * P1 / (np.pi)
    C = -1.5 * (k) * n_K_2minusK**2 * gamma * P1 / (np.pi)
    superSin = S * np.sin(2*np.pi * Times / superP + superPhase)
    superCos = C * np.cos(2*np.pi * Times / superP + superPhase)
    basis_function_matrix=np.vstack((superSin,superCos)).T
    return basis_function_matrix
########################################
