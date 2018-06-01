import numpy as np
from scipy.special import ellipk,ellipkinc,ellipe,ellipeinc
import scipy.integrate as integrate

##################################
########## dl terms #############
##################################
def ellipkOsc(x,m):
    return ellipkinc(x , m) - 2*x*ellipk(m)/(np.pi)
def ellipeOsc(x,m):
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

def A(alpha,psi):
    indirect= -3 * np.sin(psi) / alpha
    direct = 3 * DinvOsc(alpha,psi)  / alpha / alpha
    return direct + indirect
def B(alpha,psi):
    indirect = -2 * np.sqrt(alpha) * np.sin(psi)
    direct = 2 * np.sqrt(alpha) * dDdalphaInvOsc(alpha,psi)
    return  direct + indirect

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
    dl = dlFn(alpha,psi) #np.array([l for l in map( lambda x: dlFn(alpha,x), psi)])
    
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
    # dl1=np.array([l for l in map( lambda x: dl1Fn(alpha,(alpha**1.5-1)*x + psi0), times)])
    
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
from sympy import hyper,hyperexpand,S,binomial,diff,Subs,N
_j = S("j")
_alpha = S("alpha")
_laplaceB= 2 * (-1)**_j *_alpha**_j * binomial(-1/S(2),_j)*hyper([1/S(2),_j+1/S(2)],[1+_j],_alpha*_alpha)
_aDlaplaceB = _alpha * diff(_laplaceB,_alpha)
fCoeffInExprn = -_j * _laplaceB - (1/S(2)) * _aDlaplaceB
fCoeffOutExprn = (_j + 1/S(2)) * _laplaceB + (1/S(2)) * _aDlaplaceB
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
def ttv_basis_function_matrix_inner(P,P1,T0,T10,Ntrans,IncludeLinearBasis=True):
    j = get_nearest_firstorder(P/P1)
    superP = get_superperiod(P,P1)
    superPhase = 2*np.pi * (j * (-T10/P1) - (j-1) * (-T0/P) )
    Times = T0 + np.arange(Ntrans) * P

    # Normalize super-sin/cos so that basis fn amplitude is mu * (Zx/Zy)
    #
    j_2 = 1.0 / j / j
    Delta = (j-1) * P1 / P / j - 1
    Delta_2 = 1.0 / Delta / Delta
    alpha = (P/P1)**(2./3.)
    alpha_2 = 1.0 / alpha / alpha
    fCoeffIn,fCoeffOut=get_fCoeffs(j,alpha)
    denom = (fCoeffIn*fCoeffIn + fCoeffOut*fCoeffOut)**(0.5)
    S =  1.5 * (1-j) * alpha_2 * j_2 * Delta_2 * denom * P / (np.pi)
    C = -1.5 * (1-j) * alpha_2 * j_2 * Delta_2 * denom * P / (np.pi)

    dt0 = dt0_InnerPlanet(P,P1,T0,T10,Ntrans)
    superSin = S * np.sin(2*np.pi * Times / superP + superPhase)
    superCos = C * np.cos(2*np.pi * Times / superP + superPhase)
    
    if IncludeLinearBasis:
        basis_function_matrix=np.vstack((np.ones(Ntrans),np.arange(Ntrans),dt0,superSin,superCos)).T
    else:
        basis_function_matrix=np.vstack((dt0,superSin,superCos)).T

    return basis_function_matrix

def ttv_basis_function_matrix_outer(P,P1,T0,T10,Ntrans,IncludeLinearBasis=True):
    j = get_nearest_firstorder(P/P1)
    superP = get_superperiod(P,P1)
    superPhase = 2*np.pi * (j * (-T10/P1) - (j-1) * (-T0/P) )
    Times = T10 + np.arange(Ntrans) * P1

    # Normalize super-sin/cos so that basis fn amplitude is mu * (Zx/Zy)
    #
    j_2 = 1.0 / j / j
    Delta = (j-1) * P1 / P / j - 1
    Delta_2 = 1.0 / Delta / Delta
    alpha = (P/P1)**(2./3.)
    alpha_2 = 1.0 / alpha / alpha
    fCoeffIn,fCoeffOut=get_fCoeffs(j,alpha)
    denom = (fCoeffIn*fCoeffIn + fCoeffOut*fCoeffOut)**(0.5)
    S1 =  1.5 * (j) * j_2 * Delta_2 * denom  * P1 / (np.pi)
    C1 = -1.5 * (j) * j_2 * Delta_2 * denom  * P1 / (np.pi)

    dt0 = dt0_OuterPlanet(P,P1,T0,T10,Ntrans)
    superSin = S1 * np.sin(2*np.pi * Times / superP + superPhase)
    superCos = C1 * np.cos(2*np.pi * Times / superP + superPhase)
    
    if IncludeLinearBasis:
        basis_function_matrix=np.vstack((np.ones(Ntrans),np.arange(Ntrans),dt0,superSin,superCos)).T
    else:
        basis_function_matrix=np.vstack((dt0,superSin,superCos)).T
    return basis_function_matrix
    
###################################
def PlanetPropertiestoLinearModelAmplitudes(T0,P,mass,e,varpi,T10,P1,mass1,e1,varpi1):
    #assert(P<P1): print("Inner planet period is larger than outer planet's!")
    j=get_nearest_firstorder(P/P1)
    j_2 = 1.0 / j / j
    Delta = (j-1) * P1 / P / j - 1
    Delta_2 = 1.0 / Delta / Delta

    ex = e * np.cos(varpi)
    ey = e * np.sin(varpi)
    ex1 = e1 * np.cos(varpi1)
    ey1 = e1 * np.sin(varpi1)

    alpha = (P/P1)**(2./3.)
    alpha_2 = 1.0 / alpha / alpha
    fCoeffIn,fCoeffOut=get_fCoeffs(j,alpha)
    denom = (fCoeffIn*fCoeffIn + fCoeffOut*fCoeffOut)**(0.5)
    Zx = fCoeffIn * ex + fCoeffOut * ex1
    Zy = fCoeffIn * ey + fCoeffOut * ey1
    Zx /= denom
    Zy /= denom

    #S = mass1 * 1.5 * (1-j) * alpha_2 * j_2 * Delta_2 * denom * Zx * P / (np.pi)
    #C = -mass1 * 1.5 * (1-j) * alpha_2 * j_2 * Delta_2 * denom * Zy * P / (np.pi)
    #
    #S1 = mass * 1.5 * (j) * j_2 * Delta_2 * denom * Zx * P1 / (np.pi)
    #C1 = -mass * 1.5 * (j) * j_2 * Delta_2 * denom * Zy * P1 / (np.pi)
    S = mass1 * Zx
    C = mass1 * Zy
    S1 = mass * Zx
    C1 = mass * Zy
    return [T0,P,mass1,S,C],[T10,P1,mass,S1,C1]
###################################
def get_ttv_basis_function_matrix(Pi,Pj,T0i,T0j,Ntrans):
    if Pi < Pj:
        return ttv_basis_function_matrix_inner(Pi,Pj,T0i,T0j,Ntrans,IncludeLinearBasis=False)
    else:
        return ttv_basis_function_matrix_outer(Pj,Pi,T0j,T0i,Ntrans,IncludeLinearBasis=False)
def get_linear_basis_basis_function_matrix(Ntransits):
    return np.vstack((np.ones(Ntransits),np.arange(Ntransits))).T
###################################
def MultiplanetSystemBasisFunctionMatrices(Nplanets,Periods,T0s,Ntransits,**kwargs):
    #assert(len(Periods)==Nplanets): "Improper period array size"
    #assert(len(T0s)==Nplanets): "Improper T0 array size"
    #assert(len(Ntransits)==Nplanets): "Improper Ntransits array size"
    InteractionMatrix=kwargs.get("InteractionMatrix",np.ones((Nplanets,Nplanets),dtype=bool))
    for i in range(Nplanets):
        # No self-interactions!
        InteractionMatrix[i,i]=False

    BasisFunctionMatrices=[get_linear_basis_basis_function_matrix(Nt) for Nt in Ntransits]
    for i in range(Nplanets):
        Pi = Periods[i]
        T0i = T0s[i]
        Ntransi = Ntransits[i]
        for j in range(Nplanets):
            if InteractionMatrix[i,j]:
                Pj = Periods[j]
                T0j = T0s[j]
                A = get_ttv_basis_function_matrix(Pi,Pj,T0i,T0j,Ntransi)
                BasisFunctionMatrices[i] = np.hstack((BasisFunctionMatrices[i],A))
            else:
                continue
    return BasisFunctionMatrices
###################################
def SetupInteractionMatrixWithMaxPeriodRatio(Periods, MaxRatio):
    Np=len(Periods)
    entries = []
    for P1 in Periods:
        for P2 in Periods:
            entries.append(np.max((P1/P2,P2/P1)) < MaxRatio)
    entries=np.array(entries).reshape((Np,Np))
    for i in range(Np):
        entries[i,i]=False
    return entries
###################################
def get_ttv_model_amplitudes(Pi,Pj,T0i,T0j,massi,massj,ei,ej,pmgi,pmgj):
    if Pi<Pj:
        Xi,Xj = PlanetPropertiestoLinearModelAmplitudes(T0i,Pi,massi,ei,pmgi,T0j,Pj,massj,ej,pmgj)
    else:
        Xj,Xi = PlanetPropertiestoLinearModelAmplitudes(T0j,Pj,massj,ej,pmgj,T0i,Pi,massi,ei,pmgi)
    return Xi[2:],Xj[2:]
###################################
def MultiplanetSystemLinearModelAmplitudes(Nplanets,Periods,T0s,masses,eccs,pomegas,**kwargs):
    InteractionMatrix=kwargs.get("InteractionMatrix",np.ones((Nplanets,Nplanets),dtype=bool))
    for i in range(Nplanets):
        # No self-interactions!
        InteractionMatrix[i,i]=False

    Xs=[]
    for i in range(Nplanets):
        lenXi = 2 + 3 * np.sum(InteractionMatrix[i])
        Xi = np.zeros(lenXi)
        Xi[0] = T0s[i]
        Xi[1] = Periods[i]
        Xs.append(Xi)
    
    for i in range(Nplanets):
        Pi = Periods[i]
        T0i = T0s[i]
        massi=masses[i]
        ei=eccs[i]
        pmgi=pomegas[i]
        for j in range(i+1,Nplanets):
            if InteractionMatrix[i,j] or InteractionMatrix[j,i]:
                Pj = Periods[j]
                T0j = T0s[j]
                massj = masses[j]
                ej = eccs[j]
                pmgj=pomegas[j]

                Xi,Xj = get_ttv_model_amplitudes(Pi,Pj,T0i,T0j,massi,massj,ei,ej,pmgi,pmgj)

                if InteractionMatrix[i,j]:
                    Jindex=2+3*np.sum(InteractionMatrix[i,:j])
                    Xs[i][Jindex:Jindex+3]=Xi
                if InteractionMatrix[j,i]:
                    Iindex=2+3*np.sum(InteractionMatrix[j,:i])
                    Xs[j][Iindex:Iindex+3]=Xj
            else:
                pass
    return Xs
###################################
ass PlanetTransitObservations(object):
    def __init__(self,transit_numbers,times,uncertainties):
        self.transit_numbers=transit_numbers
        self.times=times
        self.uncertainties = uncertainties
    @classmethod
    def from_times_only(cls,times,unc=0):
        Ntr = len(times)
        nums = np.arange(Ntr)
        sigma = unc * np.ones(Ntr)
        return cls(nums,times,sigma)
    @property
    def Ntransits(self):
        return len(self.times)
    
    @property
    def weighted_obs_vector(self):
        return self.times / self.uncertainties
    
    def basis_function_matrix(self):
        constant_basis = np.ones(self.Ntransits)
        return np.vstack(( constant_basis , self.transit_numbers)).T
    def linear_fit_design_matrix(self):
        constant_basis = np.ones(self.Ntransits)
        sigma = self.uncertainties
        design_matrix = np.vstack(( constant_basis / sigma , self.transit_numbers / sigma )).T
        return design_matrix
    
    def linear_best_fit(self):
        sigma = self.uncertainties
        y = self.weighted_obs_vector
        A = self.linear_fit_design_matrix()
        fitresults = np.linalg.lstsq(A,y)
        return fitresults[0]
    def linear_fit_residuals(self):
        M = self.basis_function_matrix()
        best = self.linear_best_fit()
        return self.times - M.dot(best)

    


class TransitTimesLinearModels(object):

    def __init__(self,observations_list):
        self.observations=observations_list
        initial_linear_fit_data = np.array([obs.linear_best_fit() for obs in self.observations ])
        self.T0s = initial_linear_fit_data[:,0]
        self.periods = initial_linear_fit_data[:,1]
        self.basis_function_matrices = [obs.basis_function_matrix() for obs in self.observations ]
        self._maximum_interaction_period_ratio = np.infty
        
    @classmethod
    def from_multi_tranet(cls,multi_system,ttv_interface,short_cadence=True):
        times_list = get_nominal_transit_times(multi_system,ttv_interface)
        obs_list=[]
        planets = multi_system.detected_planets
        for i,pl in enumerate(planets):
            times = times_list[i]
            if short_cadence:
                unc = pl.midtime_unc_2min
            else:
                unc = pl.midtime_unc_30min
            obs_list.append( PlanetTransitObservations.from_times_only(times, unc) )
        return cls(obs_list)    
    @property
    def maximum_interaction_period_ratio(self):
        return self._maximum_interaction_period_ratio
    @maximum_interaction_period_ratio.setter
    def maximum_interaction_period_ratio(self,value):
        self._maximum_interaction_period_ratio =  value
        
    @property
    def N(self):
        return len(self.observations)
    @property
    def weighted_obs_vectors(self):
        return [obs.weighted_obs_vector for obs in self.observations]
    @property
    def design_matrices(self):
        bfs = self.basis_function_matrices
        uncs= [o.uncertainties for o in self.observations ]
        return [np.transpose(bfs[i].T/uncs[i]) for i in range(self.N)]
    
    @property
    def best_fits(self):
        A = self.design_matrices
        y = self.weighted_obs_vectors
        return [np.linalg.lstsq(A[i],y[i])[0] for i in range(self.N)]
        
    def quicklook_plot(self,axis):
        for obs in self.observations:
            resid_MIN = 24*60*(obs.linear_fit_residuals())
            unc_MIN = 24*60*obs.uncertainties
            axis.errorbar(obs.times,resid_MIN,yerr=unc_MIN)
        axis.set_xlabel("Time [d.]")
        axis.set_ylabel("TTV [min.]")
    
    def generate_new_basis_function_matrices(self):
        MaximumPeriodRatio = self.maximum_interaction_period_ratio
        i_matrix=SetupInteractionMatrixWithMaxPeriodRatio(self.periods,MaximumPeriodRatio)
        maxTransitNumbers = [np.max(o.transit_numbers)+1 for o in self.observations]
        bf_matrices_full = MultiplanetSystemBasisFunctionMatrices(\
                            self.N,self.periods,self.T0s,maxTransitNumbers,InteractionMatrix=i_matrix)
        return [bf_matrices_full[i][(self.observations[i].transit_numbers)] for i in range(self.N)]
    
    def update_fits(self):
        self.basis_function_matrices = self.generate_new_basis_function_matrices()
        self.periods = [fit[1] for fit in self.best_fits]
