"""
.. module:: ttv2fast2furious
   :platform: Unix, Mac, Windows
   :synopsis: Routines for TTV analysis

.. moduleauthor:: Sam Hadden <samuel.hadden@cfa.harvard.edu>

"""
import warnings
import numpy as np
from scipy.optimize import nnls
from scipy.special import gammainc,gammaincinv
from scipy.special import ellipk,ellipkinc,ellipe,ellipeinc
import scipy.integrate as integrate
from ttv2fast2furious.ttv_basis_functions import get_nearest_firstorder,get_superperiod,dt0_InnerPlanet,dt0_OuterPlanet,get_fCoeffs
from ttv2fast2furious.ttv_basis_functions import get_nearest_secondorder,dt2_InnerPlanet,dt2_OuterPlanet

def ttv_basis_function_matrix_inner(P,P1,T0,T10,Ntrans,IncludeLinearBasis=True):
    """
    Compute the transit time basis function matrix, 'M', for a planet subject to an external perturber.

    Args:
        P (real): Inner planet period.
        P1 (real): Inner planet period.
        T (real): Outer planet initial time of transit
        T1 (real): Outer planet initial time of transit
        Ntrans (int):  number of transits 
        IncludeLinearBasis (bool): Whether basis functions representing a linear transit emphemris is included. Default is True.
           
    Returns:
        Array: the resulting basis function matrix is returned an (Ntrans, 5) array or an (Ntrans, 3) array if IncludeLinearBasis=False.
    """
    j = get_nearest_firstorder(P/P1)
    assert j > 0 , "Bad period ratio!!! P,P1 = %.3f \t %.3f"%(P,P1)
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
    """
    Compute the transit time basis function matrix, 'M', for a planet subject to an interior perturber.

    Args:
        P (real): Inner planet period.
        P1 (real): Inner planet period.
        T (real): Outer planet initial time of transit
        T1 (real): Outer planet initial time of transit
        Ntrans (int):  number of transits 
        IncludeLinearBasis (bool): Whether basis functions representing a linear transit emphemris is included. Default is True.
           
    Returns:
        Array: the resulting basis function matrix is returned an (Ntrans, 5) array or an (Ntrans, 3) array if IncludeLinearBasis=False.

    """
    j = get_nearest_firstorder(P/P1)
    assert j > 0 , "Bad period ratio!!! P,P1 = %.3f \t %.3f"%(P,P1)
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
    """
    """

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

    S = mass1 * Zx
    C = mass1 * Zy
    S1 = mass * Zx
    C1 = mass * Zy
    return np.array([T0,P,mass1,S,C]),np.array([T10,P1,mass,S1,C1])
###################################
def get_ttv_basis_function_matrix(Pi,Pj,T0i,T0j,Ntrans):
    """
    Compute basis function matrix for planet `i`'s TTV due to planet `j`. 
    
    Arguments
    --------
    Pi: real
        TTV planet's period
    Pj: real
        Perturbing planet's period
    T0i: real
        TTV planet's intial time of transit
    """
    if Pi < Pj:
        return ttv_basis_function_matrix_inner(Pi,Pj,T0i,T0j,Ntrans,IncludeLinearBasis=False)
    else:
        return ttv_basis_function_matrix_outer(Pj,Pi,T0j,T0i,Ntrans,IncludeLinearBasis=False)
def get_linear_basis_basis_function_matrix(Ntransits):
    return np.vstack((np.ones(Ntransits),np.arange(Ntransits))).T
###################################
def MultiplanetSystemBasisFunctionMatrices(Nplanets,Periods,T0s,Ntransits,**kwargs):
    """
    Compute basis function matrices for the transit times of an `Nplanet` system.

    Parameters
    ----------
    Nplanets : int
        The number of transting planets to model.
    Periods  : ndarray
        Array listing planets' orbital periods.
    T0s      : ndarray
        Array listing the times of planets' first transits
    Ntransits: ndarray
        Array listing the number of transits to compute for each planet.

    Keyword Arguments
    -----------------
    ``InteractionMatrix``
        Specify the interactions between planets as a matrix. 
        By default, all planets are assumed to interact with one
        antoher.

    Returns
    -------
    list
        List of ndarrays with columns containing TTV basis functions of each planet.
    """
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
    """
    Compute amplitudes for linear model basis functions for user-specified planet parameters.

    Parameters
    ----------

    Nplanets    :   int
        The number of planets in the system.

    Periods     :   ndarray
        Periods of the planets.

    T0s         :   ndarray
        Times of first transit

    masses      :   ndarray
        Planet masses, in units of host-star mass.

    eccs        :   ndarray
        Eccentricities.

    pomegas     :   ndarray
        Longitudes of periapse.

    Returns
    -------

    :obj:`list` of ndarrays
        List of model amplitudes 
    """
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
class PlanetTransitObservations(object):
    """
    Object to store transit timing measurement information.

    """
    def __init__(self,transit_numbers,times,uncertainties):
        """
        Parameters
        ----------

        transit_numbers : ndarray
            List that numbers the series of transit observations

        times           : ndarray
            List of transit mid-times

        uncertainties   : ndarray
            List of tramsit mid-time measurement :math:`1\sigma` uncertainties

        """
        self._transit_numbers=transit_numbers
        self._times=times
        self._uncertainties = uncertainties
        self._Ntransits = len(times)
        self._mask = np.ones(self._Ntransits,dtype=bool)

    @classmethod
    def from_times_only(cls,times,unc=0):
        """
        Initialize transit observation object diectly from list of transit times.

        Return a :class:`PlanetTransitObservations` object initialized from a list of transit times.
        Transit numbers are automatically assigned sequentially. Uniform uncertainties are
        assigned to the observations according to the user-specified `unc`.
        
        Parameters
        ----------

        times : array_like
            List of transit mid-times
        unc : float, optional
            Uncertainty assigned to transit observations

        """
        Ntr = len(times)
        nums = np.arange(Ntr)
        sigma = unc * np.ones(Ntr)
        return cls(nums,times,sigma)

    def function_mask(self,func):
        """
        Mask out transit times by applying function `func` to the times.

        Parameters
        ----------

        func : callable
            Function to mask times. `func` should return `False` for times that are
            to be masked out. Otherwise `func` should return `True`.
        """
        self._mask = np.array(list(map(func,self._times)))

    def unmask(self):
        self._mask = np.ones(self._Ntransits,dtype=bool)

    @property
    def times(self):
        """
        ndarray: List of transit mid-times
        """
        return self._times[self._mask]    
    @property
    def transit_numbers(self):
        """
        ndarray: List of transit epoch numbers.
        """
        return self._transit_numbers[self._mask]    
    @property
    def uncertainties(self):
        """
        ndarray: List of :math:`1\sigma` transit mid-time measurement uncertainties.
        """
        return self._uncertainties[self._mask]    

    @uncertainties.setter
    def uncertainties(self,value):
        self._uncertainties[self._mask] = value   

    @property
    def Ntransits(self):
        return len(self.times)
    
    @property
    def weighted_obs_vector(self):
        return self.times / self.uncertainties
    
    def basis_function_matrix(self):
        """ 
        Generate the basis function matrix for a linear transit ephemeris.

        Returns
        -------
        ndarray
            basis function matrix
        """
        constant_basis = np.ones(self.Ntransits)
        return np.vstack(( constant_basis , self.transit_numbers)).T
    def linear_fit_design_matrix(self):
        """ 
        Generate the design matrix for a linear transit ephemeris.
        """
        constant_basis = np.ones(self.Ntransits)
        sigma = self.uncertainties
        design_matrix = np.vstack(( constant_basis / sigma , self.transit_numbers / sigma )).T
        return design_matrix
    
    def linear_best_fit(self):
        """ 
        Determine the best-fit period and initial transit time for a linear transit ephemeris.
        """
        sigma = self.uncertainties
        y = self.weighted_obs_vector
        A = self.linear_fit_design_matrix()
        fitresults = np.linalg.lstsq(A,y,rcond=-1)
        return fitresults[0]
    def linear_fit_residuals(self):
        """
        Return the transit timing residuals of a best-fit linear transit ephemeris.
        """
        M = self.basis_function_matrix()
        best = self.linear_best_fit()
        return self.times - M.dot(best)

    


class TransitTimesLinearModels(object):
    """
    Object representing a collection of transit time linear models in a system of interacting planets.

    Attributes
    ----------
    observations : list
        A list of transit time observation objects representing the transit observations for a system
    
    basis_function_matrices: list
        List of each planet's transit itme basis function matrix
    
    periods : ndarray
        Best-fit periods of all planets

    T0s : ndarray
        Best-fit initial times of transit for all planets

    best_fit : :obj:`list` of ndarray
        List of ndarray best fit amplitudes of each planets' TTV basis functions.

    covariance_matrices : :obj:`list` of ndarray
        List of ndarray covariance matrices for each planets' TTV basis functions.

   Parameters
   ----------
   observations_list : :obj:`list` of :obj:`PlanetTransitObservations`
       Set of transit observations to model.
    """

    def __init__(self,observations_list):
        self.observations=observations_list
        for obs in self.observations:
            errmsg1 = "'TransitObservations' contains transits with negative transit numbers. Please re-number transits." 
            assert np.alltrue(obs.transit_numbers>=0), errmsg1

        initial_linear_fit_data = np.array([obs.linear_best_fit() for obs in self.observations ])
        self.T0s = initial_linear_fit_data[:,0]
        self.periods = initial_linear_fit_data[:,1]
        self.basis_function_matrices = [obs.basis_function_matrix() for obs in self.observations ]
        self._maximum_interaction_period_ratio = np.infty
        self._interaction_matrix = SetupInteractionMatrixWithMaxPeriodRatio(self.periods,self.maximum_interaction_period_ratio)
        
    def reset(self):
        """
        Reset TTV model.

        All TTV basis functions are erased and a linear ephemeris is re-fit to each planet's
        transit times. The interaction matrix is reset so that all pair-wise interactions
        are considered. 
        """
        initial_linear_fit_data = np.array([obs.linear_best_fit() for obs in self.observations ])
        self.T0s = initial_linear_fit_data[:,0]
        self.periods = initial_linear_fit_data[:,1]
        self.basis_function_matrices = [obs.basis_function_matrix() for obs in self.observations ]
        self._maximum_interaction_period_ratio = np.infty
        self._interaction_matrix = SetupInteractionMatrixWithMaxPeriodRatio(self.periods,self.maximum_interaction_period_ratio)


    @property 
    def interaction_matrix(self):
        """
        ndarray: Matrix with elements that record whether pair-wise interactions are 
        in each planet's set of TTV basis functions. If :math:`I_{ij}=` True, then 
        the basis functions accounting for perturbations by planet j on
        planet *i* are included in the model for planet i's TTV.
        """
        return self._interaction_matrix
    @interaction_matrix.setter
    def interaction_matrix(self,value):
        self._interaction_matrix = value

    @property
    def maximum_interaction_period_ratio(self):
        """
        float: Maximum period ratio above which planet-planet interactions are ignored in the TTV model.
        """
        return self._maximum_interaction_period_ratio

    @maximum_interaction_period_ratio.setter
    def maximum_interaction_period_ratio(self,value):
       self.interaction_matrix = SetupInteractionMatrixWithMaxPeriodRatio(self.periods,value)
       self._maximum_interaction_period_ratio =  value
    
    @property
    def N(self):
        """int: Number of planets with transit observations."""
        return len(self.observations)
    @property
    def weighted_obs_vectors(self):
        return [obs.weighted_obs_vector for obs in self.observations]
    @property
    def design_matrices(self):
        """:obj:`list` of ndarrays: Design matrices for each planet's TTV model."""
        bfs = self.basis_function_matrices
        uncs= [o.uncertainties for o in self.observations ]
        return [np.transpose(bfs[i].T/uncs[i]) for i in range(self.N)]
    @property
    def covariance_matrices(self):
        A = self.design_matrices
        return [ np.linalg.inv( A[i].T.dot(A[i]) ) for i in range(self.N)]
    
    @property
    def best_fits(self):
        A = self.design_matrices
        y = self.weighted_obs_vectors
        return [np.linalg.lstsq(A[i],y[i],rcond=None)[0] for i in range(self.N)]

    def chi_squareds(self,per_dof=False):
        dms = self.design_matrices
        obs_weighted = self.weighted_obs_vectors
        bests = self.best_fits
        normalized_resids = [obs_weighted[i] - dms[i].dot(bests[i]) for i in range(self.N)]
        if per_dof:
            return [nr.dot(nr) / len(nr) for nr in normalized_resids]  
        else:
            return [nr.dot(nr) for nr in normalized_resids]  
    def Delta_BICs(self):
        chi_squareds = self.chi_squareds()
        bfm_shapes = [bfm.shape for bfm in self.basis_function_matrices] 
        penalty_terms = [ x[1] * np.log( x[0] ) for x in bfm_shapes ]
        BICs = np.array(chi_squareds) + np.array(penalty_terms)
        
        line_fit_resids = [ obs.linear_fit_residuals() / obs.uncertainties for obs in self.observations ]
        line_fit_BICs = np.array( [lfr.dot(lfr) + 2 * np.log(len(lfr)) for lfr in line_fit_resids] )
        return line_fit_BICs - BICs
    def quicklook_plot(self,axis):
        for obs in self.observations:
            resid_MIN = 24*60*(obs.linear_fit_residuals())
            unc_MIN = 24*60*obs.uncertainties
            axis.errorbar(obs.times,resid_MIN,yerr=unc_MIN)
        axis.set_xlabel("Time [d.]")
        axis.set_ylabel("TTV [min.]")
    
    def generate_new_basis_function_matrices(self):
        i_matrix=self.interaction_matrix
        maxTransitNumbers = [np.max(o.transit_numbers)+1 for o in self.observations]
        bf_matrices_full = MultiplanetSystemBasisFunctionMatrices(\
                            self.N,self.periods,self.T0s,maxTransitNumbers,InteractionMatrix=i_matrix)
        return [bf_matrices_full[i][(self.observations[i].transit_numbers)].astype(float) for i in range(self.N)]

    def update_fits(self):
        neg_periods_Q = np.any(np.array(self.periods)<0)
        if neg_periods_Q:
            warnings.warn("Negative period(s) found, resetting periods to linear best fit values!",RuntimeWarning)
            old_max_int_per = self.maximum_interaction_period_ratio
            self.reset()
            self.maximum_interaction_period_ratio = old_max_int_per
        
        self.basis_function_matrices = self.generate_new_basis_function_matrices()
        self.periods = [fit[1] for fit in self.best_fits]

    def update_with_second_order_resonance(self,i1,i2):
        self.basis_function_matrices = self.generate_new_basis_function_matrices()
        pIn = self.periods[i1]
        pOut = self.periods[i2]
        tIn = self.T0s[i1]
        tOut = self.T0s[i2]

        obsIn = self.observations[i1]
        obsOut = self.observations[i2]
        NtrIn = obsIn.transit_numbers[-1] + 1
        NtrOut = obsOut.transit_numbers[-1] + 1

        t2in = dt2_InnerPlanet(pIn,pOut,tIn,tOut,NtrIn)
        t2out = dt2_OuterPlanet(pIn,pOut,tIn,tOut,NtrOut)
        t2in = t2in[obsIn.transit_numbers]
        t2out = t2out[obsOut.transit_numbers]
        self.basis_function_matrices[i1]=np.hstack((self.basis_function_matrices[i1],t2in)).astype(float)
        self.basis_function_matrices[i2]=np.hstack((self.basis_function_matrices[i2],t2out)).astype(float)

        self.periods = [fit[1] for fit in self.best_fits]

    def compute_ttv_significance(self):
        Sigma = self.covariance_matrices
        mu = self.best_fits
        significance_in_sigmas=[]
        for i in range(self.N):
            if len(mu[i] >2):
                muTTV = mu[i][2:]
                SigmaTTVinv= np.linalg.inv(Sigma[i][2:,2:])
                chisquared = muTTV.dot(SigmaTTVinv.dot(muTTV))
                dof = len(muTTV)
                sigma=chiSquared_to_sigmas(chisquared,dof)
                significance_in_sigmas.append(sigma)
            else:
                significance_in_sigmas.append(0)
        return significance_in_sigmas

def interactionIndicies(LMsystem,i,j):
    i_matrix = LMsystem.interaction_matrix
    if  i_matrix[i,j]:
        k0 = 2 + 3 * np.sum( i_matrix[i,:j])
        i_indices = k0 + np.arange(3,dtype=int)
    else:
        i_indices = []

    if i_matrix[j,i]:
        l0 = 2 + 3 * np.sum( i_matrix[j,:i])
        j_indices = l0 + np.arange(3,dtype=int)
    else:
        j_indices = []
    
    return i_indices, j_indices

def chiSquared_to_sigmas(chi2,dof):
    """
    Convert a chi-squared value to a confidence level, in terms of 'sigmas'

    Arguments
    ---------
    
    chi2 : float
        Chi-squared value

    dof  : int
        Number of degrees of freedom

    Returns
    -------
    
    float
        The 'sigma's with the same confidence level for a 1D Gaussian
    """
    p = gammainc(0.5 * dof, 0.5 * chi2)
    return np.sqrt( 2 * gammaincinv( 0.5 , p ) )
#############################################
