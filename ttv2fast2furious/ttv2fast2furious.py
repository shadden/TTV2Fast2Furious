"""
.. module:: ttv2fast2furious
   :platform: Unix, Mac, Windows
   :synopsis: Routines for TTV analysis

.. moduleauthor:: Sam Hadden <samuel.hadden@cfa.harvard.edu>

"""
import warnings
import numpy as np
from scipy.optimize import lsq_linear
from scipy.special import gammainc,gammaincinv
from scipy.special import ellipk,ellipkinc,ellipe,ellipeinc
import scipy.integrate as integrate
from ttv2fast2furious.ttv_basis_functions import get_nearest_firstorder,get_superperiod,dt0_InnerPlanet,dt0_OuterPlanet,get_fCoeffs
from ttv2fast2furious.ttv_basis_functions import get_nearest_secondorder,dt2_InnerPlanet,dt2_OuterPlanet
from collections import OrderedDict

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
    assert j > 0 , "Bad period ratio! P,P1 = %.3f \t %.3f"%(P,P1)
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

class transit_times_model(OrderedDict):
    """
    Class representing a linear model for a single set of transit times.

    Attributes
    ----------
    observations : :obj:`PlanetTransitObservations` 
        Transit observations to model.
    parameter_bounds  : :obj:`OrderedDict`
        A dictionary that contains bounding intervals
        for each linear model amplitude 
    MaxTransitNumber : int
        Largest transit epoch number of the transits recorded in
        'observations'. This is used when generating new basis 
        functions to add to the model. 
    """
    def __init__(self,observations,suffix=''):
        super(transit_times_model,self).__init__()
        self.observations = observations
        self.parameter_bounds = OrderedDict()
        self.MaxTransitNumber = observations._transit_numbers.max()
        self['T0{}'.format(suffix)] = np.ones(observations._Ntransits)
        self['P{}'.format(suffix)] = observations._transit_numbers
        self.parameter_bounds['P{}'.format(suffix)] = (0,np.inf)
        self._suffix = suffix
    def __reduce__(self):
        """Helps the class play nice with pickle_. 

        .. _pickle: https://docs.python.org/3/library/pickle.html#object.__reduce__"""
        red = (self.__class__, (self.observations,
                                self._suffix),
                                None,None,iter(self.items()))
        return red

   
    def __getitem__(self,key):
        return super().__getitem__(key)[self.mask]
       
    def __setitem__(self,key,val):
        if key not in self.parameter_bounds.keys():  
            self.parameter_bounds[key]=(-np.inf,np.inf)
        super().__setitem__(key,val)
        
    @property
    def mask(self):
        return self.observations._mask

    @property
    def Nrow(self):
        """int: Number of basis function matrix rows"""
        return len(self)

    @property
    def Ncol(self):
        """int: Number of basis function matrix columns"""
        return self.observations.Ntransits

    @property
    def basis_function_matrix(self):
        """Return the basis function matrix."""
        return np.vstack([val[self.mask] for val in self.values()]).T
    
    def design_matrix(self):
        """Return the design matrix."""
        sigma_vec = self.observations.uncertainties
        Atranspose = np.transpose(self.basis_function_matrix) / sigma_vec
        return np.transpose(Atranspose)

    def cov_inv(self):
        """Return the inverese covariance matrix"""
        A = self.design_matrix()
        return np.transpose(A).dot(A)

    def cov(self):
        """Return the covariance matrix."""
        return np.linalg.inv(self.cov_inv())

    def list_columns(self):
        """List the column labels of the basis function matrix"""
        return [key for key in self.keys()]
    
    def weighted_obs_vector(self):
        """Observation times weighed by uncertainties.
           
           Used primarily in least-squares fitting along with design
           matrix. 
        """
        return self.observations.times / self.observations.uncertainties

    def function_mask(self,maskfn):
        """Mask the underlying observations using function 'maskfn'
        Arguments
        ---------
        maskfn : callable
            Function of observationt time used to mask observations. 
            Observations times for which 'maskfn' returns 'False' are 
            masked out, otherwise they are included.
        """
        if not callable(maskfn):
            raise TypeError("'maskfn' must be callable.")
        self.observations.function_mask(maskfn)

    def best_fit(self,full_output=False):
        """Compute the best-fit transit model amplitudes
        subject to the constraints set in 'parameter_bounds'
        attribute.
        
        Arguments
        ---------
        full_output : bool (optional)
            If 'True', return the 'OptimizeResult' object generated
            by scipy.lsq_linear along with the default dictionary
            containing best-fit amplitudes

        Returns
        -------
        dictionary :
            A dictionary containing the best-fit model amplitudes.
        """
        A = self.design_matrix()
        y = self.weighted_obs_vector()
        lb,ub = np.transpose([bounds for bounds in self.parameter_bounds.values()])
        min_result = lsq_linear(A,y,bounds = (lb,ub))
        best_dict = {key:val for key,val in  zip(self.keys(),min_result.x)}
        if full_output:
            return best_dict,min_result
        return best_dict

    def best_fit_vec(self):
        """Get best-fit transit model amplitudes as a vector.

        Returns
        -------
        ndarray : 
            Vector representing the best-fit TTV model amplitudes
        """
        bfdict = self.best_fit()
        return np.array([bfdict[x] for x in self.list_columns()])

    def residuals(self):
        """Return the normalized residuals of the best-fit solution"""
        best_dict,min_result = self.best_fit(full_output=True)
        return min_result.fun
    def chi_squared(self,per_dof=False):
        """Return the chi-squared value of the best-fit solution.

        Arguments
        ---------
        per_dof : bool (optional)
            If true, return the chi-sqaured divided by the 
            number of degrees of freedom (i.e., the number
            of observations minus the number of model 
            parameters). The default value is False.
        """
        resids = self.residuals()
        chisq = resids.dot(resids)
        if per_dof:
            dof = self.Ncol - self.Nrow
            return chisq / dof
        return chisq

    def Delta_BIC(self):
        """
        Return the difference in Bayesian information criteria (BICs)
        between a purely linear transit time ephemeris and the full 
        transit time models.
        """
        chisq = self.chi_squared()
        penalty_term=np.log(self.Ncol)*self.Nrow
        BIC_full = chisq + penalty_term

        line_fit_resids = self.observations.linear_fit_residuals()
        chisq_linear_fit = line_fit_resids.dot(line_fit_resids)
        linear_fit_penalty = np.log(self.Ncol) * 2
        BIC_linear = chisq_linear_fit + linear_fit_penalty
        
        return BIC_linear - BIC_full
        

class TransitTimesLinearModels(object):
    """
    Object representing a collection of transit time linear models in a system of interacting planets.

   Parameters
   ----------
   observations_list : :obj:`list` of :obj:`PlanetTransitObservations`
       Set of transit observations to model.

    Keyword Arguments
    -----------------
    periods : :obj:`list` of floats
        Orbital periods of the transiting planets. Periods are determined
        by an initial least-squares fit if they are not supplied as a 
        keyword argument.
    T0s : :obj:`list` of floats
        Inital times of transits. Determined by a least-squares fit
        when not supplied as a keyword argument
    max_period_ratio : float
        Maximum period ratio for which planet-planet interactions are 
        included in the analytic TTV models. Default value is infinite.
    planet_names : str or list of str
        Names to label each planet with. The planet names appear as
        suffixes on basis function labels. 

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

    """

    def __init__(self,observations_list,**kwargs):
        self.observations=observations_list
        for obs in self.observations:
            errmsg1 = "'TransitObservations' contains transits with negative transit numbers. Please re-number transits." 
            assert np.all(obs.transit_numbers>=0), errmsg1

        planet_names = kwargs.get("planet_names",["{}".format(i) for i in range(len(observations_list))])
        assert len(planet_names) == len(self.observations),\
            "Planet name string '{}' lengh does not match number of observations ({:d})".format(planet_names,len(self.observations))
        self.planet_names = planet_names
        self.planet_name_dict = {planet_name:i for i,planet_name in enumerate(planet_names)}

        periods=kwargs.get("periods",None)
        T0s =kwargs.get("T0s",None)
        max_period_ratio=kwargs.get("max_period_ratio",np.inf)
        if periods is None or T0s is None:
            initial_linear_fit_data = np.array([obs.linear_best_fit() for obs in self.observations ])
            if periods is None:
                periods = initial_linear_fit_data[:,1] 
            if T0s is None:
                T0s = initial_linear_fit_data[:,0]
        self.T0s = T0s
        self.periods = periods
        self.models = [ transit_times_model(obs,suffix = self.planet_names[i]) for i,obs in enumerate(self.observations) ]
        self._maximum_interaction_period_ratio = max_period_ratio
        self._interaction_matrix = SetupInteractionMatrixWithMaxPeriodRatio(self.periods,self.maximum_interaction_period_ratio)
        self.generate_basis_functions()


    def basis_function_matrices(self):
        """list : List containing the basis function matrix of each planet"""
        return [model.basis_function_matrix for model in self.models]

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
        self.models = [ transit_times_model(obs) for obs in self.observations ]
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


    def design_matrices(self):
        """:obj:`list` of ndarrays: Design matrices for each planet's TTV model."""
        
        return [model.design_matrix() for model in self.models]


    def covariance_matrices(self):
        """:obj:`list` of ndarrays: Covariance matrices of each planet's TTV model."""
        return [model.cov() for model in self.models]
    
    def best_fits(self):
        """:obj:`list` of ndarrays: best-fit model amplitudes for each planet's TTV model."""
        best = dict()
        for model in self.models:
            best.update(model.best_fit())
        return best
        
    def chi_squareds(self,per_dof=False):
        """Return the chi-squareds of each transit time model.

        Arguments
        ---------
        per_dof : bool (optional)
            If true, return the chi-sqaured divided by the 
            number of degrees of freedom (i.e., the number
            of observations minus the number of model 
            parameters). The default value is False.

        Returns
        -------
        ndarray : 
            Chi-squared value of each planet's transit time
            model.
        """
        return np.array([mdl.chi_squared(per_dof) for mdl in self.models])

    def Delta_BICs(self):
        """Return the Delta-BIC of each transit model relative
        to linear ephemerides"""
        return np.array([mdl.Delta_BIC() for mdl in self.models])

    def generate_basis_functions(self,second_order_resonances=None):
        """Generate TTV basis functions and update planets' TTV models."""
        for name_i,i in self.planet_name_dict.items():
            per_i = self.periods[i]
            T0_i = self.T0s[i]
            model  = self.models[i]
            Ntrans = model.MaxTransitNumber+1
            for name_j,j in self.planet_name_dict.items():
                if self.interaction_matrix[i,j]:
                    per_j = self.periods[j]
                    T0_j = self.T0s[j]
                    bf_mtrx = get_ttv_basis_function_matrix(
                            per_i, per_j, T0_i, T0_j,Ntrans
                            )
                    bf_mtrx = bf_mtrx[model.observations._transit_numbers]
                    model['dt0_{}{}'.format(name_i,name_j)] = bf_mtrx[:,0]
                    model.parameter_bounds['dt0_{}{}'.format(name_i,name_j)] = (0,1)
                    model['dt1x_{}{}'.format(name_i,name_j)] = bf_mtrx[:,1]
                    model['dt1y_{}{}'.format(name_i,name_j)] = bf_mtrx[:,2]
                           
    def add_second_order_resonance(self,planet1,planet2):
        """Add basis-functions for a second-order resonance.

        Arguments
        ---------
        planet1 : str or int
            Name or index of the first planet
        planet2 : str or int
            Name or index of the second planet
        """

        if type(planet1) is str:
            i1str = planet1
            i1 = self.planet_name_dict[planet1]
        elif type(planet1) is int:
            i1str = list(self.planet_name_dict.keys())[planet1]
            i1 = planet1
        else:
            raise ValueError("'planet1' must be of type 'int' or 'str'")

        if type(planet2) is str:
            i2str = planet2
            i2 = self.planet_name_dict[planet2]
        elif type(planet2) is int:
            i2str = list(self.planet_name_dict.keys())[planet2]
            i2 = planet2
        else:
            raise ValueError("'planet2' must be of type 'int' or 'str'")
        
        p1 = self.periods[i1]
        p2 = self.periods[i2]
        T01 = self.T0s[i1]
        T02 = self.T0s[i2]
        Ntr1 = self.models[i1].MaxTransitNumber + 1
        Ntr2 = self.models[i2].MaxTransitNumber + 1
        if p1 < p2:
            dt2_1 = dt2_InnerPlanet(p1,p2,T01,T02,Ntr1).astype(float)
            dt2_2 = dt2_OuterPlanet(p1,p2,T01,T02,Ntr2).astype(float)
        else:
            dt2_1 = dt2_OuterPlanet(p2,p1,T02,T01,Ntr1).astype(float)
            dt2_2 = dt2_InnerPlanet(p2,p1,T02,T01,Ntr2).astype(float)

        self.models[i1]['dt2x_{}{}'.format(i1str,i2str)] = dt2_1[:,0]
        self.models[i1]['dt2y_{}{}'.format(i1str,i2str)] = dt2_1[:,1]
        self.models[i2]['dt2x_{}{}'.format(i2str,i1str)] = dt2_2[:,0]
        self.models[i2]['dt2y_{}{}'.format(i2str,i1str)] = dt2_2[:,1]

    def update_fits(self):
        """
        Compute the best-fit periods using current model then re-compute
        model basis functions.
        """
        fit_dict = self.best_fits()
        self.periods = [fit_dict['P{}'.format(name)] for name in self.planet_names]
        self.generate_basis_functions()

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

    def mask_observations(self,maskfn):
        """Mask transit observations using function 'maskfn'
        Arguments
        ---------
        maskfn : callable
            Function of observation time used to mask observations. 
            Observations times for which 'maskfn' returns 'False' are 
            masked out, otherwise they are included.
        """
        if not callable(maskfn):
            raise TypeError("'maskfn' must be callable.")
        for model in self.models:
            model.function_mask(maskfn)

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
