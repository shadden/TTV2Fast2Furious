import pandas as pd
import os
from . import PlanetTransitObservations

_pymodulepath = os.path.dirname(__file__)
_RoweTableFile = _pymodulepath+"/../data/RoweTableDataframe.pkl"

KOITransitTimeDataframe = pd.read_pickle(_RoweTableFile)

def KOISystemObservations(KOINumber):
    """
    Retrieve the transit time observations for a KOI system

    Parameters
    ----------
    KOINumber : int or str
        The KOI number of the system to retrieve

    Returns
    -------
    observations : dict of PlanetTransitObservations
        Transit time observations of the KOIs in the system.
    """
    KOI = int(KOINumber)
    if KOI not in KOITransitTimeDataframe.KOI:
        print("No entry found for KOI%d"%KOI)
        return None
    observations = dict()
    koi_system_dframe = KOITransitTimeDataframe.query("KOI_System=='%d'"%KOI)
    for koi,obs in koi_system_dframe.groupby('KOI'):
        observations.update({
            koi:PlanetTransitObservations(obs.TransitNumber.values,obs.TransitTime.values,obs.eTTV.values)
            })
    return observations
