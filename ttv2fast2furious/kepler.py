import pandas as pd
import os
from . import PlanetTransitObservations

_pymodulepath = os.path.dirname(__file__)
_RoweTableFile = _pymodulepath+"/../data/RoweTableDataframe.pkl"
_fileExists = os.path.isfile(_RoweTableFile)
if not _fileExists:
    import urllib.request
    print("Downloading transit time table file...\n\t (This may take a minute but you'll only need to do it once.)")
    _url = "https://www.cfa.harvard.edu/~shadden/RoweTableDataframe.pkl"
    os.mkdir(_pymodulepath+"/../data")
    try:
        urllib.request.urlretrieve(_url,_RoweTableFile)
        _fileExists = True
    except: 
        print("Table download failed.")
        print("ttv2fast2furious.kepler requires the data file: \n\t %s"%_RoweTableFile)
        print("The data file can be retrieved manually from %s"%_url)

if _fileExists:
    KOITransitTimeDataframe = pd.read_pickle(_RoweTableFile)
else:
    print("ttv2fast2furious.kepler requires the data file: \n\t %s"%_RoweTableFile)
    print("The data file can be retrieved manually from %s"%_url)
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
    if not _fileExists:
        print("Table data file required.")
        return 
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
