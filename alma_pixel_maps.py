"""Create 2D maps for fitted parameters."""

from __future__ import print_function
import os
import sys
import pickle

# import cPickle as pickle

import numpy as np
from scipy.stats import norm
from astropy.io import fits

from alma_pixel_fitting import create_empty_model, read_fits_file, get_velocity
from lmfit.models import LinearModel, GaussianModel
from matplotlib import pyplot as plt

try:
    from mylmfit import MyGaussianModel
except ImportError:
    print("Using the original GaussianModel")
    print("Fits have not been corrected for instrumental effetcs.")
    MyGaussianModel = GaussianModel


def load_fit_results(infile):
    data = {}
    with open(infile, "rb") as f:
        while True:
            try:
                ind, res = pickle.loads(pickle.load(f))
                data[ind] = res
            except EOFError:
                break
    return data


def old_load_fit_results(infile):
    with open(infile, "rb") as f:
        data = pickle.load(f)
    return data


def merge_fit_results(infiles):
    data = [load_fit_results(f) for f in infiles]
    best = data[0]

    for d in data[1:]:
        for ind in d:
            if ind not in best:
                best[ind] = d[ind]
            elif d[ind][2] > best[ind][2]:
                best[ind] = d[ind]
    return best


def build_model_list(fitResults):
    modelList = {}
    for ind in fitResults:
        try:
            spec, pars, numComps, error = fitResults[ind]
            vel = spec["vel"]

            mod = create_empty_model(numComps)
            res = mod.fit(
                spec["flux"], pars, x=vel, weights=1 / error["val"], scale_covar=False
            )
            modelList[ind] = res
        except TypeError:
            print("Index {} did not fit properly".format(ind))
    return modelList


def create2Dimg(imgsize, modelList, comp, key, sortedPrefixes):
    """----------------------------------------
    Create a 2D image of both the parameter value
    and its uncertainty for a given component number
    and model parameter.

    Parameters
    ----------
    imgsize: int
        length of a side in the square image
    modelList: dictionary of {index: ModelResult}
    comp: component number specifying the model
          0 - linear model (usually held constant)
          1 - first gaussian component
          2 - second gaussian component
    key: name of the parameter of interest
         Linear: 'intercept' and 'slope'
         Gaussian: 'amplitude', 'center', 'sigma', 'fwhm'

    Returns
    -------
    img: dictionary
        Dictionary containing images of the parameter
    value and uncertainty ('value' and 'error')
    ----------------------------------------"""

    # Create base image
    img = {"value": np.empty((imgsize, imgsize)), "error": np.empty((imgsize, imgsize))}
    img["value"].fill(np.nan)
    img["error"].fill(np.nan)

    for ind in modelList:
        mod = modelList[ind]

        if isinstance(sortedPrefixes[ind], list):
            if comp >= len(mod.components):
                continue
        elif (comp - 1) not in sortedPrefixes[ind].keys():
            continue

        try:
            prefix = sortedPrefixes[ind][comp - 1]  # mod.components[comp].prefix
            xi, yi = ind

            img["value"][yi, xi] = mod.params[prefix + key].value
            img["error"][yi, xi] = mod.params[prefix + key].stderr

        except:
            break

    return img


def createTotalFluxImg(imgsize, modelList):

    # Create base image
    img = {"value": np.empty((imgsize, imgsize)), "error": np.empty((imgsize, imgsize))}
    img["value"].fill(np.nan)
    img["error"].fill(np.nan)

    for ind in modelList:

        mod = modelList[ind]

        if len(mod.components[1:]) == 0:
            continue

        fluxes = [mod.params[c.prefix + "amplitude"].value for c in mod.components[1:]]
        ferrs = [mod.params[c.prefix + "amplitude"].stderr for c in mod.components[1:]]
        ferrs = np.array(ferrs)

        xi, yi = ind
        img["value"][yi, xi] = sum(fluxes)
        img["error"][yi, xi] = np.sqrt(np.sum(ferrs ** 2))

    return img


def produceFitsOutput(img, almaimg, fitsname):

    # Do not create image if no pixels provided a good fit
    if np.all(np.isnan(img["value"])):
        return

    def copyALMAkeywords(hdr, almaimg):
        almafits = fits.open(almaimg)
        almahdr = almafits[0].header

        tocopy = [
            "bmaj",
            "bmin",
            "bpa",
            "object",
            "radesys",
            "ctype1",
            "crval1",
            "cdelt1",
            "crpix1",
            "cunit1",
            "ctype2",
            "crval2",
            "cdelt2",
            "crpix2",
            "cunit2",
        ]

        for c in tocopy:
            try:
                hdr[c] = almahdr[c]
            except KeyError:
                print("Key {} not in header".format(c))

        return

    hdr = fits.Header()
    copyALMAkeywords(hdr, almaimg)

    hdu = fits.PrimaryHDU()
    hd1 = fits.ImageHDU(data=img["value"], header=hdr, name="value")
    hd2 = fits.ImageHDU(data=img["error"], header=hdr, name="error")
    hdulist = fits.HDUList([hdu, hd1, hd2])
    hdulist.writeto(fitsname, overwrite=True)

    return


def getAllPrefixes(model):
    return [comp.prefix for comp in model.components[1:]]


def getUnusedPrefixes(model, prefixes):
    allPrefixes = getAllPrefixes(model)
    return [p for p in allPrefixes if p not in prefixes]


def findAdjacentIndices(ind):
    x, y = ind
    return [
        ((x + 1), y),
        ((x - 1), y),
        (x, (y + 1)),
        (x, (y - 1)),
        ((x + 1), (y + 1)),
        ((x - 1), (y + 1)),
        ((x + 1), (y - 1)),
        ((x - 1), (y - 1)),
    ]


def findClosestComponent(ind, filledAdjacent, prefixList, modelList):
    """
    Could bypass this if len(filledAdjacent) == 1
    """

    allDistances = [
        distanceBetweenComponents(ind, adj, prefixList, modelList)
        for adj in filledAdjacent
    ]
    minDistIdx = np.argmin([comparison["dist"] for comparison in allDistances])
    closestPrefix = allDistances[minDistIdx]["prefix"]

    # unused = getUnusedPrefixes( modelList[ind], prefixList[ind] )
    return closestPrefix


def distanceBetweenComponents(ind, adj, pList, mList):

    compPrefix = pList[adj][-1]
    unusedPrefixes = getUnusedPrefixes(mList[ind], pList[ind])

    dists = [distance(mList[ind], mList[adj], uP, compPrefix) for uP in unusedPrefixes]
    closestPrefix = unusedPrefixes[np.argmin(dists)]

    return {"prefix": closestPrefix, "dist": min(dists)}


def distance(model1, model2, p1, p2):
    """
    Find the distance of a component of model1 from
    the already-filled-in component of model2.
    """

    v1 = model1.params[p1 + "center"].value
    v2 = model2.params[p2 + "center"].value
    w2 = model2.params[p2 + "sigma"].value  # /2.3548200450309493

    a1 = model1.params[p1 + "amplitude"].value
    a2 = model2.params[p2 + "amplitude"].value
    a2e = model2.params[p2 + "amplitude"].stderr  # Is this the best width to use?

    # Should I renormalize??
    s2p = np.sqrt(2 * np.pi)
    vdist = 1 - s2p * w2 * norm.pdf(v1, loc=v2, scale=w2)
    adist = 1 - s2p * a2e * norm.pdf(a1, loc=a2, scale=a2e)

    # How should these distances be combined?
    m = 2
    n = 1
    dist = (vdist ** m * adist ** n) ** (1.0 / (m + n))
    return dist


def sort_prefixes(modelList, algorithm="iterative", **kwargs):
    if algorithm == "iterative":
        return iterative_sorting(modelList, **kwargs)
    elif algorithm == "amplitude":
        return amplitude_sorting(modelList, **kwargs)
    elif algorithm == "velocity":
        return velocity_sorting(modelList, **kwargs)
    elif algorithm == "fwhm":
        return fwhm_sorting(modelList, **kwargs)
    elif algorithm == "match_velocity" or algorithm == "match_vel":
        return velocity_matching(modelList, **kwargs)
    elif algorithm == "match_fwhm":
        return fwhm_matching(modelList, **kwargs)
    elif algorithm == "unsorted":
        return makePrefixList(modelList)
    else:
        raise TypeError("Unidentified algorithm")


def iterative_sorting(modelList, maxComps=3, **kwargs):

    prefixList = {ind: [] for ind in modelList}

    for N in range(1, maxComps + 1):  # N is the component number of the Gaussian

        trimmedModelList = dict(modelList)
        # trimmedModelList = {ind: modelList[ind] for ind in modelList
        #                                        if len(modelList[ind].components) < N+1}

        # Fill in the components with only N components
        for ind in modelList:

            # Ignore anything that has already been taken care of
            if len(modelList[ind].components[1:]) < N:
                trimmedModelList.pop(ind)

            # These should have exactly one unused prefix
            if len(modelList[ind].components[1:]) == N:
                # Find the unused prefix and append it to the list
                unusedPref = getUnusedPrefixes(modelList[ind], prefixList[ind])
                if len(unusedPref) != 1:
                    print("Something went wrong in index {}".format(ind))
                    print(maxComps, N)
                    print(prefixes)
                    print(prefixList[ind])
                    print(unusedPref)
                    sys.exit(-1)
                prefixList[ind].append(unusedPref[0])
                trimmedModelList.pop(ind)

        prevLength = len(trimmedModelList)
        while len(trimmedModelList) != 0:
            """ Need to check that the list is still shrinking. """
            for ind in modelList:
                if ind not in trimmedModelList:
                    continue

                adjacent = findAdjacentIndices(ind)  # list of tuples

                # The adjacent pixels must be in the model (i.e. cannot be out
                # of the original pixel list and must already be filled in
                filledAdjacent = [
                    adj
                    for adj in adjacent
                    if adj in modelList and len(prefixList[adj]) == N
                ]

                # Need to wait longer for another pixel to be filled in
                minNearest = 4
                if len(filledAdjacent) < minNearest:
                    continue

                # Find the prefix of the closest component
                closestPrefix = findClosestComponent(
                    ind, filledAdjacent, prefixList, modelList
                )

                prefixList[ind].append(closestPrefix)
                trimmedModelList.pop(ind)

            curLength = len(trimmedModelList)
            if curLength == prevLength:
                ind = trimmedModelList.keys()[0]
                unused = getUnusedPrefixes(modelList[ind], prefixList[ind])
                prefixList[ind].append(unused[0])
                trimmedModelList.pop(ind)
            prevLength = len(trimmedModelList)

    return prefixList


def amplitude_sorting(modelList, **kwargs):

    prefixList = {ind: {} for ind in modelList}
    for ind in modelList:
        mod = modelList[ind]
        vals = mod.params.valuesdict()
        prefixes = getAllPrefixes(mod)

        amp = [vals[p + "amplitude"] for p in prefixes]
        amp = np.array(amp)

        sortedPref = [prefixes[i] for i in np.argsort(amp)[::-1]]

        for (comp_ind, pref) in enumerate(sortedPref):
            prefixList[ind][comp_ind] = pref

    return prefixList


def velocity_matching(modelList, velocities=[-200, 200], **kwargs):

    velocities = np.array(velocities)
    prefixList = {ind: {} for ind in modelList}
    for ind in modelList:
        mod = modelList[ind]
        vals = mod.params.valuesdict()
        prefixes = getAllPrefixes(mod)

        unusedPref = list(prefixes)
        remainingComps = list(velocities)

        while len(unusedPref) != 0:
            vels = [vals[p + "center"] for p in unusedPref]
            pairs = np.array(np.meshgrid(vels, remainingComps)).T.reshape((-1, 2))
            closest_ind = np.argmin(abs(pairs[:, 0] - pairs[:, 1]))

            index_pairs = np.array(
                np.meshgrid(range(len(unusedPref)), range(len(remainingComps)))
            ).T.reshape((-1, 2))
            prefix_ind, rcomp_ind = index_pairs[closest_ind]

            comp_ind = list(velocities == remainingComps[rcomp_ind]).index(True)
            prefixList[ind][comp_ind] = unusedPref[prefix_ind]

            unusedPref.pop(prefix_ind)
            remainingComps.pop(rcomp_ind)

    return prefixList


def velocity_sorting(modelList, mag=True, **kwargs):

    prefixList = {ind: {} for ind in modelList}
    for ind in modelList:
        mod = modelList[ind]
        vals = mod.params.valuesdict()
        prefixes = getAllPrefixes(mod)

        vels = [vals[p + "center"] for p in prefixes]

        if mag == True:
            vels = np.abs(np.array(vels))
        else:
            vels = np.array(vels)
        sortedPref = [prefixes[i] for i in np.argsort(vels)]

        for (comp_ind, pref) in enumerate(sortedPref):
            prefixList[ind][comp_ind] = pref

    return prefixList


def fwhm_sorting(modelList, **kwargs):

    prefixList = {ind: {} for ind in modelList}
    for ind in modelList:
        mod = modelList[ind]
        vals = mod.params.valuesdict()
        prefixes = getAllPrefixes(mod)

        fwhm = [vals[p + "fwhm"] for p in prefixes]
        fwhm = np.array(fwhm)

        sortedPref = [prefixes[i] for i in np.argsort(fwhm)[::-1]]

        for (comp_ind, pref) in enumerate(sortedPref):
            prefixList[ind][comp_ind] = pref

    return prefixList


def fwhm_matching(modelList, fwhms=[450, 100], **kwargs):

    fwhms = np.array(fwhms)
    prefixList = {ind: {} for ind in modelList}
    for ind in modelList:
        mod = modelList[ind]
        vals = mod.params.valuesdict()
        prefixes = getAllPrefixes(mod)

        unusedPref = list(prefixes)
        remainingComps = list(fwhms)

        while len(unusedPref) != 0:
            vels = [vals[p + "fwhm"] for p in unusedPref]
            pairs = np.array(np.meshgrid(vels, remainingComps)).T.reshape((-1, 2))
            closest_ind = np.argmin(abs(pairs[:, 0] - pairs[:, 1]))

            index_pairs = np.array(
                np.meshgrid(range(len(unusedPref)), range(len(remainingComps)))
            ).T.reshape((-1, 2))
            prefix_ind, rcomp_ind = index_pairs[closest_ind]

            comp_ind = list(fwhms == remainingComps[rcomp_ind]).index(True)
            prefixList[ind][comp_ind] = unusedPref[prefix_ind]

            unusedPref.pop(prefix_ind)
            remainingComps.pop(rcomp_ind)

    return prefixList


def makePrefixList(modelList):
    """-----------------------
    Create a dictionary composed of a list of prefixes for each index
    prefix = {idx1: [prefix1, prefix2],
              idx2: [prefix1], idx3: [prefix1]}     # and so on
    -----------------------"""
    prefixList = {}
    for ind in modelList:
        prefixList[ind] = {
            i: comp.prefix for (i, comp) in enumerate(modelList[ind].components[1:])
        }
    return prefixList


def checkPrefixList(prefixList, modelList):
    """------------------------------------
    Check to make sure that the prefix list
    has the correct number of components.
    ------------------------------------"""
    isgood = True
    for ind in modelList:
        try:
            p = prefixList[ind]
            if len(p) != len(modelList[ind].components[1:]):
                print("Incorrect number of prefixes for index {}".format(ind))
                isgood = False
        except KeyError:
            print("Index {} not in prefixList".format(ind))
            isgood = False
    return isgood


if __name__ == "__main__":

    if len(sys.argv) < 3:
        print(
            "USAGE: {} fitsimg data1 [data2 ...]".format(sys.argv[0]), file=sys.stderr
        )
        sys.exit(-1)

    fitsimg = sys.argv[1]
    data = sys.argv[2:]

    if "chunk" in data[0]:
        outbase = data[0].split("_chunk")[0]
    else:
        outbase = data[0].split(".pkl")[0]

    ##### Extra Inputs #####
    # outbase = 'Phoenix_CO32_20kmps_2sig'
    ########################

    print("Loading the data")
    fitResults = merge_fit_results(data)
    print("...done")

    print("Building the model list")
    modelList = build_model_list(fitResults)
    print("...done")

    print("Sorting the components")
    # sortedPrefixes = sort_prefixes(modelList, algorithm='unsorted')
    # sortedPrefixes = sort_prefixes(modelList, algorithm='iterative', maxComps=2)
    sortedPrefixes = sort_prefixes(modelList, algorithm="amplitude")
    # sortedPrefixes = sort_prefixes(modelList, algorithm='velocity', mag=True)
    # sortedPrefixes = sort_prefixes(modelList, algorithm='fwhm')
    # sortedPrefixes = sort_prefixes(modelList, algorithm='match_vel', velocities=[50,-400])
    # sortedPrefixes = sort_prefixes(modelList, algorithm='match_fwhm', fwhms=[100,450])
    print("...done")

    hdr, cube = read_fits_file(fitsimg)
    imgsize = hdr["NAXIS1"]

    print("Creating the images")
    for comp in range(1, 3):
        for key in ["amplitude", "center", "fwhm"]:
            outfits = "{}_{}{}.fits".format(outbase, key, comp)
            img = create2Dimg(imgsize, modelList, comp, key, sortedPrefixes)
            produceFitsOutput(img, fitsimg, outfits)

    img = createTotalFluxImg(imgsize, modelList)
    produceFitsOutput(img, fitsimg, "{}_totalflux.fits".format(outbase))
    print("...done")
