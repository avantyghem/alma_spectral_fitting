#!/usr/bin/env python

from __future__ import print_function
import sys
import time
import warnings
import numpy as np
import pickle
import multiprocessing as mp
from itertools import product

from scipy.stats import f, norm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.stats import sigma_clip
import pyregion

try:
    import tqdm

    tqdm_flag = True
except ImportError:
    tqdm_flag = False

from lmfit.models import LinearModel, GaussianModel

try:
    from mylmfit import MyGaussianModel
except ImportError:
    MyGaussianModel = GaussianModel
from matplotlib import pyplot as plt


def read_fits_file(fitscube):

    hdulist = fits.open(fitscube)
    hdr = hdulist[0].header
    data = hdulist[0].data
    cube = data[0, :, :, :]  # first dimension is the Stokes parameter

    return hdr, cube


def trim_hdu(fitsfile, hdu=0):
    # Extract a 2D image from a radio cube
    # Remove info for axes 3 (frequency) and 4 (polarization)

    hdulist = fits.open(fitsfile)
    hdr = hdulist[0].header

    # wcs = WCS(hdr).celestial
    extra_keys = ["NAXIS", "CTYPE", "CRVAL", "CDELT", "CRPIX", "CUNIT", "CROTA"]
    toremove = [r + i for r in extra_keys for i in ["3", "4"]]

    # pc_axes = list(product(range(1, 4), [3, 4]))
    pc_axes = list(set(product(range(1,5), range(1,5))) - set(product(range(1,3), range(1,3))))
    zero = "0" if "PC03_01" in hdr else ""
    pc_keys = [f"PC{zero}{x[0]}_{zero}{x[1]}" for x in pc_axes]

    #     pc_keys = ["PC0{0}_0{1}".format(*x) for x in pc_axes]
    # else:
    #     # pc_keys = [f"PC{x[0]}_{x[1]}" for x in pc_axes]
    #     pc_keys = ["PC{0}_{1}".format(*x) for x in pc_axes]
    toremove += pc_keys

    for key in toremove:
        if key in hdr:
            hdr.pop(key)
    hdr["NAXIS"] = 2

    data = hdulist[hdu].data
    im = data[0, 0, :, :]  # take a single slice

    newhdu = fits.PrimaryHDU(data=im, header=hdr)
    return newhdu


def create_mask(cubefile, region):
    """Create a 2D mask of all values within 'reg_file'. Values inside the 
    region are True, while those outside are False"""
    hdu = fits.open(cubefile)
    cube = hdu[0].data[0, :, :, :]
    trimhdu = trim_hdu(cubefile)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = pyregion.open(region).as_imagecoord(trimhdu.header)
        mask = r.get_mask(hdu=hdu[0], shape=cube.shape[-2:])
    return mask


def get_velocity(hdr):
    width = float(hdr["CDELT3"]) / 1000.0  # in km/s
    vmin = float(hdr["CRVAL3"]) / 1000.0
    N = float(hdr["NAXIS3"])

    velocity = vmin + width * np.arange(N)
    return velocity


def extract_beam_info(hdr):

    pixsize = abs(hdr["CDELT1"])  # Pixel size in degrees

    # Define the beam properties
    minor = hdr["BMIN"] / pixsize  # major axis in pixels
    major = hdr["BMAJ"] / pixsize
    angle = hdr["BPA"]
    area = 1.1331 * major * minor  # in pix**2

    beam = {
        "major": major,
        "minor": minor,
        "pa": angle,
        "area": area,
        "pixsize": pixsize * 3600,
    }

    def getSemiAxis(width):
        if round(width) % 2 == 0:  # is even
            width = width + 1  # Make it odd
        return int(round((width - 1) / 2.0))

    beam["a"] = getSemiAxis(major)
    beam["b"] = getSemiAxis(minor)

    if (angle > -30) and (angle < 30):  # ellipse mainly along y-axis
        beam["y"] = beam["a"]
        beam["x"] = beam["b"]
    elif (angle < -60) or (angle > 60):  # ellipse mainly along x-axis
        beam["y"] = beam["b"]
        beam["x"] = beam["a"]
    else:
        beam["y"] = beam["a"]  # ellipse is roughly diagonal
        beam["x"] = beam["a"]  # use the major axis for both directions

    beam["boxarea"] = (2 * beam["x"] + 1) * (2 * beam["y"] + 1)

    """
    print 'Beam: {} x {} PA {}'.format(major, minor, angle)
    print 'Semi-major: {a}, Semi-minor: {b}'.format(**beam)
    print 'X-coord: {x}, Y-coord: {y}'.format(**beam)
    print 'Area: {area}, Box area: {boxarea}'.format(**beam)
    """

    return beam


def extract_spectrum(cube, vel, xi, yi, beam=None, velocity_binning=1):

    if beam is None:
        return cube[:, yi, xi]

    xrng = [xi - beam["x"], xi + beam["x"] + 1]
    yrng = [yi - beam["y"], yi + beam["y"] + 1]

    box = cube[:, yrng[0] : yrng[1], xrng[0] : xrng[1]]

    # Each pixel is in Jy/beam
    # Sum to get Jy.pixel/beam
    # Divide by boxarea to get Jy/beam again
    # Multiply by N_beams = boxarea/area to get Jy
    # sum(box)/boxarea * (boxarea/area) = sum(box)/area
    flux = box.sum(axis=2).sum(axis=1) / beam["area"]

    dv = vel[1] - vel[0]  # full channel width
    spec = {"vel": vel, "flux": flux, "dv": dv}
    spec = bin_spectrum(spec, velocity_binning)

    spec["error"] = get_error(spec)

    if DEBUG:
        print("Spectrum rms: {}".format(spec["error"]))

    return spec


def bin_spectrum(spec, w=1):
    # Bin the spectrum over width w

    if w == 1:
        return spec

    flux = spec["flux"]
    vel = spec["vel"]

    newv = np.array([np.mean(vel[w * i : w * (i + 1)]) for i in range(len(vel) / w)])
    newf = np.array([np.mean(flux[w * i : w * (i + 1)]) for i in range(len(vel) / w)])

    binspec = dict(vel=newv, flux=newf)
    binspec["dv"] = binspec["vel"][1] - binspec["vel"][0]
    return binspec


def rms(x):
    # return np.sqrt( np.sum( (x-np.mean(x))**2 / len(x) ) )
    return np.std(x)


def get_error(spec, sigma=2.0, bounds=None):
    """Model-independent determination of the rms in line-free channels"""

    if bounds is not None:
        vlo, vhi = bounds
        mask = (spec["vel"] >= vlo) & (spec["vel"] <= vhi)
    else:
        mask = sigma_clip(
            spec["flux"], sigma_lower=5.0, sigma_upper=sigma, maxiters=20
        ).mask

    noise = spec["flux"][~mask]
    clipV = spec["vel"][~mask]
    error = {"val": rms(noise), "nbins": len(noise)}

    if DEBUG:
        plt.plot(spec["vel"], spec["flux"], ls="", marker="o", color="r")
        plt.plot(clipV, noise, ls="", marker="o", color="b")
        plt.show()

    return error


def fit_spectrum(spec, mod, pars):
    try:
        err = spec["error"]["val"]
    except:
        err = spec["error"]
    result = mod.fit(
        spec["flux"], pars, x=spec["vel"], weights=1.0 / err, scale_covar=False
    )
    return result


def create_empty_model(numComps):
    m = LinearModel()
    for i in range(numComps):
        m += MyGaussianModel(prefix="g{}_".format(i + 1))
    return m


def create_base_model(varyCont=False, varySlope=False):
    lin = LinearModel()
    lin.set_param_hint("slope", vary=varySlope)
    lin.set_param_hint("intercept", vary=varyCont)
    pars = lin.make_params(slope=0, intercept=0)
    return lin, pars


def add_component(m1, p1, initSpec, initVel=None):
    # Make a copy of the input parameter list so that p1 is not changed
    p = p1.__deepcopy__(None)

    # Create the new component and update the parameters
    pfx = "g{}_".format(len(m1.components))
    g = MyGaussianModel(prefix=pfx)
    g.set_param_hint("amplitude", min=0)

    p.update(g.guess(initSpec["flux"], initSpec["vel"], w=2))

    if abs(p[pfx + "center"].value) < 3:  # lmfit explodes if the initial velocity is 0
        p[pfx + "center"].value = abs(p[pfx + "center"].value) + 3.0

    if initVel is not None:
        p[pfx + "center"].value = initVel

    m = m1 + g
    return m, p


def inspect_components(spec, result, maxComps):

    if result is None:
        return False, False

    numComps = len(result.components[1:])

    lastMod = False
    if numComps == maxComps:
        lastMod = True

    validity = [
        inspect_component(spec, result, comp, lastMod)
        for comp in range(1, numComps + 1)
    ]
    decent, great = np.all(validity, axis=0)  # All components must be acceptable

    return decent, great


def inspect_component(spec, result, comp, lastMod, vel_limits=None):

    # Qualities of newest component
    prefix = result.components[comp].prefix
    par = result.params

    # First check that amplitude >= 2sigma
    key = "amplitude"
    significance = par[prefix + key].value / par[prefix + key].stderr

    # Need at least 2sigma for a fit worth exploring
    goodfit = (significance >= 1.5) & (significance != np.inf)

    # Flag components too close to the end of the spectrum
    if par[prefix + "center"] - 1.82 * par[prefix + "fwhm"] / 2 < min(
        spec["vel"]
    ) or par[prefix + "center"] + 1.82 * par[prefix + "fwhm"] / 2 > max(spec["vel"]):
        goodFit = False

    if vel_limits is None:
        vel_limits = [min(spec["vel"]), max(spec["vel"])]
    if par[prefix + "center"] < min(vel_limits) or par[prefix + "center"] > max(
        vel_limits
    ):
        goodFit = False

    # Make sure the line is broad enough, but not too broad
    if par[prefix + "fwhm"] < 2.0 * spec["dv"]:
        goodfit = False
    if lastMod:
        # if par[prefix+'fwhm'] > 0.6*(max(spec['vel'])-min(spec['vel'])):
        if par[prefix + "fwhm"] > 700.0:
            goodfit = False

    # Will skip MC if the line is better than 5 sigma significant
    greatfit = significance >= 5.0
    if not goodfit:
        greatfit = False

    return goodfit, greatfit


def ftest(newfit, oldfit):
    deltaDof = float(oldfit["dof"] - newfit["dof"])
    fstat = (oldfit["chi"] - newfit["chi"]) * newfit["dof"] / (deltaDof * newfit["chi"])
    return fstat


def get_fit_stats(result, err=None):
    numComps = len(result.components) - 1
    pars = result.params

    if err is None:
        err = 1.0 / result.weights

    npar = 3 * numComps + pars["slope"].vary + pars["intercept"].vary
    dof = len(result.data) - npar
    chi = np.sum(((result.best_fit - result.data) / err) ** 2)

    return {"chi": chi, "dof": dof}


def check_significance(result, sig):
    for comp in range(1, len(result.components)):
        prefix = result.components[comp].prefix
        par = result.params

        significance = (
            par[prefix + "amplitude"].value / par[prefix + "amplitude"].stderr
        )
        comp_sig = (significance >= sig) & (significance != np.inf)

        if not comp_sig:
            return False

    return True


def simulate_data(model, err):
    cont = model.params["intercept"].value
    data = model.best_fit + np.random.normal(cont, err, model.data.shape)
    return data


def improved_model(spec, alternate, null, iterations=5000, threshold=3.0):

    if iterations == 0:
        return check_significance(alternate, threshold)

    try:
        err = spec["error"]["val"]
    except:
        err = spec["error"]
    origNull = get_fit_stats(null)
    origAlt = get_fit_stats(alternate)
    fcrit = ftest(origAlt, origNull)
    if DEBUG:
        print(origNull, origAlt)

    if DEBUG:
        print("Running {} MC iterations...".format(iterations))

    global runMC  # Pickle needs to be able to import the module
    # Alternatively, could move this outside of improved_model(),
    # but would not be able to use map()
    def runMC(_iter):
        y = simulate_data(null, err)

        mod1 = create_empty_model(len(null.components) - 1)
        newNull = mod1.fit(
            y, null.params, x=spec["vel"], weights=1.0 / err, scale_covar=False
        )
        nullStats = get_fit_stats(newNull)

        mod2 = create_empty_model(len(alternate.components) - 1)
        newAlt = mod2.fit(
            y, alternate.params, x=spec["vel"], weights=1.0 / err, scale_covar=False
        )
        altStats = get_fit_stats(newAlt)

        return ftest(altStats, nullStats)

    if not DEBUG:  # automated, parallelized fitting
        fprob = np.zeros(iterations)
        mask = np.ones(iterations, dtype=bool)
        for _iter in range(iterations):
            try:
                fprob[_iter] = runMC(_iter)
            except ZeroDivisionError:
                mask[_iter] = False
                continue
        fprob = fprob[mask]
        if len(fprob) < 0.8 * iterations:
            raise ZeroDivisionError
    else:
        # Single spectrum, so parallelize this piece
        pool = mp.Pool(mp.cpu_count())
        # ftestresult = [runMC(_) for _ in range(iterations)]
        if tqdm_flag:
            rs = tqdm.tqdm(pool.imap(runMC, range(iterations)), total=iterations)
            ftestresult = [r for r in rs]
        else:
            ftestresult = pool.map(runMC, range(iterations))
        fprob = np.array(ftestresult)

    p = np.sum(fprob > fcrit)

    if DEBUG:
        print(fcrit, p, (1 - norm.cdf(threshold)) * len(fprob))
        hist, edges = np.histogram(fprob, bins=iterations // 10)
        plt.bar(edges[:-1], hist, width=(edges[:-1] - edges[1:]))
        plt.axvline(fcrit, ls="--", color="k")
        plt.show()

    if p < (1 - norm.cdf(threshold)) * len(fprob):
        return True
    else:
        return False


def run_spectral_fitting(
    spec,
    maxComps=2,
    varyCont=False,
    varySlope=False,
    initVels=None,
    threshold=2.0,
    iterations=1000,
):

    mod, pars = create_base_model(varyCont, varySlope)  # linear model
    result = fit_spectrum(spec, mod, pars)
    modelList = {0: result}

    bestModel = 0
    for compNum in range(1, maxComps + 1):

        resid = spec["flux"] - modelList[compNum - 1].best_fit
        initialize = {"flux": resid, "vel": spec["vel"]}

        if initVels is not None:
            iv = initVels[compNum - 1]
            mod, pars = add_component(mod, pars, initialize, iv)
        else:
            mod, pars = add_component(mod, pars, initialize)

        if DEBUG:
            plt.figure(figsize=[12, 9])
            ax = plt.subplot(111)
            ax.plot(
                spec["vel"] + 0.5 * spec["dv"],
                1000 * initialize["flux"],
                ds="steps",
                marker="",
                color="k",
                linewidth=2,
            )
            compmod = mod.eval_components(params=pars, x=spec["vel"])
            p = "g{}_".format(compNum)
            ax.plot(spec["vel"], 1000 * compmod[p], ls="-", marker="")
            plt.show()

        result = fit_spectrum(spec, mod, pars)
        decentfit, greatfit = inspect_components(spec, result, maxComps)

        if DEBUG and result is not None:
            print("Gaussian component #{}".format(compNum))
            print(result.params)
            print(decentfit, greatfit)
            plot_spectrum(spec, result, spec["error"])

        if decentfit and greatfit:
            better = True
        elif decentfit and not greatfit:
            better = improved_model(
                spec,
                result,
                modelList[compNum - 1],
                iterations=iterations,
                threshold=threshold,
            )
        else:
            better = False

        if better:
            bestModel = compNum
            modelList[compNum] = result
            pars = result.params
        else:
            break

    if DEBUG:
        plot_spectrum(spec, modelList[bestModel], spec["error"])

    return modelList[bestModel]


def plot_spectrum(spec, result=None, error=None):

    plt.figure(figsize=[12, 9])
    ax = plt.subplot(111)

    ax.plot(
        spec["vel"] + 0.5 * spec["dv"],
        1000 * spec["flux"],
        ds="steps",
        marker="",
        color="k",
        linewidth=2,
    )
    ax.axhline(0, color="black", linewidth=1)

    if result is not None:
        smoothv = np.linspace(min(spec["vel"]), max(spec["vel"]), 10 * len(spec["vel"]))
        ax.plot(
            smoothv,
            1000 * result.eval(x=smoothv),
            ls="-",
            marker="",
            color="k",
            linewidth=2,
        )
        compmod = result.eval_components(x=smoothv)
        for comp in range(1, len(result.components)):
            p = result.components[comp].prefix
            ycomp = compmod[p] + compmod["linear"]
            plt.plot(smoothv, 1000 * ycomp, ls="--", marker="", color="k")

    ax.set_xlim([-1000, 1000])

    if error is not None:
        if isinstance(error, dict):
            error = error["val"]
        lheight = 1.0
        xlim = plt.xlim()
        xind = xlim[0] + 0.05 * (xlim[1] - xlim[0])
        ylim = plt.ylim()
        yind = ylim[0] + lheight * (ylim[1] - ylim[0]) - error  # - 0.25
        ax.errorbar(
            xind,
            yind,
            1000 * error,
            ls="",
            marker=".",
            color="k",
            linewidth=2,
            markersize=15,
        )

    ax.set_xlabel("Velocity [km/s]", fontsize=24, labelpad=20)
    ax.set_ylabel("Flux Density [mJy]", fontsize=24)

    ax.xaxis.set_tick_params(pad=8)
    ax.yaxis.set_tick_params(pad=10)

    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.gcf().subplots_adjust(bottom=0.15)
    plt.show()
    return


def trim(full, chunk, num_chunks):
    # Starts counting at 1
    if num_chunks == 1:
        return full

    num_items = len(full) // num_chunks
    if len(full) % num_chunks != 0:
        num_items += 1

    start = (chunk - 1) * num_items
    stop = min(chunk * num_items, len(full) + 1)
    return full[start:stop]


if __name__ == "__main__":

    global DEBUG
    DEBUG = False

    if len(sys.argv) != 2:
        print("USAGE: {} input.txt".format(sys.argv[0]), file=sys.stderr)
        sys.exit(-1)

    inputs = sys.argv[1]

    ### DEFAULTS ###
    DEBUG = False
    region = None
    varyCont = False
    varySlope = False
    initVels = None
    numProcessors = 0
    velocity_binning = 1
    chunk = 1
    num_chunks = 1
    ################

    exec(open(inputs).read())

    """
    Input data file should contain:
    fitscube: <string> name of input data cube
    region: <string> name of region file for the mask
    outbase: <string> 
    DEBUG: <bool>
    varyCont: <bool>
    varySlope: <bool>
    initVels: None or <list>
    numProcessors: <int> -1 for all but 1 processor, 0 for all processors
    thresh: <float>
    numIter: <int>
    maxComps: <int>
    velocity_binning: <int>
    """

    hdr, cube = read_fits_file(fitscube)
    vel = get_velocity(hdr)
    beam = extract_beam_info(hdr)

    if num_chunks != 1:
        outbase += "_chunk{}of{}".format(chunk, num_chunks)

    if DEBUG:
        print(beam)

    if region is None:
        XX, YY = np.mgrid[: hdr["NAXIS1"], : hdr["NAXIS2"]]  # ra, dec
        xx = XX.reshape(-1)
        yy = YY.reshape(-1)
    else:
        mask = create_mask(fitscube, region)
        yy, xx = np.where(mask == True)
    pixlist = list(zip(trim(xx, chunk, num_chunks), trim(yy, chunk, num_chunks)))

    def fit_pixel(ind):
        xi, yi = ind
        spec = extract_spectrum(cube, vel, xi, yi, beam, velocity_binning)
        kwargs = dict(
            varyCont=varyCont,
            varySlope=varySlope,
            maxComps=maxComps,
            initVels=initVels,
            threshold=thresh,
            iterations=numIter,
        )
        if DEBUG:
            result = run_spectral_fitting(spec, **kwargs)
            return
        else:
            try:
                result = run_spectral_fitting(spec, **kwargs)
                numComps = len(result.components) - 1
                output = [spec, result.params, numComps, spec["error"]]
                pstr = pickle.dumps((ind, output), -1)
                return pstr
            except:
                return "None"

    if DEBUG:
        for ind in pixlist:
            fit_pixel(ind)
            break

    else:

        def run_parallel_fitting(indices, nProcess=-1):
            """Parallelized fitting routine"""

            if nProcess <= 0:
                nProcess = max(1, mp.cpu_count() + nProcess)

            num_tasks = len(indices)
            p = mp.Pool(processes=nProcess)

            if tqdm_flag:
                # Very nice progress bar
                rs = tqdm.tqdm(p.imap(fit_pixel, indices), total=num_tasks)
            else:
                rs = p.imap(fit_pixel, indices)
                p.close()

                while True:
                    completed = rs._index
                    if completed == num_tasks:
                        break
                    print(
                        "Waiting for",
                        num_tasks - completed,
                        "of",
                        num_tasks,
                        "tasks to complete...",
                    )
                    time.sleep(60)

            out = [r for r in rs]  # tracks how many pixels failed
            return out

        all_pstrings = run_parallel_fitting(pixlist, numProcessors)
        pstrs = [p for p in all_pstrings if p != "None"]
        unfinished = [ix for (ix, ps) in zip(pixlist, all_pstrings) if ps == "None"]
        num_finished = len(all_pstrings) - len(unfinished)

        if len(unfinished) != 0:
            print("Did not finish the following pixels:")
            print(unfinished)

        outfile = outbase + ".pkl"
        print('Writing results to "{}"'.format(outfile))
        with open(outfile, "wb") as outf:
            for pstr in pstrs:
                pickle.dump(pstr, outf, -1)

        print("Completed {} of {} pixels".format(num_finished, len(pixlist)))
