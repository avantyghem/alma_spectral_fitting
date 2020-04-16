#!/usr/bin/env python

from __future__ import print_function
import os
import sys
import pdb
import warnings
import numpy as np
from matplotlib import pyplot as plt

import pyregion
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy.table import Table
from astropy import units as u
from astropy.constants import c, k_B
c = c.to(u.km/u.s)

from lmfit.models import LinearModel, GaussianModel
try:
    from mylmfit import MyGaussianModel
except ImportError:
    MyGaussianModel = GaussianModel
from GasMass import calcGasMassFromResult, calcGasMass, calcLineLuminosity


class Spectrum(object):

    def __init__(self, data, **kwargs):
        self.data = data
        self.meta = dict()
        self.set(**kwargs)
        if 'intensity' not in self.data:
            try:
                self.set_intensity()
            except:
                pass
        return

    def __getitem__(self, item):
        item = item.lower()
        if item == 'vel':
            item = 'velocity'
        elif item == 'dv':
            item = 'delta_v'

        if item in self.data:
            return self.data[item]
        elif hasattr(self, item):
            return getattr(self, item)
        else:
            raise KeyError('Error: key {} not in object'.format(item))
    
    def __contains__(self, value):
        return hasattr(self, value)
    
    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key.lower(), val)
            self.meta[key.lower()] = val
        return

    @property
    def frequency(self):
        # Assumes the radio velocity convention:
        return self.radio_velocity_to_frequency(self["velocity"])

    def radio_velocity_to_frequency(self, velocity):
        # v_radio = c (f0 - f)/f0
        return self.restfrq * (1 - velocity/c.value)

    @classmethod
    def from_file(cls, infile):
        hdulist = fits.open(infile)
        hdr = hdulist[1].header
        data = hdulist[1].data
        
        spec = {key.name.lower(): data[key.name] for key in data.columns}
        kwds = hdr[8+2*len(data.columns):]
        return cls(spec, **kwds)

    def write(self, outfile, overwrite=False):
        # Save the spectrum in a fits file
        if os.path.exists(outfile) and overwrite == False:
            print('File exists and overwrite=False')
            return

        cols = [self.data['velocity'], self.data['flux'], self.data['intensity']]
        names = ('velocity', 'flux', 'intensity')
        t = Table(cols, meta=self.meta, names=names)
        t.write(outfile, format='fits', overwrite=overwrite)
        return

    def to_kelvin(self):
        flux = self['flux']*u.Jy
        area = (self['area']*u.arcsec**2).to(u.rad**2).value
        vobs = self['restfrq']*u.GHz
        kelvin = c*c / (2*k_B*vobs**2) * flux/area
        return kelvin.to(u.K)

    def set_intensity(self):
        self.data['intensity'] = self.to_kelvin()

    # @property
    # def intensity(self):
    #     self.data['intensity'] = self.to_kelvin()

    @classmethod
    def multiple_spectra_from_file(cls, specfile, region_file):
        with open(region_file, "r") as regf:
            lines = regf.readlines()
        if "DS9" in lines[0]:
            prefix = lines[:3]
            regions = lines[3:]
        else:
            prefix = lines[0]
            regions = lines[1:]

        for regstr in regions:
            # new_regstr = "".join(prefix+[regstr])
            # print(new_regstr)
            yield cls.extract_from_region(specfile, regstr)

    @classmethod
    def extract_from_region(cls, specfile, region):
        # Each pixel is in Jy/beam
        # Sum to get Jy.pixel/beam
        # Divide by boxarea to get Jy/beam again
        # Multiply by N_beams = boxarea/area to get Jy
        # sum(box)/boxarea * (boxarea/area) = sum(box)/area

        hdu  = fits.open(specfile)
        hdr  = hdu[0].header
        cube = hdu[0].data[0,:,:,:]

        vel = get_velocity(hdr)
        beam = extract_beam_info(hdr)
        trimhdu = get_trimmed_HDU(specfile)
        objname = hdr['object']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try: 
                r = pyregion.open(region).as_imagecoord(trimhdu.header)
            except IOError:
                r = pyregion.parse(region).as_imagecoord(trimhdu.header)
            mask = r.get_mask(hdu=hdu[0], shape=cube.shape[-2:]) #hdu=trimhdu)

        flux = np.zeros(vel.shape)
        for i in range(len(vel)):
            im = cube[i,:,:]
            flux[i] = np.sum(im[mask]) / beam['area']

        pix_asec = abs(hdr['CDELT1'])*3600
        regarea = np.sum(mask) * pix_asec**2
        dv = vel[1] - vel[0]
        vobs = hdr['RESTFRQ']

        spec = dict(velocity=vel, flux=flux)
        kwds = dict(delta_v=dv, area=regarea, pix_asec=pix_asec, restfrq=vobs/1e9,
                    bmaj=beam['major'], bmin=beam['minor'], bpa=beam['pa'], 
                    barea=beam['area'], objname=objname)

        return cls(spec, **kwds)

    def smooth(self, w=1):
        # Bin the spectrum over width w

        if w == 1:
            return

        flux = self['flux']
        vel = self['vel']

        newv = np.array([np.mean(vel[w*i:w*(i+1)]) for i in range(len(vel)/w)])
        newf = np.array([np.mean(flux[w*i:w*(i+1)]) for i in range(len(vel)/w)])
        newdv = (newv[1] - newv[0])

        self.data = dict(velocity=newv, flux=newf)
        self.set_intensity()
        self.set(delta_v=newdv)
        return
    
    def get_error(self, sigma=2.0, bounds=None):
        '''---------------------------------
        Determine the rms of the line-free channels -- model-independent
        ---------------------------------'''
    
        if bounds is not None:
            vlo, vhi = bounds
            mask = (self['vel'] >= vlo) & (self['vel'] <= vhi)
        else:
            mask = sigma_clip(self['flux'], sigma_lower=5.0, sigma_upper=sigma, 
                              iters=20).mask

        noise = self['flux'][~mask]
        #clipV = spec['vel'][~mask]
        self.set(error=np.std(noise), num_error_bins=len(noise))
        return

    '''
    @property
    def error(self):
        if not self._reset_error:
            return self._error
            
        sigma = self._sigma
        bounds = self._bounds
        if bounds is not None:
            vlo, vhi = bounds
            mask = (self['vel'] >= vlo) & (self['vel'] <= vhi)
        else:
            mask = sigma_clip(self['flux'], sigma_lower=5.0, sigma_upper=sigma, 
                              iters=20).mask

        noise = self['flux'][~mask]
        #clipV = spec['vel'][~mask]
        self.set(error=np.std(noise), num_error_bins=len(noise))
        self._error = np.std(noise)
        self._num_error_bins = len(noise)

    @error.setter()
    def error():
        return self._error

    def set_error_params(self, sigma=2.0, bounds=None):
        self._sigma = sigma
        self._bounds = bounds
        self._reset_error = True
    '''

    def integrate(self, bounds=None):
        if bounds is not None:
            vmin, vmax = bounds
            mask = (self["vel"] > vmin) & (self["vel"] < vmax)
            vel = self["vel"][mask]
            data = self["flux"][mask]
        else:
            vel = self["vel"]
            data = self["flux"]
        integ = data.sum()*self.delta_v
        Nl = len(vel)
        Nb = self.num_error_bins
        uncert = np.sqrt(self.delta_v**2 * self.error**2 * Nl*(1+Nl/Nb))
        return integ, uncert

    def flux_weighted_velocity(self, bounds=None):
        if bounds is not None:
            vmin, vmax = bounds
            mask = (self["vel"] > vmin) & (self["vel"] < vmax)
            vel = self["vel"][mask]
            data = self["flux"][mask]
        else:
            vel = self["vel"]
            data = self["flux"]
        fvel = np.sum(data*vel)/np.sum(data)
        return fvel    
    
    def _create_model(self, num_comps=1, guess=True):
        lin = LinearModel()
        lin.set_param_hint('slope', vary=False)
        lin.set_param_hint('intercept', vary=False)
        pars = lin.make_params(slope=0, intercept=0)
        mod = lin
    
        for i in range(num_comps):
            g = MyGaussianModel(prefix='g{}_'.format(i+1))
            g.set_param_hint('amplitude', min=0.)
    
            if guess:
                pars.update( g.guess(self['flux'], self['vel'], w=2) )
            else:
                pars.update( g.make_params(amplitude=0.15, center=0, sigma=200/2.35) )
    
            mod = mod + g
    
        return mod, pars
    
    def fit(self, num_comps=1, guess=True, bounds=None):
        self.get_error(bounds=bounds)
        self.model, self.params = self._create_model(num_comps, guess=guess)
        self.result = self.model.fit(self['flux'], self.params, x=self['velocity'], 
                                     weights=1./self.error, scale_covar=False)
        return self.result

    def fit_summary(self):
        pars = self.result.params
        for mod in self.result.components[1:]:
            p = mod.prefix
            out = []
            for key in ['amplitude', 'center', 'fwhm']:
                out.extend([pars[p+key].value, pars[p+key].stderr])
            yield out

    def output_results(self, outfile, z, J, XCO=1., iso='12CO'):
        '''Calculate the gas mass and output the best-fitting parameters
        to a file'''
        result = self.result
        out = open(outfile,'w')

        print(result.fit_report(), file=out)
        print('Bin size: {} km/s'.format(self.delta_v), file=out)
        print('rms: {obj.error} over {obj.num_error_bins} bins'.format(obj=self), file=out)

        fluxratio = {1:1.0, 2:3.2, 3:7.2}

        totalFlux = {'value': 0, 'error': 0}
        for mod in result.components[1:]:
            p = mod.prefix
            pars = result.params
            Mass = calcGasMassFromResult(result, p, z, COratio=fluxratio[J], XCO=XCO)
            print('Mass: {value:.3g} +/- {error:.2g} Msun'.format(**Mass), file=out)
            print('\t'.join(str(x) for x in [
                            pars[p+'center'].value, pars[p+'center'].stderr,
                            pars[p+'fwhm'].value, pars[p+'fwhm'].stderr,
                            pars[p+'amplitude'].value, pars[p+'amplitude'].stderr,
                            Mass['value']/1e8, Mass['error']/1e8]))

            totalFlux['value'] += result.params[p+'amplitude'].value
            totalFlux['error'] += result.params[p+'amplitude'].stderr**2
        totalFlux['error'] = np.sqrt(totalFlux['error'])

        totalMass = calcGasMass(totalFlux, z, COratio=fluxratio[J], XCO=XCO)
        totalLum = calcLineLuminosity(totalFlux, z, J, iso)
        print('Total Flux: {value:.3g} +/- {error:.2g} Jy km/s'.format(**totalFlux), file=out)
        print('Total Luminosity: {value:.3g} +/- {error:.2g} K km/s pc2'.format(**totalLum), file=out)
        print('Total Mass: {value:.3g} +/- {error:.2g} Msun'.format(**totalMass), file=out)

        return


    def plot(self, ax=None, plot_fit=False, plotfile=None, **kwargs):
        '''Plot the spectrum along with the best-fitting model'''

        vel = self["velocity"]
        # res = self.result
        scale = 1000

        if ax is None:
            plt.figure(figsize=[12,9])
            ax = plt.subplot(111)

        ax.plot(vel+0.5*self.delta_v, scale*self["flux"],
                ls='steps', marker='', color='k', linewidth=2)
        ax.fill_between(steppify(self["vel"], isX=True),
                        steppify(scale*(self["flux"]-self.error)),
                        steppify(scale*(self["flux"]+self.error)),
                        facecolor="0.75", edgecolor="k")
        ax.axhline(0, color='k', linewidth=1)

        if plot_fit:
            self.plot_model(ax=ax, c="k")

        ax.set_xlabel('Velocity [km/s]',fontsize=24, labelpad=20)
        ax.set_ylabel('Flux Density [mJy]',fontsize=24)

        ax.xaxis.set_tick_params(pad=8)
        ax.yaxis.set_tick_params(pad=10)

        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)

        # ax.set_xlim([-1000,600])
        #ax.set_ylim([-2,5])

        # ax.errorbar(xind, yind, scale*spec['error'], ls='', marker='.', 
        #             color='k', linewidth=2, markersize=15)

        plt.gcf().subplots_adjust(bottom=0.15)
        #plt.tight_layout()
        if plotfile is not None:
            plt.savefig(plotfile)

        return

    def plot_model(self, result=None, ax=None, **kwargs):
        vel = self["velocity"]
        scale = 1000
        if result is None:
            result = self.result

        if ax is None:
            ax = plt.gca()

        smoothv = np.linspace(min(vel), max(vel), 10*len(vel))
        pl = ax.plot(smoothv, 1000*result.eval(x=smoothv), ls='-', marker='', 
                    linewidth=2, **kwargs)

        compmod = result.eval_components(x=smoothv)
        for comp in result.components[1:]:
            p = comp.prefix
            ycomp = compmod[p] + compmod["linear"]
            plt.plot(smoothv, scale*ycomp, ls='--', marker='', color=pl[0].get_color())

    
def get_velocity(hdr):
    width = float(hdr['CDELT3'])/1000.  # in km/s
    vmin = float(hdr['CRVAL3'])/1000.
    N = float(hdr['NAXIS3'])

    velocity = vmin + width*np.arange(N)
    return velocity


def extract_beam_info(hdr):

    pixsize = abs(hdr['CDELT1'])    # Pixel size in degrees

    minor = hdr['BMIN'] / pixsize   # major axis in pixels
    major = hdr['BMAJ'] / pixsize
    angle = hdr['BPA']
    area  = 1.1331*major*minor      # in pix**2

    beam = {'major':major, 'minor':minor, 'pa':angle, 'area':area}
    return beam


def get_trimmed_HDU(fitsfile):

    hdu = fits.open(fitsfile)
    hdr = hdu[0].header
    hdr['NAXIS'] = 2

    toremove = ['NAXIS3', 'NAXIS4', 
                'CTYPE3', 'CRVAL3', 'CDELT3', 'CRPIX3', 'CUNIT3',
                'CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4']
    
    pckwds = ['PC03_01', 'PC03_02', 'PC03_03', 'PC03_04',
              'PC04_01', 'PC04_02', 'PC04_03', 'PC04_04',
              'PC01_03', 'PC01_04', 'PC02_03', 'PC02_04']
    if pckwds[0] not in hdr:
        pckwds = [pc.replace("0", "") for pc in pckwds]
    toremove += pckwds

    for key in toremove:
        hdr.pop(key)

    data = hdu[0].data
    im = data[0,0,:,:]  # take a single slice

    newhdu = fits.PrimaryHDU(data=im, header=hdr)
    return newhdu

def steppify(arr, isX=False):
    """
    Stolen from pyspeckit
    *support function*
    Converts an array to double-length for step plotting
    """
    if isX:
        interval = abs(arr[1:]-arr[:-1]) / 2.0
        newarr = np.array(list(zip(arr[:-1]-interval,arr[:-1]+interval))).ravel()
        newarr = np.concatenate([newarr,2*[newarr[-1]+interval[-1]]])
    else:
        newarr = np.array(list(zip(arr,arr))).ravel()
    return newarr


if __name__ == '__main__':

    if len(sys.argv) not in [3, 4]:
        print('USAGE: {} cube region [outfile]'.format(sys.argv[0]))
        sys.exit(-1)

    cubefile = sys.argv[1]
    region = sys.argv[2]

    if len(sys.argv) == 4:
        outfile = sys.argv[3]
    else:
        outfile = region.replace('.reg', '_spec.fits')

    spec = Spectrum.extract_from_region(cubefile, region)
    spec.write(outfile, overwrite=True)
