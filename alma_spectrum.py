#!/usr/bin/env python

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

from lmfit.models import LinearModel, GaussianModel
try:
    from mylmfit import MyGaussianModel
except ImportError:
    MyGaussianModel = GaussianModel


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
            print 'File exists and overwrite=False'
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
        kelvin = c**2 / (2*k_B*vobs**2) * flux/area
        return kelvin.to(u.K)

    def set_intensity(self):
        self.data['intensity'] = self.to_kelvin()

    @classmethod
    def extract_from_region(cls, specfile, region):

        hdu  = fits.open(specfile)
        hdr  = hdu[0].header
        cube = hdu[0].data[0,:,:,:]

        vel = get_velocity(hdr)
        beam = extract_beam_info(hdr)
        trimhdu = get_trimmed_HDU(specfile)
        objname = hdr['object']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = pyregion.open(region).as_imagecoord(trimhdu.header)
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

        self.data = dict(velocity=vel, flux=flux)
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
                'CTYPE4', 'CRVAL4', 'CDELT4', 'CRPIX4', 'CUNIT4',
                'PC03_01', 'PC03_02', 'PC03_03', 'PC03_04',
                'PC04_01', 'PC04_02', 'PC04_03', 'PC04_04',
                'PC01_03', 'PC01_04', 'PC02_03', 'PC02_04']
    for key in toremove:
        hdr.pop(key)

    data = hdu[0].data
    im = data[0,0,:,:]  # take a single slice

    newhdu = fits.PrimaryHDU(data=im, header=hdr)
    return newhdu


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print 'USAGE: {} cube region'.format(sys.argv[0])
        sys.exit(-1)

    cubefile = sys.argv[1]
    region = sys.argv[2]
    outfile = region.replace('.reg', '_spec.fits')

    spec = Spectrum.extract_from_region(cubefile, region)
    spec.write(outfile, overwrite=True)
