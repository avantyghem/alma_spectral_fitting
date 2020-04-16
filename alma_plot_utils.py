#!/usr/bin/env python

import sys
import numpy as np
from glob import glob

import matplotlib
# matplotlib.use('Agg')

from matplotlib import pyplot as plt
import aplpy
from astropy import units as u
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM as LCDM
import pdb


class ALMAMap(aplpy.FITSFigure):
    cmaps = dict(amplitude="inferno", center="bwr", fwhm="viridis")
    cbar_label = dict(amplitude="Integrated Flux", 
                      center="Velocity Centroid",
                      fwhm="FWHM")
    unit_label = dict(amplitude="Jy km/s", center="km/s", fwhm="km/s")

    def add_fractional_scalebar(self, z, imgsize, frac=0.2, color='black', lw=4, fontsize=20):
        scale, label = scalebar_info(z, imgsize, frac=frac)
        self.add_scalebar(scale)
        self.scalebar.set_corner('bottom right')
        self.scalebar.set(color=color)
        self.scalebar.set_linewidth(lw)
        self.scalebar.set_label(label)
        self.scalebar.set_font(size=fontsize, weight='bold')

    def coord_labels(self, s=16, hide=False):
        if hide:
            self.axis_labels.hide()
            self.tick_labels.hide()
        else:
            self.axis_labels.set_xtext('RA')
            self.axis_labels.set_ytext('Dec')
            self.tick_labels.set_xformat('hh:mm:ss.s')
            self.axis_labels.set_font(size=s)
            self.tick_labels.set_font(size=s)
            self.ticks.set_color('black')

    def insert_beam(self, major, minor, pa, **kwargs):
        self.add_beam(major=major, minor=minor, angle=pa)
        self.beam.set_color('black')
        self.beam.set(**kwargs)

    def insert_colorbar(self, img_type, cbar_kwargs=None, font_kwargs=None):
        label = f"{self.cbar_label[img_type]} [{self.unit_label[img_type]}]"
        self.add_colorbar()
        # self.colorbar.set_font(size='medium', weight='medium',
        #                        stretch='normal', family='sans-serif',
        #                        style='normal', variant='normal')
        if font_kwargs is not None:
            self.colorbar.set_font(**font_kwargs)
        self.colorbar.set_axis_label_font(size=16)
        defaults = dict(location='right', width=0.2, pad=0.05, ticks=None, 
                        labels=True, box=None, box_orientation='vertical', 
                        axis_label_rotation=None, axis_label_pad=5)
        if cbar_kwargs is not None:
            defaults.update(cbar_kwargs)
        self.colorbar.show(**defaults)
        self.colorbar.set_axis_label_text(label)


def scalebar_info(z, imgsize, frac=0.2):
    imgsize = u.Quantity(imgsize, u.arcsec)
    if isinstance(imgsize.value, np.ndarray):
        imgsize = imgsize[0]
    scale = frac*imgsize
    cosmo = LCDM(70,0.3)
    conv  = cosmo.arcsec_per_kpc_proper(z)
    label = '{:.1f} kpc'.format( (scale/conv).value)
    return scale, label

def beam_info(fitsfile):
    hdu = fits.open(fitsfile)
    if len(hdu) > 1:
        hdr = hdu["VALUE"].header
    else:
        hdr = hdu[0].header
    major = hdr['BMAJ'] # already in degrees
    minor = hdr['BMIN']
    pa = hdr['BPA']
    return major, minor, pa

def contour_levels(vmin=0, vmax=1, nlevs=8, stretch="linear"):
    return np.linspace(vmin, vmax, nlevs+2)[1:-1]

def image_dimensions(imgsize):
    # Formats a dictionary for recentering the figure
    imgsize = u.Quantity(imgsize, u.arcsec)
    if not isinstance(imgsize.value, np.ndarray):
        radius = (0.5*imgsize).to(u.deg).value
        dims = {'radius': radius}
    else:
        width  = imgsize[0].to(u.deg).value
        height = imgsize[1].to(u.deg).value
        dims = {'width': width, 'height': height}
    return dims

def all_maps_single_image(prefix, coords, imgsize, cscale, outfile, 
                          regs=None, cfile=None, levels=8):

    num_comps = len(glob(f'{prefix}_amplitude*.fits'))
    # cfile = '{}_totalflux.fits'.format(prefix)

    dims = image_dimensions(imgsize)
    width = 18
    if "radius" in dims.keys():
        aspect = 1
    else:
        aspect = dims["height"]/dims["width"]
    height = width*aspect*num_comps/3
    fig = plt.figure(figsize=[width+3,height], constrained_layout=True)
    print(dims)

    for comp in range(1, num_comps+1):

        for i,img in enumerate(['amplitude', 'center', 'fwhm']):
            infile = f'{prefix}_{img}{comp}.fits'
            major, minor, pa = beam_info(infile)

            f = ALMAMap(infile, hdu="VALUE", figure=fig, 
                        subplot=(num_comps, 3, i+1+(comp-1)*3))

            vmin, vmax = cscale[img]
            f.show_colorscale(vmin=vmin, vmax=vmax, cmap=f.cmaps[img], 
                              stretch='power', exponent=1.0)

            # f.add_label(0.25,0.925,'Component {}'.format(comp), relative=True, 
            #             color='black', size=20, weight='bold')

            f.recenter(coords.ra.value, coords.dec.value, **dims)

            if cfile is not None:
                f.show_contour(cfile, hdu="VALUE", levels=levels, cmap=None, 
                               colors='black', overlap=True)
            
            f.coord_labels(hide=True)
            f.insert_beam(major, minor, pa)

            # if comp == 1:
            #     f.add_colorbar()
            #     # f.colorbar.set_box([0.025, 0.9, 0.4, 0.01], box_orientation="horizontal")
            #     f.colorbar.set_location('top')
            #     if img == 'center':
            #         f.colorbar.set_axis_label_text('Velocity Centroid [km/s]')
            #     elif img == 'fwhm':
            #         f.colorbar.set_axis_label_text('FWHM [km/s]')
            #     elif img == 'amplitude':
            #         f.colorbar.set_axis_label_text('Integrated Flux [Jy km/s]')
            #         relativeTicks = np.linspace(0.1,0.9,5)
            #         ticks = vmin + relativeTicks * (vmax - vmin)
            #         f.colorbar.set_ticks(ticks)
            #     f.colorbar.set_font(size=16)
            #     f.colorbar.set_axis_label_font(size=20)
            #     f.colorbar.set_axis_label_pad(10)

            if regs is not None:
                for reg in regs:
                    f.show_regions(reg)

            # if comp == 1 and img == 'center':
            #     l = cluster + ' CO({}-{})'.format(J,J-1)
            #     f.add_label(0.5, 1.25, l, relative=True, color='black', size=28, weight='bold')
            # if comp < num_comps:
            #     f.axis_labels.hide_x()
            #     f.tick_labels.hide_x()
            # if comp > 1:
            #     f.colorbar.hide()
            # if img != 'amplitude':
            #     f.axis_labels.hide_y()
            #     f.tick_labels.hide_y()
    return f

def all_maps_separately(prefix, coords, imgsize, cscale, outfile, 
                        regs=None, cfile=None, levels=8):
    return

def plot_map(img_file, coords, imgsize, hdu=0, img_type="amplitude", 
             radec_label=False, img_kwargs=None, regs=None, 
             cbar=True, cbar_kwargs=None, cbar_font=None,
             contour_file=None, contour_kwargs=None):
    '''
    infile: prefix, img_type, comp
    img_size: 
    img_kwargs: stretch (default 'power'), exponent (default 1), cmap, vmin, vmax
    '''
    # img_file = f"{prefix}_{img_type}{comp}.fits"
    major, minor, pa = beam_info(img_file)
    dims = image_dimensions(imgsize)

    width = 6
    if "radius" in dims.keys():
        aspect = 1
    else:
        aspect = dims["height"]/dims["width"]
    height = width*aspect
    # if radec_label:
    #     width += 3
    #     height += 1.5

    fig = plt.figure(figsize=(width, height), constrained_layout=True)
    f = ALMAMap(img_file, hdu=hdu, figure=fig, north=True)

    # This does not check for conflicting parameters for different stretches
    im_args = dict(cmap=f.cmaps[img_type], stretch='power', exponent=1.0)
    if img_kwargs is not None:
        im_args.update(img_kwargs)
    f.show_colorscale(**im_args)
    f.recenter(coords.ra.value, coords.dec.value, **dims)

    if contour_file is not None:
        contour_defaults = dict(hdu=1, levels=8, cmap=None, colors='black', overlap=True)
        if contour_kwargs is not None:
            contour_defaults.update(contour_kwargs)
        f.show_contour(contour_file, **contour_defaults)
    
    f.coord_labels(hide=(not radec_label))
    f.insert_beam(major, minor, pa)

    if cbar:
        f.insert_colorbar(img_type, cbar_kwargs=cbar_kwargs, font_kwargs=cbar_font)

    if regs is not None:
        for reg in list([regs]):
            f.show_regions(reg)

    return f

