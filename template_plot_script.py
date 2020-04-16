#!/usr/bin/env python

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import alma_plot_utils as apu

prefix = "CO10_20kmps_1comps_10kiter"
# prefix = "CO10_20kmps_2comps_1kiter"
c = SkyCoord("10:23:39.6205 +4:11:10.703", unit=(u.hourangle, u.deg), frame="fk5")
imgsize = [4, 3] # arcsec by default
# num_comps = len(glob(f"{prefix}_amplitude*.fits"))

cscale = dict(amplitude = dict(vmin=0, vmax=0.22), 
              center = dict(vmin=-300, vmax=0, cmap="cool"),
              fwhm = dict(vmin=100, vmax=400))
cfile = f"{prefix}_totalflux.fits"

# f = apu.all_maps_single_image(prefix, c, imgsize, cscale, None)

for key in ["amplitude", "center", "fwhm"]:
    comp = 1
    img_kwargs = cscale[key]
    cbar_kwargs = dict(box=(0.1, 0.9, 0.8, 0.05), box_orientation="horizontal")
    cbar_font = dict(size=12)
    ckwargs = dict(levels=apu.contour_levels(0, 0.22, 8))
    f = apu.plot_map(prefix, c, imgsize, img_type=key, comp=comp, radec_label=False,
                     cbar=True, cbar_kwargs=cbar_kwargs, cbar_font=cbar_font,
                     img_kwargs=img_kwargs, 
                     contour_file=cfile, contour_kwargs=ckwargs)
    f.add_fractional_scalebar(imgsize, 0.2906)
    outfile = f"{prefix}_{key}{comp}_plot.png"
    f.save(outfile)
    # plt.show()
