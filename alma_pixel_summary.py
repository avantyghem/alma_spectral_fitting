#!/usr/bin/env python

from __future__ import print_function, division

import sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

from alma_pixel_fitting import CreateEmptyModel
from alma_pixel_maps import loadFitResults


class pixel_fit(object):
    def __init__(self, ind, *args):
        self.ind = ind
        spec, pars, numComps, error = args
        vel = spec['vel']
        mod = CreateEmptyModel(numComps)
        res = mod.fit(spec['flux'], pars, x=vel, 
                      weights=1/error['val'], scale_covar=False)
        self.num_comps = numComps
        self.result = res

    def meta_formatter(self):
        x, y = self.ind
        return [x, y, self.num_comps, self.result.chisqr]

    def table_formatter(self):
        pars = self.result.params
        for mod in self.result.components[1:]:
            p = mod.prefix
            out = self.meta_formatter()
            for key in ['amplitude', 'center', 'fwhm']:
                out.extend([pars[p+key].value, pars[p+key].stderr])
            yield out


if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('USAGE: {} pkl_file'.format(sys.argv[0]))
        sys.exit(-1)

    pkl_file = sys.argv[1]
    outfile = '.'.join(x for x in pkl_file.split('.')[:-1]) + '.csv'

    # Dictionary of ind: [spec, pars, numComps, error]
    data = loadFitResults(sys.argv[1])
    print("Finished loading data")

    unpacked_data = []
    for ind in data:
        pf = pixel_fit(ind, *data[ind])
        for pi in pf.table_formatter():
            unpacked_data.append(pi)

    trans_data = map(list, zip(*unpacked_data))
    dt = Table(trans_data, names=('x', 'y', 'num_comps', 'chisq', 
        'amp', 'amp_err', 'center', 'center_err', 'fwhm', 'fwhm_err'))
    dt.write(outfile, format='ascii', delimiter=',')
