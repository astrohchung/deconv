#modified from pPXF example file (ppxf_example_kinematics_sdss.py)
#make the code as class, so that it loads the galaxy templates only one time and re-use the templates for each pPXF fitting on each MaNGA spaxel.

import numpy as np
from astropy.io import fits
# import glob
from os import path
import ppxf as ppxf_package
from ppxf.ppxf import ppxf
import ppxf.ppxf_util as util


def airtovac(wave_air):
    wave_vac=wave_air.copy()
    for i in range(2):
        sigma2 = (1e4/wave_vac)**2.    # ;Convert to wavenumber squared
        fact = 1.+5.792105e-2/(238.0185 - sigma2) + 1.67917e-3/( 57.362 - sigma2)
        wave_vac = wave_air*fact        #      ;Convert Wavelengt
    return wave_vac

class ppxf_wrap():
    def __init__(self, redshift, wave, specres):
        wave=wave/(1+redshift) #  When measure velocity from IFS observation, 
                               # deredshift to the barycenter should be applied before measuement.
        specres=specres/(1+redshift)
        
# Only use the wavelength range in common between galaxy and stellar library.
        wave_mask = (wave > 3700) & (wave < 7400)
        lam_gal = wave[wave_mask]
        specres=specres[wave_mask]

        c = 299792.458                  # speed of light in km/s
        frac = lam_gal[1]/lam_gal[0]    # Constant lambda fraction per pixel
        fwhm_gal=lam_gal/specres          # Resolution FWHM of every pixel, in Angstroms
        velscale = np.log(frac)*c       # Constant velocity scale in km/s per pixel

#         file_dir='../miles_models/'
#         galaxy_templates=glob.glob(file_dir+'Mun1.30Z*.fits')

        miles_models_dir = path.dirname(path.realpath(ppxf_package.__file__))+'/miles_models/'
        galaxy_templates=['Mun1.30Zp0.00T01.9953_iPp0.00_baseFe_linear_FWHM_2.51.fits',
                          'Mun1.30Zp0.00T10.0000_iPp0.00_baseFe_linear_FWHM_2.51.fits',
                         ]
        print('List of galaxy template files: ',galaxy_templates)

        fwhm_tem=2.51 # Vazdekis+10 spectra have a constant resolution FWHM of 2.51A.
        hdu = fits.open(miles_models_dir+galaxy_templates[0])
        ssp = hdu[0].data
        h2 = hdu[0].header
        lam_temp = h2['CRVAL1'] + h2['CDELT1']*np.arange(h2['NAXIS1'])
        lamRange_temp = np.array([np.min(lam_temp), np.max(lam_temp)])
        
        lam_temp=airtovac(lam_temp)
        lamRange_temp=airtovac(lamRange_temp)

        sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
        templates = np.empty((sspNew.size, len(galaxy_templates)))
        
        fwhm_gal = np.interp(lam_temp, lam_gal, fwhm_gal)        
        fwhm_dif = np.sqrt((fwhm_gal**2 - fwhm_tem**2).clip(0))
        sigma = fwhm_dif/2.355/h2['CDELT1'] # Sigma difference in pixels

        goodpixels = util.determine_goodpixels(np.log(lam_gal), lamRange_temp, 0)
        dv = np.log(lam_temp[0]/lam_gal[0])*c    # km/s
        
        for j, fname in enumerate(galaxy_templates):
            hdu = fits.open(miles_models_dir+fname)
            ssp = hdu[0].data
            ssp = util.gaussian_filter1d(ssp, sigma)  # perform convolution with variable sigma
            sspNew = util.log_rebin(lamRange_temp, ssp, velscale=velscale)[0]
            templates[:, j] = sspNew/np.median(sspNew) # Normalizes templates
                
        self.templates=templates
        self.flux=None
        self.ivar=None
        self.mask=None
        self.wave_mask=wave_mask
        self.lam_gal=lam_gal
        self.goodpixels=goodpixels
        self.redshift=redshift
        self.dv=dv
        self.goodpixels=goodpixels
        self.velscale=velscale

        
    def run(self):        
        c = 299792.458                  # speed of light in km/s
        meps=np.finfo('float64').eps
        
        dv=self.dv
        velscale=self.velscale
        templates=self.templates
        lam_gal=self.lam_gal
        mask=self.mask
        wave_mask=self.wave_mask
        
        use_mask=mask[wave_mask]
        flux=((self.flux)[wave_mask])
        noise=((self.ivar**(-0.5))[wave_mask])

        nonzero_finite_bool=use_mask & (noise > 0) & (np.isfinite(noise)) 
        igoodpixels=np.zeros(len(lam_gal)).astype(int)
        igoodpixels[self.goodpixels]=1  
        maskpixels=(igoodpixels & nonzero_finite_bool) > 0 
        
        noise[(noise <= 0) | ~np.isfinite(noise)]=meps
        
        galaxy = flux/np.median(flux[maskpixels])   # Normalize spectrum to avoid numerical issues
        noise = noise/np.median(flux[maskpixels])
        medsn = np.median(galaxy[maskpixels]/noise[maskpixels])

# Gas emission lines are excluded from the pPXF fit using the GOODPIXELS keyword.
#         vel = c*np.log(1 + z)   # eq.(8) of Cappellari (2017)
        start = [0, 200.]  # (km/s), starting guess for [V, sigma]

        adegree=0
        mdegree=0
        pp=ppxf(templates, galaxy, noise, velscale, start,
                  mask=maskpixels, plot=False, quiet=True, moments=2,
                  degree=adegree, mdegree=mdegree, vsyst=dv, clean=False, lam=lam_gal)
        return pp, medsn
