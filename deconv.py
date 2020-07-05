#!/usr/bin/env python
# coding: utf-8

# !jupyter nbconvert --no-prompt --to=python deconv.ipynb 


import numpy as np
from scipy.signal import convolve2d 
from os import path, system
from astropy.io import fits
from numpy.fft import fft2, ifft2
from time import perf_counter


def psf_gaussian(npixel=0, ndimension=2, fwhm=0):
    cntrd=np.array([(npixel-1)/2., (npixel-1)/2.])
    x, y = np.meshgrid(np.arange(npixel)-cntrd[0], np.arange(npixel)-cntrd[1], sparse=False)
    d = np.sqrt(x*x+y*y)
    mu=0
    sigma=fwhm/(2*(2*np.log(2))**(0.5))
    psf= np.exp(-( 0.5*(d-mu)**2 / ( sigma**2 ) ) )
    return (psf/np.sum(psf)).astype('float64')


def arr_extension(arr, n_ext_max=999, minv=np.finfo('float64').eps):
    meps=np.finfo('float64').eps
    n_iter=1
    
    ncomp=arr.size
    
    # extension kernel in horizontal/vertical directions
    ext_kernel=np.array([[0,1,0],[1,0,1],[0,1,0]])    
    
    # extension kernel in diagonal directions
    ext_kernel_d=np.array([[1,0,1],[0,0,0],[1,0,1]])  

    while np.sum(arr != minv) != ncomp:
        if n_iter > n_ext_max:
            break
        # mark only non-minimum values
        non_min_mark=(arr != minv)*1   
        
        # weight horizontal/vertical and diagnonal direction differently
        arr_ext=convolve2d(arr, ext_kernel+ext_kernel_d/2**0.5, mode='same')   
        
        # calculate normalization factor
        norm_factor_sum=convolve2d(non_min_mark, ext_kernel+ext_kernel_d*8, mode='same')
        norm_factor=norm_factor_sum % 8
        norm_factor_d=norm_factor_sum // 8
        replace_idx=np.nonzero((non_min_mark == 0) & (norm_factor > 0))

        repcnt=len(replace_idx[0])
        if repcnt > 0:
            arr[replace_idx]=np.clip((arr_ext[replace_idx])/
                                      (norm_factor[replace_idx]+norm_factor_d[replace_idx]/2**0.5),meps,None)
            
        n_iter+=1
    return arr.astype('float64')


def deconv(data,psf,psi,nit):
# modified from IDL rountine "decon.pro" written by Wolfgang Brandner

    meps=np.finfo('float64').eps
    minv=1e-10

    dshape=data.shape

    psfn=psf.copy()
    ngidx=np.nonzero(psfn <= 0) 
    if len(ngidx) > 0:
        psfn[ngidx] = minv
        
    #PSF Normalization
    psfn=psfn/np.sum(psfn) 
    
    psfn = np.roll(psfn,(int(dshape[0]*0.5),int(dshape[1]*0.5)),(0,1))

    norm=np.sum(data)
    fpsf=(fft2(psfn))
    for i in range(nit):
        phi = (ifft2(fft2(psi)*fpsf)).astype('float64')
        check_phi=(phi == 0.)
        if np.sum(check_phi):
            phi = phi+check_phi*meps
        div=(data/phi)
        psi=psi*((ifft2(fft2(div)*fpsf)).astype('float64'))

    return psi


def cube_deconv(flux, wave, mask, psf_fwhm_func, pixelscale=0.5, niter=20, 
                size_up_order=1, 
                min_size=7, #in log base 2; 7 means 128 by 128
                meps=np.finfo('float64').eps,
                cube_margin=13):
    
    flux=flux.astype('float64')
    wave=wave.astype('float64')
    
    flux_size_2d=flux.shape
    dc_arr_size_1d=2**(np.clip(((np.log(flux_size_2d[0]+cube_margin*2)/np.log(2))+size_up_order).astype(int),7,None))
    dc_arr_shape=(dc_arr_size_1d,dc_arr_size_1d)
    empty_fft_arr=(np.zeros(dc_arr_shape)).astype('float64')+meps
    r_min=int((dc_arr_size_1d-flux_size_2d[0])/2)
    r_max=int((dc_arr_size_1d+flux_size_2d[0])/2)
    
    dcflux=flux.copy()
    dcmask=mask.copy()

    print('Start Deconvolution')

    n_finish=0
    t_start=perf_counter()
    for i in range(len(wave)):
        if n_finish==10:
            t = perf_counter()
        if n_finish==20:
            remain_time=(len(wave)-i)*(perf_counter()-t)/10
            print('remaining time to finish deconvolution (approx.): '+('%d' % int(remain_time))+' sec')
            
        fwhm_i=psf_fwhm_func(wave[i])
        ori_arr=empty_fft_arr.copy() 
        
        flux_i=flux[:,:,i]

        nonzero_mask=(mask[:,:,i] == 1)
        nonzero_count=np.sum(nonzero_mask)

        if nonzero_count < 3:
            dcmask[:,:,i]=np.ones((flux_size_2d[0],flux_size_2d[1]))
            continue
        
        median_value=np.median(flux_i[nonzero_mask])
        if median_value < 0:
            dcmask[:,:,i]=np.ones((flux_size_2d[0],flux_size_2d[1]))
            continue

        finite_mask=((np.isfinite(flux_i)) & (flux_i > 0))
        finite_count=np.sum(finite_mask)
        
    
        if finite_count >0 :
            flux_i[~finite_mask]=meps
            ori_arr[r_min:r_max,r_min:r_max]=flux_i
            n_ext_max=int(fwhm_i/pixelscale*3)
            ori_arr=arr_extension(ori_arr, n_ext_max=n_ext_max)

            size_arr=ori_arr.shape[0]
            psf_size=(size_arr+1)
    
            psf_fwhm=(fwhm_i/pixelscale).astype('float64')
            psf_gauss=psf_gaussian(npixel=psf_size, ndimension=2, fwhm=psf_fwhm)
            psf_gauss=(psf_gauss[0:size_arr,0:size_arr]).astype('float64')
            dc_arr=deconv(ori_arr, psf_gauss, ori_arr.copy(), niter)
        else:
            dc_arr=empty_fft_arr.copy()


        dcflux_i=dc_arr[r_min:r_max,r_min:r_max]

        dcmask[:,:,i]=np.isfinite(dcflux_i) & nonzero_mask
        dcflux_i[~dcmask[:,:,i]]=0.

        dcflux[:,:,i]=dcflux_i
        
        n_finish+=1
    print('deconvolution finished\n'+'total time: '+('%d' % int(perf_counter()-t_start))+' sec')
    
    return dcflux, dcmask

