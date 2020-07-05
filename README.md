# Example: Deconvolve SDSS-IV MaNGA data and compare the differences in kinematics

Author: Haeun Chung (University of Arizona, http://astrohchung.github.io)

This example deconvolve one of the [SDSS-IV MaNGA](https://www.sdss.org/surveys/manga/) and compare the 2D line-of-sight velocity distribution difference measured from the original MaNGA data and the deconvolved MaNGA data.
Specifically, this example reproduces the Figure 9 of Chung et. al. 2020 (in prep) for the MaNGA data 8313-12705. User can apply the example to any other MaNAGA data just by changing 'plate' and 'ifudsgn' in the example file.

## Required packages:
* PYTHON > v3.5
* NUMPY: > v1.15
* SCIPY: > v1.1.0
* ASTROPY: > v3.0.4
* pPXF: Tested with v6.7.1 - v6.7.7. should work with earlier versions
    
**pPXF needs to be downloaded and installed separately from https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf**


## Installation:
  type and execute the following commands on the git-installed Linux 
  terminal
  
    git clone https://github.com/astrohchung/deconv.git
    
  if you would like to check whether the program is successfully installed,
  please run the simple example code as below.
  
    cd deconv/examples
    python3 deconv_example_manga.py
    
  or open the 'deconv_example_manga.ipynb' in Jupyter Notebook for interactive use.
  
  The example code downloads and produces multiple files:
  - Downloads 'manga-8313-12705-LOGCUBE.fits.gz', '8313-12705.png' and 'drpall-v2_4_3.fits' from SDSS-IV DR16 archieve.
  - Produces deconvolved MaNGA data cube ('manga-8313-12705-LOGCDCUBE.fits.gz')
  - Produces 2D kinematics measurements result from both the original and the deconvolved MaNGA data cube ('manga-8313-12705-LOGCUBE_2DLOS.fits', manga-8313-12705-LOGDCCUBE_2DLOS.fits)
  - Produces five plots ('8313-12705_spectrum_37_37.png', '8313-12705_2dflux.png', '8313-12705_ORI_STELLAR_VEL.png', '8313-12705_DECONV_STELLAR_VEL.png', '8313-12705_ORI_STELLAR_SIGMA.png', '8313-12705_DECONV_STELLAR_SIGMA.png')

  
## Short note for beginners: 
  1. Download and install the latest Anaconda distribution from here: https://www.anaconda.com/download/
  2. Download example code.
      * Windows:
        * Download this repository by clicking "Clone or Download" - "Download ZIP" in this page
        * Unzip the downloaded ZIP file. (Recommended directory for Windows users: Documents)
        * Needs to download an outdated version of ppxf from https://www-astro.physics.ox.ac.uk/~mxc/software/#ppxf (search page with 'outdated version of ppxf' and click the hyperlink on 'here') 
        * Unzip the ppxf outdated version file and put the 'ppxf' folder at the same directory as this file. 
        * For example, if this deconvolution file is located at the 'C:\Users\username\deconv', put the 'ppxf' folder at the same directory so the 'ppxf.py' file will be located at 'C:\Users\username\deconv\ppxf\ppxf.py'
      * Mac/Linux with terminal:
        * execute below commands
          <pre><code>git clone https://github.com/astrohchung/deconv.git
          cd deconv/examples
          </code></pre>
  3. Run Jupyter Notebook. Check this link: https://jupyter-notebook-beginner-guide.readthedocs.io/en/latest/execute.html
  4. Open "deconv_example_manga.ipynb" jupyter notebook file.


## MODIFICATION HISTORY:
    - Haeun Chung, Jul. 5, 2020: Initial Commit
    - Haeun Chung, Jul. 5, 2020: Modify to use miles model templates in pPXF package, remove compression (gzip) command on deconvolved file
    - Haeun Chung, Jul. 5, 2020: Update instruction for Windows users
