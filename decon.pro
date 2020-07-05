pro decon,data,psf,psi,nit,ALGO=algo, flag=flag
;+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
; NAME:
;    decon
; PURPOSE:
;    deconvolve an image with known point spread function using
;    recursive algorithms
; CATEGORY:
;    image processing
; CALLING SEQUENCE:
;    decon,data,psf,psi,nit,ALGO=algo
; INPUTS:
;    data - image
;    psf  - point spread function
;    psi  - old guess for original image
;    nit  - number of iterations
; OPTIONAL KEYWORD INPUTS:
;    ALGO - determines deconvolution algorithm
;           algo = 'simple'  simple inversion
;           algo = 'lucy'    Poisson max. likelihood (Lucy)
;           algo = 'frieden' Frieden max. entropy
;           algo = 'gaussm'  Gaussian max. likelihood
;	    algo = 'jvc'     Jansson-van Cittert
;    default : lucy
; OUTPUTS:
;    psi  - new guess for original image
; RESTRICTIONS:
;    currently works only with 1,2 or 3 dimensional data arrays
; NOTES:
;    size of each dimension must be a positive integer power of 2
; MODIFICATION HISTORY:
;    Wolfgang Brandner, April 1992
;-----------------------------------------------------------------------
;
on_error,1
!except=0
;res=check_math()
;
; initialization
;
list=['simple','lucy','frieden','gaussm','jvc']
listname=['Simple inversion','Poisson max. likelihood (Lucy)',$
          'Frieden max. entropy','Gaussian max. likelihood',$
          'Jansson-van Cittert']
unt=0.000001
dim=1.
;
if n_params() lt 4 then begin
	print,'CALLING SEQUENCE: ',$
	'decon,data,psf,psi,nit[,ALGO=algo]'
	return
endif
;
sd=size(data)
sp=size(psf)
ier = (sd eq sp)
i = where(ier eq 0)
if (i(0) ne -1) then message,$
              'DATA and PSF must be of the same dimension and size'
if (sd(0) gt 3) then message,'DATA should have not more then 3 dimensions'
;
for i=1,sd(0) do begin
	di=sd(i)
	quo = alog(di)/alog(2)
        if (fix(quo) ne quo) then message,$
          'size of each dimension must be a positive integer power of 2'
	dim = dim * di
endfor
;
if not keyword_set(ALGO) then algo = 'lucy'
index=where(algo eq list)
;print,algo
if (index(0) eq -1) then message,'algo must be one of '+list
;
psfn=psf
in=where(psfn le 0) 
if (in(0) ne -1) then psfn(in) = unt
psfn=psfn/total(psfn) ;Normierung
psfn=psfn*dim
;
if (sd(0) eq 1) then psfn = shift(psfn,sd(1)*0.5)
if (sd(0) eq 2) then psfn = shift(psfn,sd(1)*0.5,sd(2)*0.5)
if (sd(0) eq 3) then psfn = shift(psfn,sd(1)*0.5,sd(2)*0.5,sd(3)*0.5)
;
norm=total(data)
;
; main loop
;
;print,'doing deconvolution by '+listname(index)
fpsf=fft(psfn,-1)
;fpsf=float(fft(psfn,-1))
for i = 1,nit do begin
	phi = float(fft(fft(psi,-1)*fpsf,1))
;	phi = double(fft(fft(psi,-1)*fpsf,1))
;	phi = (fft(fft(psi,-1)*fpsf,1))
;if keyword_set(flag) then print, psi
;if keyword_set(flag) and i eq 6 then print, float(fft(fft(data/phi,-1)*fpsf,1))
;if keyword_set(flag) and i eq 6 then print, fft(data/phi,-1)
		div=float(data/phi)
;if keyword_set(flag) and i eq 6 then print, div
;if keyword_set(flag) and i eq 6 then begin
;		nidx=where(finite(div) eq 0, ncnt)
;print, data[nidx], phi[nidx], ncnt
;if ncnt eq 1 then div[nidx]=0.
;endif
;if keyword_set(flag) and i eq 3 then print, div
;if keyword_set(flag) and i eq 3 then print, div
	case algo of
	'simple'  : begin
		    psi = psi * data /phi    
                    psi = psi / total(psi) * norm
                    end
	'lucy'    : psi = psi * float(fft(fft(div,-1)*fpsf,1))
;	'lucy'    : psi = psi * float(fft(fft(data/phi,-1)*fpsf,1))
;	'lucy'    : psi = psi * (fft(fft(data/phi,-1)*fpsf,1))
	'frieden'   : begin
                    psi = psi * exp(float(fft(fft(data/phi,-1)*fpsf,1)))
                    psi = psi / total(psi) * norm
                    end
	'gaussm'  : begin
	            psi = psi * float(exp(fft(fft(data-phi,-1)*fpsf,1)))
                    psi = psi / total(psi) * norm
                    end
	'jvc'	  : begin
		    r= (1.-2*abs(psi-0.5))
		    psi=psi + r * (data-phi)
                    end 
	endcase
;if keyword_set(flag) and i eq nit then print, psi
if keyword_set(flag) and i eq 2 then print, psi
endfor
;if keyword_set(flag) and i eq 7 then begin
;	idx=where(psi gt 0.01)
;	print, psi[idx]
;endif
;
end
