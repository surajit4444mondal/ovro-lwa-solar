from astropy.io import fits
import h5py
import numpy as np

catalogue='VLSSR.CATALOG.FIT'

outfile='vlssr_catalogue_after_threshold.hdf5'

hdu=fits.open(catalogue,ignore_missing_end=True,mode='readonly')

try:
	head=hdu[1].header
	data=hdu[1].data
finally:
	hdu.close()
	
tot_sources=head['NAXIS2']

threshold=5 ### Jy/beam in OVRO-LWA image


ra=data['RA(2000)']
dec=data['DEC(2000)']
peak=data['PEAK INT']
major_ax=data['MAJOR AX']
minor_ax=data['MINOR AX']

pos=np.where(peak>threshold)[0]


hf=h5py.File(outfile,'w')

try:
	hf.attrs['bm_major']=head['BM_MAJOR']
	hf.attrs['bm_minor']=head['BM_MINOR']
	hf.attrs['threshold_peak_I']=threshold
	hf.attrs['threshold_unit']='Jy/beam'
	hf.attrs['bm_major_unit']='degree'
	hf.attrs['bm_minor_unit']='degree'
	hf.attrs['ra_unit']='degree'
	hf.attrs['dec_unit']='degree'
	hf.create_dataset('ra_2000',data=ra)
	hf.create_dataset('dec_2000',data=dec)
	hf.create_dataset('I_peak',data=peak)
	hf.create_dataset('source_major_axis',data=major_ax)
	hf.create_dataset('source_minor_axis',data=minor_ax)
finally:
	hf.close()




	



