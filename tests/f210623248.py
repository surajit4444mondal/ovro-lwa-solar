from astropy.coordinates import SkyCoord,SkyOffsetFrame
import h5py,os
import pandas as pd
import numpy as np
from astropy import units as u
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
import bdsf,logging,utils
import timeit
from astropy.time import Time
from scipy.interpolate import RBFInterpolator
from astropy.wcs import WCS

def solar_filter(source_loc,solar_loc,solar_filter_radius,frame):
    '''
    solar_loc and source_loc should be Astropy quantities in angles 
    '''
    solar_ra=solar_loc[0].to(u.rad)
    solar_dec=solar_loc[1].to(u.rad)
    source_ra=source_loc[0].to(u.rad)
    source_dec=source_loc[1].to(u.rad)
    
    solar_coord=SkyCoord(solar_ra,solar_dec,frame=frame)
    source_coord=SkyCoord(source_ra,source_dec,frame=frame)
    ang_dist=solar_coord.separation(source_coord)
    return ang_dist.to(u.degree).value
    
def get_peaks_in_image(image,beam_size_pix,doplot=False):
    '''
    This function is adapted from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    '''
    from scipy.ndimage import maximum_filter
    from scipy.ndimage import generate_binary_structure, binary_erosion
    # define an 8-connected neighborhood
    
    
    neighborhood = np.zeros((int(beam_size_pix),int(beam_size_pix)),dtype=int)
    
    for i in range(0,int(beam_size_pix)):
        for j in range(0,int(beam_size_pix)):
            if np.sqrt((i-beam_size_pix/2)**2+(j-beam_size_pix/2)**2)<beam_size_pix/2:
                neighborhood[i,j]=int(1)
    

    

    #apply the local maximum filter; all pixel of maximal value 
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are 
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to 
    #successfully subtract it form local_max, otherwise a line will 
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks, 
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    if doplot:
        fig=plt.figure()
        ax=fig.add_subplot(121)
        plt.imshow(image)
        ax=fig.add_subplot(122)
        plt.imshow(detected_peaks)
        plt.show()
    
    peak_locs=np.where(detected_peaks)
    
    return peak_locs

class match_source_in_facet():
    '''
    This class will take a facet and produce a cross-matched source catalog
    for that facet. No interaction with other image facets will happen here.
    '''
    def __init__(self,refcat_xy,\
                    refcat_coord,\
                    peak_flux,\
                    xmin=None,\
                    ymin=None,\
                    xmax=None,\
                    ymax=None,\
                    beam_size=None,\
                    header=None,\
                    max_sep=None):
        '''
        xmin,xmax,ymin,ymax are the corodinates specifying the facet
        beam size is the size of the restoring beam of the image
        header is the headre of the image
        '''
        self.xmin=xmin
        self.xmax=xmax
        self.ymin=ymin
        self.ymax=ymax
        self.img_ra_facet=None
        self.img_dec_facet=None
        self.img_flux_facet=None
        self.refcat_xy=refcat_xy
        self.refcat_coord=refcat_coord
        self.peak_flux=peak_flux
        self.beam_size=beam_size
        self.head=header
        self.max_sep=max_sep
        self.solar_loc=None
        self.solar_filter_radius=None
        self.filter_sun=None
        
    def produce_reference_image(self,source_num=None):    
        '''
        The goal of this function is to produce a simulated reference image for each facet.
        We will first get all sources which le inside a facet. I assume that all sources are
        point sources and location of the source to be the location of their location provided
        in the catalog. I will smooth the image using a gaussian of appropriate to the resolution
        of the image provided. This image will be returned. is source_num is supplied, only the
        brightest source_num sources will be kept. Remaining will be set to zero.
        '''
        
        from astropy.convolution import convolve, Gaussian2DKernel
                
       
        
        
        
        box_xrange=(self.xmax-self.xmin+1)
        box_yrange=(self.ymax-self.ymin+1)
        
        self.sim_img=np.zeros((box_yrange,box_xrange))
        
        pos=np.where((self.refcat_xy[:,0]>=self.xmin) & \
                        (self.refcat_xy[:,0]<=self.xmax) & \
                        (self.refcat_xy[:,1]>=self.ymin) & \
                        (self.refcat_xy[:,1]<=self.ymax))

        self.ref_facet_source_ra=self.refcat_coord.ra[pos]
        self.ref_facet_source_dec=self.refcat_coord.dec[pos]     
        self.ref_facet_source_x=self.refcat_xy[pos][:,0]
        self.ref_facet_source_y=self.refcat_xy[pos][:,1]
        self.ref_facet_source_peak_flux=self.peak_flux[pos]
        
        
        sorted_pos=np.argsort(self.ref_facet_source_peak_flux)[::-1]
        
        if source_num is not None:
            if source_num<len(sorted_pos):
                self.ref_facet_source_ra=self.ref_facet_source_ra[sorted_pos[0:source_num]]
                self.ref_facet_source_dec=self.ref_facet_source_dec[sorted_pos[0:source_num]]
                self.ref_facet_source_x=self.ref_facet_source_x[sorted_pos[0:source_num]]
                self.ref_facet_source_y=self.ref_facet_source_y[sorted_pos[0:source_num]]
                self.ref_facet_source_peak_flux=self.ref_facet_source_peak_flux[sorted_pos[0:source_num]]
                
               
                
        self.sim_img[(np.array(self.ref_facet_source_y,dtype=int)-self.ymin,\
                  np.array(self.ref_facet_source_x,dtype=int)-self.xmin)]=self.ref_facet_source_peak_flux
                  
        
        

        if self.head['CUNIT1']=='degree' or self.head['CUNIT1']=='deg':
            self.beam_size_pix=self.beam_size/abs(self.head['CDELT1']*60)
        elif self.head['CUNIT1']=='ASEC' or self.head['CUNIT1']=='ARCSEC':
            self.beam_size_pix=self.beam_size*60/abs(self.head['CDELT1'])
        elif self.head['CUNIT1']=='AMIN' or self.head['CUNIT1']=='ARCMIN':
            self.beam_size_pix=self.beam_size/abs(self.head['CDELT1'])           
        else:
            logging.warning("Convolution will not be done.")
            print (self.head['CUNIT1'])
            self.beam_size_pix=None
        
        if self.beam_size_pix is not None:
            beam_stddev=self.beam_size_pix/(2*np.sqrt(2*np.log(2)))
            gauss_kernel=Gaussian2DKernel(beam_stddev,beam_stddev)
            self.sim_img=convolve(self.sim_img, gauss_kernel)
        
    
    def match_sources_in_facet(self):
    
        num_source_in_facet=len(self.img_ra_facet)
        
        self.produce_reference_image(num_source_in_facet*10)  
        
        ### I am finding the peaks in the simulated image because those are the
        ### which I plan to compare with the observed image.
        peak_locs=get_peaks_in_image(self.sim_img,self.beam_size_pix,doplot=False)
        
        if len(peak_locs[0])==0:
            return None, None
        ### converting the relative coordinates to absolute x and y
        peak_locx=peak_locs[1]+self.xmin
        peak_locy=peak_locs[0]+self.ymin
        self.ref_peak_flux=self.sim_img[peak_locs]
        
        wcs=WCS(self.head)
        
        ref_facet_source=wcs.all_pix2world(np.array([peak_locx,peak_locy,np.ones(peak_locx.size),np.ones(peak_locx.size)]).T,0)
        
        self.ref_facet_source=SkyCoord(ref_facet_source[:,0]*u.deg,ref_facet_source[:,1]*u.deg,frame='icrs')
        
        img_facet_source=SkyCoord(self.img_ra_facet,\
                                    self.img_dec_facet,\
                                    frame='icrs')
        
        #### sorting both the image and simulated sources in decreasing order of flux                            
        pos=np.argsort(self.img_flux_facet)[::-1]
        self.img_facet_source_sorted=img_facet_source[pos]
        self.img_flux_facet_sorted=self.img_flux_facet[pos]
        
        #pos=np.argsort(ref_peak_flux)[::-1]
        #self.ref_source_sorted=ref_facet_source#[pos]
        #self.ref_flux_facet_sorted=ref_peak_flux#[pos]
        
        matched_image_source, matched_cat_source= self.find_unique_source_match()
        return matched_image_source, matched_cat_source

      
    def find_unique_source_match(self):
        '''
        This function uses the matched sources returned by Astropy search_around_sky. For each reference source,
        we can have multiple matched sources. We implement a cross-matching scheme to pick the best matched souce.
        '''
        
        num_img_source=len(self.img_facet_source_sorted)
        max_sep_current=self.max_sep
        
        num_trial=0
        max_trial=1
        while num_trial<max_trial:
            matched_img_ra=[]
            matched_img_dec=[]
            matched_cat_ra=[]
            matched_cat_dec=[]
            ref_sources_already_chosen=[]
            source_dist=[]
            for i in range(num_img_source):
                distance = self.img_facet_source_sorted[i].separation(self.ref_facet_source)
                #catalogmask = distance < self.max_sep*u.deg
                #max_flux=np.max(self.ref_peak_flux[catalogmask])

                dist_ratio=distance.deg/max_sep_current
                #### closest source will have smallest distance ratio.

                if len(ref_sources_already_chosen)!=0:

                    dist_ratio[np.array(ref_sources_already_chosen)]=np.nan
                

                flux_ratio=self.img_flux_facet_sorted[i]/self.ref_peak_flux

                #### in all cases, the brightest reference source will have the smallest flux ratio.
            
                ### assuming that the flux will remian same within a 30% fractional bandwidth.
                ### considering that VLSSS is at 82 MHz, the allowed bandwidth is 24 MHz.
                ### So the assumption holds between 70 and 94 MHz
                ref_freq=82
                freq=self.head['CRVAL3']*1e-6

                freq_sigma=12
                flux_func=0.5*np.exp(-(freq-ref_freq)**2/(2*freq_sigma**2))

                
                optimising_func=(1-flux_func)*dist_ratio+flux_func*flux_ratio
                #### dist_func increases as distance increases. So this part of
                ### the optimising function will be minimum at distance=0
                ### and increase as the distance increases. Similarly the brightest
                ### reference source will give smallest flux ratio. For small freq
                ### separations, the flux_func will be largest. 
                
                max_loc=np.nanargmin(optimising_func)

                
                if distance[max_loc].deg>max_sep_current:
                    break
                
                else:
                    if self.filter_sun and solar_filter([self.ref_facet_source[max_loc].ra,\
                                        self.ref_facet_source[max_loc].dec],\
                                        self.solar_loc,\
                                        self.solar_filter_radius,\
                                        'icrs')<self.solar_filter_radius:
                        continue                    
                    ref_sources_already_chosen.append(max_loc)
                    source_dist.append(distance[max_loc].deg)
                    
                    matched_img_ra.append(self.img_facet_source_sorted[i].ra.deg)
                    matched_img_dec.append(self.img_facet_source_sorted[i].dec.deg)
                    matched_cat_ra.append(self.ref_facet_source[max_loc].ra.deg)
                    matched_cat_dec.append(self.ref_facet_source[max_loc].dec.deg)
            if len(source_dist)==0:
                break
            else:
                max_sep_current=min(5*np.median(np.array(source_dist)),self.max_sep)
            num_trial+=1


        matched_image_source=SkyCoord(np.array(matched_img_ra)*u.deg,\
                                        np.array(matched_img_dec)*u.deg,\
                                        frame='icrs')
                                        
        matched_cat_source=SkyCoord(np.array(matched_cat_ra)*u.deg,\
                                        np.array(matched_cat_dec)*u.deg,\
                                        frame='icrs') 
                  
        return  matched_image_source, matched_cat_source                                              

            
        
class cross_match():
    def __init__(self,imagename=None,refcat=None):
        '''
        filter radius is used when sources within some distance from sun are not used
        filter radius is in degrees
        max_sep in degrees. If more than this, this is not considered same source
        We will consider sources which are within 1 beam size of the brightest matched source
        during centrid calculation. 
        Beam size if not gievn will be taken as 9 arcminutes which is typical for a 2.5 km
        array at 50 MHz.
        Beam size should be in arcmin
        adaptive_rms, thresh_pix, thresh_isl are used by pybdsf for source detection.
        if imgcat is not provided, we will append .pybdsf to the imagename and that is the 
        image catalogue
        matched_source_cat is the fits file of the matched sources which contains ra, dec
        of all of the matched sources in both the image and the reference catalogue. If not
        provided, the code will append ".matched.cat" to the imagename and write there.
        File will not be overwritten if overwrite is set to False
        cutout_rad=If provided, a cutout of radius cutout_rad, centered at solar_loc
                    is produced and is used for further processing. The image of the cutout
                    is such that XX-image.fits will go to XX_cutout-image.fits
        '''
        
        self.imagename=imagename
        self.refcat=refcat
        self.cutout_rad=None ### if cutout is needed degrees
                            ### specify the radius in degrees
        self.max_sep=2 ##degrees
        self.filter_sun=True
        self.solar_filter_radius=2.5 ### degrees
        self.solar_loc=None ### should be given if filter_sun=True
        self.beam_size=None
        self.adaptive_rms=True
        self.thresh_pix=10
        self.thresh_isl=8
        self.imgcat=None
        self.overwrite=False
        self.matched_source_cat=None
        self.min_elevation=30 #### degrees
        self.num_facets_1d=10 ### number of facets along 1 direction
        
        
        
    @property 
    def imagename(self):
        '''
        Get the catalog of sources in image
        '''
        return self._imagename
        
    @imagename.setter
    def imagename(self,value):
        '''
        Set the catalog of sources in image
        '''
        
        if os.path.isfile(value):
            self._imagename=value
        else:
            raise IOError ("Image not found")
            
    @property 
    def refcat(self):
        '''
        Get the reference catalog
        '''
        return self._refcat
        
    @refcat.setter
    def refcat(self,value):
        '''
        Set the reference catalog
        '''
        if os.path.isfile(value):
            self._refcat=value
        else:
            raise IOError ("Reference Source catalogue not found")
    
    def get_source_centroids(self,cat_ra,cat_dec,img_ra,img_dec,frame=None):
        cat_source_coords=[]
        img_source_coords=[]
        img_source_ids=[]
        if self.beam_size is None:
            self.beam_size=9 ### 9 arcminutes is typical beam size fo 2.5 km array at 50 MHz
            logging.debug("Beam size not provided. Setting it to 9 arcminutes")
        
        
        
        for i in range(len(self.unique_ids)):
            matched_cat_d2d=self.d2d[self.id_image[self.unique_id_indices[i]:self.unique_id_indices[i]+self.counts[i]]].to(u.arcmin).value
            matched_cat_stokesI=self.peak_flux[self.id_catalog[self.unique_id_indices[i]:self.unique_id_indices[i]+self.counts[i]]]
            matched_cat_ra=cat_ra[self.id_catalog[self.unique_id_indices[i]:self.unique_id_indices[i]+self.counts[i]]]
            matched_cat_dec=cat_dec[self.id_catalog[self.unique_id_indices[i]:self.unique_id_indices[i]+self.counts[i]]]
            brightest_matched_source_pos=np.argmax(matched_cat_stokesI)
            brightest_matched_source_ra=matched_cat_ra[brightest_matched_source_pos]
            brightest_matched_source_dec=matched_cat_dec[brightest_matched_source_pos]
            
            dist_from_brightest=np.sqrt((matched_cat_ra-brightest_matched_source_ra)**2+\
                                        (matched_cat_dec-brightest_matched_source_dec)**2)
            
            pos=np.where(dist_from_brightest>self.beam_size/2)[0]
            if np.any(matched_cat_stokesI[pos]>matched_cat_stokesI[brightest_matched_source_pos]/5):
                logging.debug("Removing source because at least two comparable brightness sources "+\
                                "found at distance greater than beam size")
                continue
            else:
                pos=np.where(dist_from_brightest<self.beam_size/2)[0]
                
                              
                                
            centroid_ra=np.sum(matched_cat_stokesI[pos]*matched_cat_ra[pos])/np.sum(matched_cat_stokesI[pos])### for this small patch of sky
                                                                                   ### flat sky approximation is fine
            centroid_dec=np.sum(matched_cat_stokesI[pos]*matched_cat_dec[pos])/np.sum(matched_cat_stokesI[pos])
            
            distance=np.sqrt((centroid_ra-brightest_matched_source_ra)**2+\
                                (centroid_dec-brightest_matched_source_dec)**2) 
                                
            if distance<self.beam_size/2:
                if self.filter_sun:
                    dist_from_sun=solar_filter(np.array([centroid_ra,centroid_dec])*u.deg,self.solar_loc,\
                                self.solar_filter_radius,frame=frame)
                else:
                    dist_from_sun=1e10
                
                if dist_from_sun>self.solar_filter_radius:                
                    cat_source_coords.append([centroid_ra,centroid_dec])
                    
                    img_source_coords.append([img_ra[self.unique_ids[i]],\
                                        img_dec[self.unique_ids[i]]])
                    img_source_ids.append(self.unique_ids[i])
            #### if distance of centroid is greater than brightest source, then we know that multiple sources are
            #### possible and hence ignored.
                else:
                    logging.debug("RA-DEC of source removed due to close proximity to sun:",\
                                img_ra[self.unique_ids[i]],\
                                img_dec[self.unique_ids[i]])
            
        
        self.img_matched_source_coords=np.array(img_source_coords)
        self.cat_matched_source_coords=np.array(cat_source_coords)
        self.img_source_ids=np.array(img_source_ids,dtype=int)

        logging.info("A total of "+str(len(img_source_coords))+" good sources matched")
        return
    
    def detect_sources(self):
        start=timeit.default_timer()
        if self.imgcat is None:
            self.imgcat=self.imagename.replace("-image.fits",".pybdsf")

        if not os.path.isfile(self.imgcat) or self.overwrite:
            outputs=bdsf.process_image(self.imagename,adaptive_rms=self.adaptive_rms,\
                                    thresh_pix=self.thresh_pix,thresh_isl=self.thresh_isl)
        
        
        
        
            outputs.write_catalog(outfile=self.imgcat,format='csv',catalog_type='srl',clobber=self.overwrite)
            logging.debug("Catalogue of sources found by PyBDSF is written to ",self.imgcat)
        else:
            logging.debug("Catalogue file not written because the file existed and user does not want to overwrite")
        end=timeit.default_timer()
        print (end-start)    
        return
    
   
    
    def prune_low_elevation_sources(self):
        if self.min_elevation is None:
            return
       
        
        obstime=Time(self.head['DATE-OBS'])
        img_coord_sky=SkyCoord(self.img_ra*u.degree,self.img_dec*u.degree,frame='icrs')
        alt,az=utils.radec_to_altaz(img_coord_sky,obstime,'ovro')

        pos=np.where(alt<self.min_elevation)
        self.img_ra[pos]=np.nan
        self.img_dec[pos]=np.nan

        
        
    def write_matched_source_catalogue(self):
        '''
        This function writes the matched source catalog and also puts in some relevant information
        like the WCS of the image and solar filtering information, maximum separation allowed during 
        the osurce finding in the header.
        '''
        self.overwrite=True
        col1 = fits.Column(name='ra', format='D', array=self.img_matched_source_coords[:,0])
        col2 = fits.Column(name='dec', format='D', array=self.img_matched_source_coords[:,1])
        col3 = fits.Column(name='RAJ2000', format='D', array=self.cat_matched_source_coords[:,0])
        col4 = fits.Column(name='DECJ2000', format='D', array=self.cat_matched_source_coords[:,1])
        coldefs = fits.ColDefs([col1, col2, col3, col4])
        hdu = fits.BinTableHDU.from_columns(coldefs)
        hdu.header['maxsep']=self.max_sep
        hdu.header['refcat']=self.refcat
        hdu.header['filtsun']=self.filter_sun
        hdu.header['filtrad']=self.solar_filter_radius
        

        wcs=WCS(self.head)
        wcs_head=wcs.to_header()
        for key in wcs_head.keys():
            hdu.header[key]=wcs_head[key]
        
        if self.matched_source_cat is None:
            print (self.imagename)
            self.matched_source_cat=self.imagename.replace("-image.fits",".facet_matched.cat.fits")
            print (self.matched_source_cat)
        
        if not os.path.isfile(self.matched_source_cat) or self.overwrite:
            hdu.writeto(self.matched_source_cat,overwrite=self.overwrite)
            logging.info("Matched sources written in %s",self.matched_source_cat)
        else:
            logging.debug("Catalogue file not written because the file existed and user does not want to overwrite")
    
    def prune_dense_sources(self):
        '''
        This function will remove sources in the image catalog which are closer than 2 times the maximum separation.
        '''
        img_coord=SkyCoord(self.img_ra*u.degree,self.img_dec*u.degree,frame='icrs')
        id_image, id_catalog, d2d, d3d = img_coord.search_around_sky(img_coord, 2*self.max_sep*u.deg)
        diff=np.diff(id_image)

        pos=np.where(diff==0)[0] ### this means that the same source has been matched twice
                              ### once match is with self, but any other match means that
                              ### the source has matched with another source in the catalog
        self.img_ra[id_image[pos]]=np.nan   ### diff is always a positive integer as id_image is always sorted
        self.img_ra[id_image[pos+1]]=np.nan  ### numpy diff calculates is given by out[i] = a[i+1] - a[i]. So
                                    ### if out[i]=0, this means a[i]=a[i+1]. So we set both pos and pos+1 to nan 
        self.img_dec[id_image[pos]]=np.nan
        self.img_dec[id_image[pos+1]]=np.nan
        
        if diff[-1]==0:
            self.img_ra[id_image[-1]]=np.nan
            self.img_dec[id_image[-1]]=np.nan                   

    def match_source_after_bulk_shift_correction(self):
        self.read_reference_catalog()

        center=SkyOffsetFrame(origin=SkyCoord(np.mean(self.img_ra)*u.degree,np.mean(self.img_dec)*u.degree,frame='icrs'))
        
        img_coord_offset=self.img_coord.transform_to(center)
        #### img_coord_offset is a relative coordinate system, relative to the origin
        
        refcat_coord_offset=self.refcat_coord.transform_to(center)
        if self.filter_sun:
            solar_loc_sky=SkyCoord(self.solar_loc[0],self.solar_loc[1],frame='icrs')
            solar_loc_offset=solar_loc_sky.transform_to(center)
            self.solar_loc=[solar_loc_offset.lon,solar_loc_offset.lat]
            
        
        
        self.id_image, self.id_catalog, self.d2d, d3d = refcat_coord_offset.search_around_sky(img_coord_offset, self.max_sep*u.deg)

        self.unique_ids,self.unique_id_indices,self.counts=np.unique(self.id_image,return_index=True,return_counts=True)  
        self.get_source_centroids(refcat_coord_offset.lon.degree,refcat_coord_offset.lat.degree,\
                                    img_coord_offset.lon.degree,img_coord_offset.lat.degree,frame=center)
        
        shift_lon=self.cat_matched_source_coords[:,0]-self.img_matched_source_coords[:,0]
        shift_lat=self.cat_matched_source_coords[:,1]-self.img_matched_source_coords[:,1]
        
        
        img_coord_offset=SkyCoord(img_coord_offset.lon+np.mean(shift_lon)*u.degree,\
                                    img_coord_offset.lat+np.mean(shift_lat)*u.degree,\
                                    frame=center)
                                    
        self.lon_shift_accumulator=np.mean(shift_lon)*np.ones(np.size(img_coord_offset.lon))
        self.lat_shift_accumulator=np.mean(shift_lat)*np.ones(np.size(img_coord_offset.lon))
        #### these two will accumulate the total shifts applied, so that this can be applied at the end
        #### for each source
        
        
        
        for i in range(5):
            self.id_image, self.id_catalog, self.d2d, d3d = refcat_coord_offset.search_around_sky(img_coord_offset, self.max_sep*u.deg)
            
            if len(self.id_image)==0:
                logging.warning("No source match found")
                break

            self.unique_ids,self.unique_id_indices,self.counts=np.unique(self.id_image,return_index=True,return_counts=True)  
            self.get_source_centroids(refcat_coord_offset.lon.degree,refcat_coord_offset.lat.degree,\
                                    img_coord_offset.lon.degree,img_coord_offset.lat.degree,frame=center)
            
            shift_lon=self.cat_matched_source_coords[:,0]-self.img_matched_source_coords[:,0]
            shift_lat=self.cat_matched_source_coords[:,1]-self.img_matched_source_coords[:,1]    
            
            
            dlonmodel = RBFInterpolator(self.cat_matched_source_coords,shift_lon, kernel="linear", smoothing=3)
            dlatmodel = RBFInterpolator(self.cat_matched_source_coords,shift_lat, kernel="linear", smoothing=3)
            
            shift_lon_interpolated=dlonmodel(np.array([img_coord_offset.lon.degree,img_coord_offset.lat.degree]).T)
            shift_lat_interpolated=dlatmodel(np.array([img_coord_offset.lon.degree,img_coord_offset.lat.degree]).T)


            img_coord_offset=SkyCoord((img_coord_offset.lon.degree+shift_lon_interpolated)*u.degree,\
                                    (img_coord_offset.lat.degree+shift_lat_interpolated)*u.degree,\
                                    frame=center)
            if np.median(np.abs(shift_lon))<0.005 and np.median(np.abs(shift_lat))<0.005 and i>=3:
                logging.debug("shift converged. Exiting at iteration",i)
                break 
            
        img_coord_corrected = img_coord_offset.transform_to(self.img_coord)
        

        
        self.id_image, self.id_catalog, self.d2d, d3d = refcat_coord_offset.search_around_sky(img_coord_offset, self.max_sep*u.deg)
        self.unique_ids,self.unique_id_indices,self.counts=np.unique(self.id_image,return_index=True,return_counts=True)  
        
        #### Now I have the correct image sources matched to correct reference sources. This matching was done
        #### after correcting for the bulk ionospheric shift. But ultimately we would like to correct sources 
        #### in the original image. So while saving I will revert the source coordinates of the image, to what
        #### it was before the corrections were done. 
        
        self.get_source_centroids(refcat_coord_offset.lon.degree,refcat_coord_offset.lat.degree,\
                                    img_coord_offset.lon.degree,img_coord_offset.lat.degree,frame=center)
        
        
        
        self.img_matched_source_coords=np.array([self.img_ra[self.img_source_ids],\
                                                self.img_dec[self.img_source_ids]]).T
          
        cat_matched_source_coords=SkyCoord(self.cat_matched_source_coords[:,0]*u.degree,\
                                            self.cat_matched_source_coords[:,1]*u.degree,\
                                            frame=center).transform_to(self.img_coord)
        self.cat_matched_source_coords=np.array([cat_matched_source_coords.ra.degree,cat_matched_source_coords.dec.degree]).T
        
                                            
        #### above I transform the reference catalog sources which were found in relative coordinates
        #### back to their absolute location
        
        self.write_matched_source_catalogue()
        logging.debug("A total of %s sources in the image has matched with catalogue",self.unique_ids)
        logging.info("Cross matching of sources done")
        
            
          
    def read_pybdsf_output(self):
        img_data=pd.read_csv(self.imgcat,skiprows=5,sep=', ',engine='python')
        img_ra=img_data['RA']
        img_dec=img_data['DEC']
        s_code=img_data['S_Code']
        img_flux=img_data['Total_flux']
        return img_ra,img_dec,s_code,img_flux 
    
    
    def convert_skycoords_to_pixel(self,sky_coordinate):
        '''
        This will take a Astropy SkyCoord and convert that
        to pixel coordinates using the header from the supplied image
        The pixel coorinates will be returned.
        '''    
        
        num_sources=np.size(sky_coordinate.ra)
        wcs=WCS(self.head)
        xy=wcs.all_world2pix(list(zip(sky_coordinate.ra.degree, sky_coordinate.dec.degree,\
                            [self.head['CRVAL3']]*num_sources,[self.head['CRVAL4']]*num_sources)), 1)
        return xy
                     
                                  
    
        
        
        
        
    def read_reference_catalog(self):
        '''
        This function reads the reference catalog. While it for now provides
        only the source location and the peak flux in Jy/beam. I am assuming that
        these are primarily point sources. In future, we can use the full gaussian
        model available from the catalog.
        '''
        with h5py.File(self.refcat,'r') as hf:
            self.cat_ra=np.array(hf['ra_2000'])
            self.cat_dec=np.array(hf['dec_2000'])
            source_major_axis=np.array(hf['source_major_axis'])
            source_minor_axis=np.array(hf['source_minor_axis'])
            beam_major=hf.attrs['bm_major']
            beam_minor=hf.attrs['bm_minor']
            self.peak_flux=np.array(hf['I_peak'])
        
        self.refcat_coord=SkyCoord(self.cat_ra*u.degree,self.cat_dec*u.degree,frame='icrs')        
    
    def match_sources(self):
        '''
        In this function, first I will divide the entire image into multiple facets.
        Next I will also separate the catalog into facets as well. Next each facet is passed 
        onto to a different class, which is specifically built to work with each facet.
        '''
        if self.beam_size is None:
            self.beam_size=9 ### 9 arcminutes is typical beam size fo 2.5 km array at 50 MHz
            logging.debug("Beam size not provided. Setting it to 9 arcminutes")
            
        self.read_reference_catalog()
        self.refcat_xy=self.convert_skycoords_to_pixel(self.refcat_coord)
        
        img_xy=self.convert_skycoords_to_pixel(self.img_coord)
   
        min_x=int(np.min(img_xy[:,0]))
        max_x=int(np.max(img_xy[:,0]))
        min_y=int(np.min(img_xy[:,1]))
        max_y=int(np.max(img_xy[:,1]))
        #TODO Check if we need to add this extra padding 
        #min_x=int(max(0,min_x-1*self.max_sep/abs(head['CDELT1']))) #### some buffer for source finding
        #max_x=int(min(max_x+1*self.max_sep/abs(head['CDELT1']),head['NAXIS1']-1)) #### some buffer for source finding
        #min_y=int(max(0,min_y-1*self.max_sep/abs(head['CDELT1']))) #### some buffer for source finding
        #max_y=int(min(max_y+1*self.max_sep/abs(head['CDELT1']),head['NAXIS2']-1)) #### some buffer for source finding
        
        pix_sep_facet_x=int((max_x-min_x)/self.num_facets_1d)
        pix_sep_facet_y=int((max_y-min_y)/self.num_facets_1d)
        
       
        
        num_trial=0
        max_trial=4
        mean_shift_ra=0
        mean_shift_dec=0
        while num_trial<max_trial:
            self.matched_img=[]
            self.matched_cat=[]
            k=0
            for i in range(min_x,max_x,pix_sep_facet_x):
                for j in range(min_y,max_y,pix_sep_facet_y):
                    facetx_min=i
                    facetx_max=i+pix_sep_facet_x
                    facety_min=j
                    facety_max=j+pix_sep_facet_y
                    
                    #### So the sources in each facet in actual image will not have any overlap.
                    pos=np.where((img_xy[:,0]>=facetx_min) & \
                                    (img_xy[:,0]<=facetx_max) & \
                                    (img_xy[:,1]>=facety_min) & \
                                    (img_xy[:,1]<=facety_max))
                    #self.img_ra_faceted.append(self.img_ra[pos]*u.degree)
                    #self.img_dec_faceted.append(self.img_dec[pos]*u.degree) 
                    #self.img_flux_faceted.append(self.img_flux[pos])
                    #### The image sources should only lie in the facet.
                    
                    
                    
                    facetx_min=max(0,i-2*self.max_sep/abs(self.head['CDELT1']))
                    facetx_max=min(i+pix_sep_facet_x+2*self.max_sep/abs(self.head['CDELT1']),self.head['NAXIS2']-1)
                    facety_min=max(0,j-2*self.max_sep/abs(self.head['CDELT1']))
                    facety_max=min(j+pix_sep_facet_y+2*self.max_sep/abs(self.head['CDELT1']),self.head['NAXIS1']-1)
                    
                    #### the facets where the reference sources are present will have overlap. This is done to
                    #### ensure that the edge sources can also be matched properly.
                    
                    fc=match_source_in_facet(self.refcat_xy,self.refcat_coord,self.peak_flux,\
                                            xmin=int(facetx_min),ymin=int(facety_min),\
                                            xmax=int(facetx_max),ymax=int(facety_max))
                    fc.img_ra_facet=(self.img_ra[pos]+mean_shift_ra)*u.degree
                    fc.img_dec_facet=(self.img_dec[pos]+mean_shift_dec)*u.degree
                    fc.img_flux_facet=self.img_flux[pos]
                    fc.beam_size=self.beam_size
                    fc.head=self.head
                    fc.max_sep=self.max_sep
                    fc.solar_loc=self.solar_loc
                    fc.solar_filter_radius=self.solar_filter_radius
                    fc.filter_sun=self.filter_sun
                                            
                    #self.facets.append([int(facetx_min),int(facety_min),int(facetx_max),int(facety_max)]) ### the facet format is same
                                                                                      ### as the CASA bbox format
                                    #### This way of addinge extra ensures that we are covering
                                    #### the whole sky and also have enough matching area for sources
                                    #### at the edge. This extra facet bit ensures that we can match the edge sources as well.
                    ### the facet dimensions will be used to simulate the reference image. 


                    matched_image_source, matched_cat_source=fc.match_sources_in_facet()  
                    if matched_image_source is None:
                        k+=1
                        continue
                    for elem1,elem2 in zip(matched_image_source,matched_cat_source):
                        if elem1 not in self.matched_img and elem2 not in self.matched_cat:
                            self.matched_img.append(elem1)
                            self.matched_cat.append(elem2)
                            sep=elem1.separation(elem2).deg
                            if sep>2:
                                print ("how!!!")
                        else:
                            index=self.matched_cat.index(elem2)
                            if self.matched_img[index]==elem1:
                                continue
                            else:
                                del self.matched_img[index]
                                del self.matched_cat[index]
                                print ("confused source, deleting")
                    k+=1
                    
            
            
            if len(self.matched_img)!=0:
                num_sources=(len(self.matched_img))
                
                self.img_matched_source_coords=np.zeros((num_sources,2))#np.array([self.matched_img.ra.deg,self.matched_img.dec.deg]).T
                self.cat_matched_source_coords=np.zeros((num_sources,2))#np.array([self.matched_cat.ra.deg,self.matched_cat.dec.deg]).T
                
                for i in range(num_sources):
                    self.img_matched_source_coords[i,0]=self.matched_img[i].ra.deg-mean_shift_ra
                    self.img_matched_source_coords[i,1]=self.matched_img[i].dec.deg-mean_shift_dec
                    self.cat_matched_source_coords[i,0]=self.matched_cat[i].ra.deg
                    self.cat_matched_source_coords[i,1]=self.matched_cat[i].dec.deg
                

                shift_ra=self.cat_matched_source_coords[:,0]-self.img_matched_source_coords[:,0]
                shift_dec=self.cat_matched_source_coords[:,1]-self.img_matched_source_coords[:,1]    
                
                mean_shift_ra=np.median(shift_ra)
                mean_shift_dec=np.median(shift_dec)
                self.max_sep=min(5*np.median(np.sqrt(shift_ra**2+shift_dec**2)),self.max_sep)
                #dlonmodel = RBFInterpolator(self.cat_matched_source_coords,shift_ra, kernel="linear", smoothing=3)
                #dlatmodel = RBFInterpolator(self.cat_matched_source_coords,shift_dec, kernel="linear", smoothing=3)
                
                #shift_ra_interpolated=dlonmodel(np.array([self.img_matched_source_coords[:,0]*u.deg,\
                 #                                           self.img_matched_source_coords[:,1]*u.deg]).T)
                #shift_dec_interpolated=dlatmodel(np.array([self.img_matched_source_coords[:,0]*u.deg,\
                  #                                          self.img_matched_source_coords[:,1]*u.deg]).T)
                num_trial+=1
            else:
                
                break
        
        
        
        
        
    
    def get_image_header(self):
        self.head=fits.getheader(self.imagename)        
    	
    def cross_match_cat(self):
        '''
        This function does the cross-matching. This does not do any of the jobs as such,
        but calls functions as needed to get the job done.
        '''
        
        if self.imgcat is None:
            if self.cutout_rad is not None and self.solar_loc is not None:
                outimage=self.imagename.replace("-image.fits","_sun_cutout-image.fits")
                if not os.path.isfile(outimage) or self.overwrite:
                    os.system("rm -rf "+outimage)
                    utils.get_cutout_image(self.imagename,\
                                                        [self.solar_loc[0].to(u.degree).value,\
                                                        self.solar_loc[1].to(u.degree).value],\
                                                        self.cutout_rad,\
                                                        outimage=outimage)
                
                self.imagename=outimage
            self.detect_sources()
        
        self.get_image_header()
        
        img_ra,img_dec,s_code,img_flux = self.read_pybdsf_output()

        pos=np.where((np.isnan(img_ra)==False) & (np.isnan(img_dec)==False) & (s_code=='S'))[0]

        self.img_ra=np.array(img_ra[pos])
        self.img_dec=np.array(img_dec[pos])
        self.img_flux=np.array(img_flux[pos])
        
        ### I am commention out the prune_dense_sources because I am looking at a method
        ### based on flux sorting and this pruning will create issues with that. 
        #self.prune_dense_sources()
        self.prune_low_elevation_sources()
        
        
        #### Only take sources which could be fit by a single gaussian
        pos=np.where((np.isnan(self.img_ra)==False) & (np.isnan(self.img_dec)==False))[0]

        self.img_ra=np.array(self.img_ra[pos])
        self.img_dec=np.array(self.img_dec[pos])
        self.img_flux=np.array(self.img_flux[pos])
        
        self.img_coord=SkyCoord(self.img_ra*u.degree,self.img_dec*u.degree,frame='icrs')
        
        self.match_sources()
  
        self.write_matched_source_catalogue()

        
        #self.match_source_after_bulk_shift_correction()
       
        return
        
        
