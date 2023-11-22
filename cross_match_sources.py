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
        self.max_sep=0.5 ##degrees
        self.filter_sun=True
        self.solar_filter_radius=2.5 ### degrees
        self.solar_loc=None ### should be given if filter_sun=True
        self.beam_size=None
        self.adaptive_rms=True
        self.thresh_pix=10
        self.thresh_isl=6
        self.imgcat=None
        self.overwrite=False
        self.matched_source_cat=None
        self.min_elevation=30 #### degrees
        self.num_facets_1d=1 ### number of facets along 1 direction
        
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
       
        head=fits.getheader(self.imagename)
        obstime=Time(head['DATE-OBS'])
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
        
        head=fits.getheader(self.imagename)
        wcs=WCS(head)
        wcs_head=wcs.to_header()
        for key in wcs_head.keys():
            hdu.header[key]=wcs_head[key]
        
        if self.matched_source_cat is None:
            print (self.imagename)
            self.matched_source_cat=self.imagename.replace("-image.fits",".matched.cat.fits")
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
        return img_ra,img_dec,s_code 
        
    def separate_into_facets(self):
        '''
        This function divides the sky into multiple facets. The facets have some overlap.
        The faceting is generated in the pixel domain, hence as the distance frrom the 
        phasecenter increases, the actual area of the facets increases as well 
        '''
        head=fits.getheader(self.imagename)	
        wcs=WCS(head)
        num_sources=len(self.img_ra)
        img_xy=wcs.all_world2pix(list(zip(self.img_ra, self.img_dec,[head['CRVAL3']]*num_sources,[head['CRVAL4']]*num_sources)), 1)
        
        min_x=np.min(img_xy[:,0])
        max_x=np.max(img_xy[:,0])
        min_y=np.min(img_xy[:,1])
        max_y=np.max(img_xy[:,1])
        
        min_x=int(max(0,min_x-1*self.max_sep/abs(head['CDELT1']))) #### some buffer for source finding
        max_x=int(min(max_x+1*self.max_sep/abs(head['CDELT1']),head['NAXIS1']-1)) #### some buffer for source finding
        min_y=int(max(0,min_y-1*self.max_sep/abs(head['CDELT1']))) #### some buffer for source finding
        max_y=int(min(max_y+1*self.max_sep/abs(head['CDELT1']),head['NAXIS2']-1)) #### some buffer for source finding
        
        pix_sep_facet_x=int((max_x-min_x)/self.num_facets_1d)
        pix_sep_facet_y=int((max_y-min_y)/self.num_facets_1d)
        
        self.facets=[]
        
        self.img_ra_faceted=[]
        self.img_dec_faceted=[]
        for i in range(min_x,max_x,pix_sep_facet_x):
            for j in range(min_y,max_y,pix_sep_facet_y):
                facetx_min=max(0,i-self.max_sep/abs(head['CDELT1']))
                facetx_max=min(i+pix_sep_facet_x+self.max_sep/abs(head['CDELT1']),head['NAXIS2']-1)
                facety_min=max(0,j-self.max_sep/abs(head['CDELT1']))
                facety_max=min(j+pix_sep_facet_y+self.max_sep/abs(head['CDELT1']),head['NAXIS1']-1)
                self.facets.append([facetx_min,facetx_max,facety_min,facety_max])
                                #### This way of addinge extra ensures that we are covering
                                #### the whole sky and also have enough matching area for sources
                                #### at the edge.
                pos=np.where((img_xy[:,0]>=facetx_min) & \
                                (img_xy[:,0]<=facetx_max) & \
                                (img_xy[:,1]>=facety_min) & \
                                (img_xy[:,1]<=facety_max))
                self.img_ra_faceted.append(self.img_ra[pos])
                self.img_dec_faceted.append(self.img_dec[pos])                 
                                
        
        
        
        
        
    	
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
        
        
        img_ra,img_dec,s_code = self.read_pybdsf_output()

        pos=np.where((np.isnan(img_ra)==False) & (np.isnan(img_dec)==False) & (s_code=='S'))[0]

        self.img_ra=np.array(img_ra[pos])
        self.img_dec=np.array(img_dec[pos])

        
        self.prune_dense_sources()
        self.prune_low_elevation_sources()
        
        
        #### Only take sources which could be fit by a single gaussian
        pos=np.where((np.isnan(self.img_ra)==False) & (np.isnan(self.img_dec)==False))[0]

        self.img_ra=np.array(self.img_ra[pos])
        self.img_dec=np.array(self.img_dec[pos])
        
        #self.prune_dense_sources()
        
        self.separate_into_facets()
        

        self.img_coord=SkyCoord(self.img_ra*u.degree,self.img_dec*u.degree,frame='icrs')
        

        with h5py.File(self.refcat,'r') as hf:
            self.cat_ra=np.array(hf['ra_2000'])
            self.cat_dec=np.array(hf['dec_2000'])
            source_major_axis=np.array(hf['source_major_axis'])
            source_minor_axis=np.array(hf['source_minor_axis'])
            beam_major=hf.attrs['bm_major']
            beam_minor=hf.attrs['bm_minor']
            self.peak_flux=np.array(hf['I_peak'])
        
        self.refcat_coord=SkyCoord(self.cat_ra*u.degree,self.cat_dec*u.degree,frame='icrs')
        
        self.match_source_after_bulk_shift_correction()
       
        return
        
        
