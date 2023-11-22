import numpy as np
import cross_match_sources
from astropy.io import fits
from astropy import wcs
from scipy.interpolate import RBFInterpolator
import logging,os
import utils
from astropy import units as u

def update_image_data(imagename, shift_ra,shift_dec):
        '''
        Since we are only concerned with the solar location and the sun occupies a very small patch
        of the sky, we will assume that this is a single global shift of the image. And global shift
        can be easily put in by using numpy roll
        '''
        pixel_scale=utils.get_pixel_scale(imagename)
        shift_ra_pix=int(round(shift_ra/pixel_scale[0]))
        shift_dec_pix=int(round(shift_dec/pixel_scale[1]))
        print (shift_ra_pix,shift_dec_pix)
        logging.info("Shift along RA in pixels: %s",shift_ra_pix)
        logging.info("Shift along DEC in pixels: %s",shift_dec_pix)
        hdu=fits.open(imagename,mode='update')
        try:
            data=hdu[0].data
            data1=np.roll(data,(shift_ra_pix,shift_dec_pix),axis=(0,1))
            hdu[0].data=data1
            hdu.flush()
        except Exception as e:
            logging.error(e)
            raise RuntimeError(e)
        finally:
            hdu.close()

class correct_shift():
    def __init__(self,solar_loc=None,\
                            vis=None,\
                            imagename=None,\
                            refcat=None,\
                            cutout_rad=None,\
                            overwrite=False):
        self.solar_loc=solar_loc
        if self.solar_loc is None:
            self.vis=vis
        self.cutout_rad=cutout_rad
        self.overwrite=overwrite
        
    @property
    def vis(self):
        return self.__vis
        
    @vis.setter
    def vis(self,value):
        if os.path.isdir(value):
            self.__vis=vis
        else:
            logging.error("MS file does not exist")
            raise IOError("MS file does not exist")
    
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
            logging.error("Image not found")
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
            logging.error("Reference Source catalogue not found")
            raise IOError ("Reference Source catalogue not found")

    def get_matched_source_list(self):
        cm=cross_match_sources.cross_match(self.imagename,self.refcat)
        cm.solar_loc=self.solar_loc
        cm.overwrite=self.overwrite
        cm.cutout_rad=self.cutout_rad
        cm.cross_match_cat()
        self.matched_cat=cm.matched_source_cat
   
    
    def get_smoothed_shift_polynomial(self,ra1=None,dec1=None,ra2=None,dec2=None,\
                    plots=False, smooth=200.0, sigcol=None, noisecol=None,\
                    SNR=10, latex=False,max_sources=None):
        '''
        This code is adapted from Natasha Hurley-Walker and Paul Hancock (fits_warp)
        https://github.com/nhurleywalker/fits_warp
        
        param ra1: column name for the ra degrees in catalogue 1 (source)
        :param dec1: column name for the dec degrees in catalogue 1 (source)
        :param ra2: column name for the ra degrees in catalogue 2 (reference)
        :param dec2: column name for the dec degrees in catalogue 2 (reference)
        :param fitsname: fitsimage upon which the pixel models will be based
        :param plots: True = Make plots
        :param smooth: smoothing radius (in pixels) for the RBF function
                       The default value is slight smaller than 300 x GLEAM pixel scale/2 arcmin 
                       300 was the default value in original fits_warp code. GLEAM pixel
                       scale is 34 arcsec. OVRO-lWA solar image default pixel scale during selfcal is
                       2 arcmin.  
        :param max_sources: Maximum number of sources to include in the construction 
                            of the warping model (defaults to None, use all sources)
        '''

        if (ra1 is None) or (dec1 is None) or \
            (ra2 is None) or (dec2 is None):
            hdu=fits.open(self.matched_cat)
            try:
                head=hdu[1].header
                ra1=head['TTYPE1']
                dec1=head['TTYPE2']
                ra2=head['TTYPE3']
                dec2=head['TTYPE4']
            except Exception as e:
                logging.error(e)
                raise e
            finally:
                hdu.close()
        
        fname=self.matched_cat
        fitsname=self.imagename
        filename, file_extension = os.path.splitext(fname)

        if file_extension == ".fits":
            with fits.open(fname) as hdu:
                raw_data=hdu[1].data
        else:
            logging.error("Only fits format is supported now")
            raise RuntimeError("Only fits format is supported now")

        # get the wcs

        hdr = fits.getheader(fitsname)
        imwcs = wcs.WCS(hdr)

        # filter the data to only include SNR>10 sources
        if sigcol is not None and noisecol is not None:
            flux_mask = np.where(raw_data[sigcol] / raw_data[noisecol] > SNR)
            data = raw_data[flux_mask]
        else:
            data = raw_data

        if max_sources is not None:
            # argsort goes in ascending order, so select from the end
            sort_idx = np.argsort(data[sigcol])[0][-max_sources:]
            data = data[sort_idx]
            print("Selected {0} brightest sources".format(max_sources))

        print("Using {0} sources to construct the pixel offset model".format(len(data)))


        num_sources=len(data[ra1])
        cat_xy = imwcs.all_world2pix(list(zip(data[ra1], data[dec1],[hdr['CRVAL3']]*num_sources,[hdr['CRVAL4']]*num_sources)), 1)
        ref_xy = imwcs.all_world2pix(list(zip(data[ra2], data[dec2],[hdr['CRVAL3']]*num_sources,[hdr['CRVAL4']]*num_sources)), 1)

        diff_xy = ref_xy - cat_xy
        
        #diff_xy=diff_xy-np.mean(diff_xy,axis=0)

        
        #dxmodel = interpolate.Rbf(
         #   cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function="linear", smooth=smooth
        #)
        self.dxmodel = RBFInterpolator(ref_xy[:,:2],diff_xy[:,0], kernel="linear", smoothing=smooth)
        
        self.dymodel = RBFInterpolator(ref_xy[:,:2], diff_xy[:, 1], kernel="linear", smoothing=smooth)

        
        logging.info("Model shift screen generated")

        if plots:
            import matplotlib

            # Super-computer-safe
            matplotlib.use("Agg")
            from matplotlib import pyplot
            from matplotlib import gridspec

            # Perceptually uniform cyclic color schemes
            try:
                import seaborn as sns

                cmap = matplotlib.colors.ListedColormap(sns.color_palette("husl", 256))
            except ImportError:
                print("seaborne not detected; using hsv color scheme")
                cmap = "hsv"
            # Attractive serif fonts
            if latex is True:
                if which("latex"):
                    try:
                        from matplotlib import rc

                        rc("text", usetex=True)
                        rc("font", **{"family": "serif", "serif": ["serif"]})
                    except:
                        print("rc not detected; using sans serif fonts")
                else:
                    print("latex not detected; using sans serif fonts")
            xmin, xmax = 0, hdr["NAXIS1"]
            ymin, ymax = 0, hdr["NAXIS2"]

            
            gx, gy = np.mgrid[
                xmin : xmax : (xmax - xmin) / 50.0, ymin : ymax : (ymax - ymin) / 50.0
            ]
            
            gx1=np.ravel(gx)
            gx2=np.ravel(gy)
            
            gx_2D=np.array([gx1,gx2]).T

            
            mdx = self.dxmodel(gx_2D)
            mdy = self.dymodel(gx_2D)
            x = cat_xy[:, 0]
            y = cat_xy[:, 1]

            # plot w.r.t. centre of image, in degrees
            try:
                delX = abs(hdr["CD1_1"])
            except:
                delX = abs(hdr["CDELT1"])
            try:
                delY = hdr["CD2_2"]
            except:
                delY = hdr["CDELT2"]
            # shift all co-ordinates and put them in degrees
            x -= hdr["NAXIS1"] / 2
            gx -= hdr["NAXIS1"] / 2
            xmin -= hdr["NAXIS1"] / 2
            xmax -= hdr["NAXIS1"] / 2
            x *= delX
            gx *= delX
            xmin *= delX
            xmax *= delX
            y -= hdr["NAXIS2"] / 2
            gy -= hdr["NAXIS2"] / 2
            ymin -= hdr["NAXIS2"] / 2
            ymax -= hdr["NAXIS2"] / 2
            y *= delY
            gy *= delY
            ymin *= delY
            ymax *= delY
            scale = 1

            dx = diff_xy[:, 0]
            dy = diff_xy[:, 1]

            fig = pyplot.figure(figsize=(12, 6))
            gs = gridspec.GridSpec(100, 100)
            gs.update(hspace=0, wspace=0)
            kwargs = {
                "angles": "xy",
                "scale_units": "xy",
                "scale": scale,
                "cmap": cmap,
                "clim": [-180, 180],
            }
            angles = np.degrees(np.arctan2(dy, dx))
            ax = fig.add_subplot(gs[0:100, 0:48])
            cax = ax.quiver(x, y, dx, dy, angles, **kwargs)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xlabel("Distance from pointing centre / degrees")
            ax.set_ylabel("Distance from pointing centre / degrees")
            ax.set_title("Source position offsets / arcsec")
            #        cbar = fig.colorbar(cax, orientation='horizontal')

            ax = fig.add_subplot(gs[0:100, 49:97])
            cax = ax.quiver(gx, gy, mdx, mdy, np.degrees(np.arctan2(mdy, mdx)), **kwargs)
            #cax = ax.quiver(x, y, dx-np.mean(dx), dy-np.mean(dy), angles, **kwargs)
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))
            ax.set_xlabel("Distance from pointing centre / degrees")
            ax.tick_params(axis="y", labelleft="off")
            #ax.set_title("Model position offsets / arcsec")
            ax.set_title("Deviation from mean offset (arcsec)")
            #        cbar = fig.colorbar(cax, orientation='vertical')
            # Color bar
            ax2 = fig.add_subplot(gs[0:100, 98:100])
            cbar3 = pyplot.colorbar(cax, cax=ax2, use_gridspec=True)
            cbar3.set_label("Angle CCW from West / degrees")  # ,labelpad=-75)
            cbar3.ax.yaxis.set_ticks_position("right")

            outname = os.path.splitext(fname)[0] + "_smooth_200pix.png"
            #        pyplot.show()
            pyplot.savefig(outname, dpi=200)
    
    def get_solar_loc_pixel(self):
        '''
        This function calculates the solar location in pixel units
        '''
        hdu=fits.open(self.matched_cat)
        try:
            hdr=hdu[1].header
        except Exception as e:
            logging.error(e)
            raise e
        finally:
            hdu.close()
        
        imwcs = wcs.WCS(hdr)
        print (self.solar_loc)
        print (self.solar_loc[0].to(u.degree).value,self.solar_loc[1].to(u.degree).value, hdr['CRVAL3'],hdr['CRVAL4'])
        self.solar_xy_pixel = (imwcs.wcs_world2pix(np.array([self.solar_loc[0].to(u.degree).value,\
                                        self.solar_loc[1].to(u.degree).value,\
                                        hdr['CRVAL3'],\
                                        hdr['CRVAL4']]).reshape((1,4)), 1).squeeze())[:2]
        self.solar_xy_pixel=self.solar_xy_pixel.reshape((1,2))
        print (self.solar_xy_pixel)  
       
    
    
                   
                    
    def correct_shift(self):
        '''
        This is a wrapper function to find the correct shifts at the location
        of the Sun. But it is not applied by this function.
        '''
        if self.solar_loc is None:
            sun_coords=utils.get_sun_pos(self.vis,str_output=False)  
            sun_ra=sun_coords['m0']
            sun_dec=sun_coords['m1']    
            self.solar_loc=[sun_ra,sun_dec]
        
        self.get_matched_source_list()
        
        self.get_smoothed_shift_polynomial(plots=True)
        
        self.get_solar_loc_pixel()
        
        self.mdx = self.dxmodel(self.solar_xy_pixel)[0]
        self.mdy = self.dymodel(self.solar_xy_pixel)[0]
        

        
        
        
        pixel_scale=utils.get_pixel_scale(self.imagename)
        
        shift_ra=self.mdx*pixel_scale[0]
        shift_dec=self.mdy*pixel_scale[1]
        
        logging.info("Shift along RA in arcsec: %s",shift_ra)
        logging.info("Shift along DEC in arcsec: %s",shift_dec)
        
        return shift_ra,shift_dec
        
        
        
        
        
        
        
        
        
        
    
