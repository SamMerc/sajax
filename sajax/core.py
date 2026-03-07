import os
import sys
import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.interpolate import interp1d
import astropy.units as u
from astropy.coordinates.matrix_utilities import rotation_matrix # for stellar inclination
import time 

class sage_class:
    
    def __init__(self, params, planet_pixel_size, wavelength,
                 flux_hot, flux_cold, 
                 spot_lat, spot_long, spot_size, ve, spotnumber, 
                 fit_ldc,
                 plot_map_wavelength, phases_rot= 0):
        # Assing input variables to as attributes of the class.
        self.params= params
        self.planet_pixel_size= planet_pixel_size
        self.wavelength= wavelength
        self.flux_hot= flux_hot
        self.flux_cold= flux_cold
        self.spot_lat= spot_lat
        self.spot_long= spot_long
        self.spot_size= spot_size
        self.ve= ve
        self.spotnumber= spotnumber
        self.fit_ldc= fit_ldc
        self.plot_map_wavelength= plot_map_wavelength
        self.phases_rot= phases_rot
        self.time_store = time.time()
        
    def rotate_star(self):

        if len(self.phases_rot) == 1:
            self.phases_rot= self.phases_rot[0] # The 0 is there to take the first entry. 
            # print('No rotation')
            lc, epsilon_wl, star_maps= self.StarSpotSpec()
            
        elif len(self.phases_rot) != 1:
            # print('Rotating the star')
            lc= []
            epsilon_wl= []
            star_maps= []
            for i, n in enumerate(self.phases_rot):
                self.phases_rot= n
                flux_norm, contamination_factor, star_map= self.StarSpotSpec()
                lc.append(flux_norm)    
                epsilon_wl.append(contamination_factor)
                star_maps.append(star_map)
            lc= lc/np.median(lc) # added to remove the peak flux 1.0 case.   #TODO discuss with ML about what's best way of normalising. clear stellar spec or median of lc.                         

        return lc, epsilon_wl, star_maps
                
        
        
    def StarSpotSpec(self):
        
        '''
        This function calculates the stellar contamination using a pixellation grid approach as presented in chakraborty et al. (in prep). 
        
        input: orbital parameters [params], planet pixel size [15-50], wavelength of input model spectrum, 
        flux_hot and flux_cold is flux model of clear and active photospheres, 
        spot_lat and spot_long to define position of spots, spot_size to define its size, 
        spot_number is the number of spots, 
        fit_ldc [n, custom, exotic, adrien, intensity_profile], plot_map_wavelength defines the wavelength at which the map is calculated
        
        '''

        if len(self.wavelength) == 1:
            wave_interp= np.zeros(2) + self.wavelength
            flux_hot_interp= np.zeros(2) + self.flux_hot
            flux_cold_interp= np.zeros(2) + self.flux_cold
            f_hot = interp1d(wave_interp, flux_hot_interp, bounds_error = False, fill_value = 0.0)
            f_cold = interp1d(wave_interp, flux_cold_interp, bounds_error = False, fill_value = 0.0)
            
        elif len(self.wavelength) >= 1:
            
            f_hot = interp1d(self.wavelength, self.flux_hot, bounds_error = False, fill_value = 0.0)
            f_cold = interp1d(self.wavelength, self.flux_cold, bounds_error = False, fill_value = 0.0)
        
        # The input parameters for grid:
        
        # for rotation phases
        phaseoff= self.phases_rot
        # for grid-size
        radiusratio = self.params[0]						
        semimajor   = self.params[1]		
        # for limb-darkening
        u1          = self.params[2]			
        u2          = self.params[3]	
        
        mu_profile= self.params[4]
        I_profile= self.params[5]
        # for stellar inclination
        inc_star= self.params[6]

        # Converting latitude to co-latitude
        spot_lat= 90 - np.asarray(self.spot_lat)
            
        # Convert input variables to system variables for further calculations:
        rs  = 1./semimajor
        rp  = radiusratio*rs
      
        # Select a grid size based on system size:
        # Creates a grid wich is twice the size of the planets' size in pixels * ratio of rs/rp
        n = (2.0*self.planet_pixel_size*(rs/rp) + 2.0*self.planet_pixel_size)
        grid = np.zeros((int(n),int(n)))
        
        star_pixel_rad = ((rs/rp) * self.planet_pixel_size)

        # Create a grid:
        x=np.arange(int(n))-(n/2.)
        y=np.arange(int(n))-(n/2.)
        x1, y1 = np.meshgrid(x,y, sparse = True)
        
        r = np.sqrt((x1**2.0) + (y1**2.0))

        ### TIME CONSUMING ###
        # Which values of the stellar grid are within the stellar (pixel) radius (find star on grid):

        x2, y2 = np.meshgrid(x/ star_pixel_rad ,y/ star_pixel_rad, sparse = False)
        starmask_rad = ((y2 >= -1.0) & (y2 <= 1.0) & (x2**2+y2**2 <= 1.0))
        
        c = 300000 #km sec^{-1}

        grid_new =  np.zeros((int(n),int(n)))
        grid_new[starmask_rad] = y2[starmask_rad] * (self.ve/ c)
        
        starmask = (r <= star_pixel_rad)
        total_pixels = len(r[starmask])      # Inside the stellar radius
        ######################
        bin_flux = []
        stellar_spec = []
        contamination_factor = []
        transit_depth = []
        
        if self.fit_ldc == 'single':       
            u1= np.zeros(len(self.wavelength))
            u2= np.zeros(len(self.wavelength))        
        elif self.fit_ldc == 'multi-color':
            u1= np.zeros(len(self.wavelength)) + self.params[2]
            u2= np.zeros(len(self.wavelength)) + self.params[3]               
        elif self.fit_ldc == 'intensity_profile':
            I_interpolated= interp1d(mu_profile[0], I_profile, bounds_error = False, fill_value = 0.0, axis=1)

        for i in range(len(self.wavelength)):
            ### TIME CONSUMING ###
            lambdaa= self.wavelength[i]

            mu = np.cos(np.arcsin(r[starmask]/star_pixel_rad))

            grid[starmask] = lambdaa
            star_grid = f_hot(grid)#/ np.max(photosHO[1][wave])
            star_grid = star_grid + (grid_new * star_grid)
            if self.fit_ldc == 'single' or self.fit_ldc == 'multi-color':           
                star_grid[starmask] = star_grid[starmask] *  (1-u1[i]*(1-mu)-u2[i]*(1-mu)**2.0)
                
            elif self.fit_ldc == 'intensity_profile':
                interpolated_intensity_prof= I_interpolated(mu)
                star_grid[starmask] = star_grid[starmask] * interpolated_intensity_prof[i] 
            ######################
            #Rotaional broadenned stellar grid
            #plt.imshow(star_grid, cmap = cm.hot , origin = 'lower', vmin = 0.0, vmax = 1.0)
            #plt.colorbar()
            #plt.title(f'Central Wavelength = {int(lambdaa)} $\AA$')
            #plt.show()
            
            star_spec = np.sum(star_grid[starmask])/ total_pixels
            stellar_spec.append(star_spec)

            if(self.spotnumber > 0.0):
                
                for sn in range(0, self.spotnumber):

                    #adding spot parameters
                    spotlong_rad = (np.pi*(self.spot_long[sn])/180.0)
                    spotlat_rad = (np.pi*(spot_lat[sn])/180.0)
                    spotsize_rad = (np.pi*(self.spot_size[sn])/180.0)

                    # Entering Cartesian cordinate system
                    sps = star_pixel_rad * np.sin(spotsize_rad)
                    spx = star_pixel_rad * np.sin(spotlong_rad) * np.sin(spotlat_rad)
                    spy = star_pixel_rad * np.cos(spotlat_rad)
                    spz = star_pixel_rad * np.cos(spotlong_rad) * np.sin(spotlat_rad)

                    spot_inCart= np.array([[spx], [spy], [spz]])
                    spx, spy, spz= stellar_rotation(active_cord= spot_inCart, phase=phaseoff) # for stellar rotation
                    
                    # Adding stellar inclination effects
                    spot_inCart= np.array([[spx], [spy], [spz]])
                    spx, spy, spz= stellar_inc(stellar_inclination= (90 - inc_star)*u.deg, active_cord=spot_inCart) # for stellar inclination

                    # Converting rotated Cartesian pixels back to GCS. 
                    spotlong_rad_rot= np.arctan(spx/spz)
                    if spz < 0: 
                        spotlong_rad_rot= spotlong_rad_rot + np.pi
                    spotlat_rad_rot= np.arccos(spy/ star_pixel_rad)

                    spot = np.zeros((int(n),int(n))) + 1.0
        
                    xpos1 = (spx-1.1*sps) 
                    xpos2 = (spx+1.1*sps)

                    ypos1 = (spy-1.1*sps)
                    ypos2 = (spy+1.1*sps)

                    if not (np.isfinite(xpos1) and np.isfinite(xpos2)) or (xpos2 <= xpos1):
                        continue 

                    xelements = np.arange(xpos1, xpos2)
                    yelements = np.arange(ypos1, ypos2)
                    #print(xelements)

                    yspot_p, xspot_p = np.meshgrid(yelements, xelements)
                    
                    xspot_p1 = xspot_p.reshape((len(xelements)*len(xelements)),)
                    yspot_p1 = yspot_p.reshape((len(yelements)*len(yelements)),)

                    zspot_p1 = np.sqrt((star_pixel_rad)**2.0 - xspot_p1**2.0 - yspot_p1**2.0)

                    # Recalculate (x,y,z)-locations to longitude and latiude (in radians):
                    longi_rad = np.arctan(xspot_p1/zspot_p1)
                    lati_rad  = np.arccos(yspot_p1/star_pixel_rad)
        
                    # Calculate absolute difference of longitudes (radians):

                    delta_lon = np.abs(spotlong_rad_rot - longi_rad)

                    # Calculate central angles (= angle between spot center and point on/in box around spot):
            
                    d_sigma = np.arccos(np.cos(spotlat_rad_rot) * np.cos(lati_rad) + 
                                        np.sin(spotlat_rad_rot) * np.sin(lati_rad) * np.cos(delta_lon))

                    # Create masks to find all angles which are <= spotsize_rad AND which are inside the star!

                    star_mask = ((xspot_p1**2.0+yspot_p1**2.0) <= star_pixel_rad**2.0)
                    inspot_mask = (d_sigma <= spotsize_rad)
            
                    grid_new_spot =  np.zeros((int(n),int(n))) + 1.0
                    grid_new_spot[(xspot_p1[star_mask & inspot_mask].astype(int) + 
                                int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + 
                                                        int(n)/2).astype(int)] = x2[(xspot_p1[star_mask & inspot_mask].astype(int) + 
                                                                                        int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + 
                                                                                                                int(n)/2).astype(int)] * (self.ve/ c)
                    ### TIME CONSUMING ###
                    # Define values for spot:
                    mu_spot = np.cos(np.arcsin(r[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)]/star_pixel_rad))
                    spot[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = lambdaa
                    spot = spot + (spot * grid_new_spot)

                    spot_grid = f_cold(spot)#/ np.max(photosHO[1][wave])

                    if self.fit_ldc == 'single' or self.fit_ldc == 'multi-color':            
                
                        spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (1-u1[i]*(1-mu_spot)-u2[i]*(1-mu_spot)**2.0)
                
                    elif self.fit_ldc == 'intensity_profile':
                        
                        interpolated_intensity_prof_spot= I_interpolated(mu_spot)
                        
                        spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (interpolated_intensity_prof_spot[i]) 
                    ####################

                    # spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = spot_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] * (1-u1[i]*(1-mu_spot)-u2[i]*(1-mu_spot)**2.0)
            
                    star_grid[(xspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int), (yspot_p1[star_mask & inspot_mask].astype(int) + int(n)/2).astype(int)] = 0.0

                    star_grid = star_grid + spot_grid            

            total_flux = np.sum(star_grid[starmask])/ total_pixels # normalised with the number of pixels. 
            bin_flux.append(total_flux)
            
            resi = (star_spec/ total_flux) # a proof for this formula is available in the paper.
            contamination_factor.append(resi)
            
            if abs(lambdaa - self.plot_map_wavelength) <= 10:
                star_map_out= star_grid     

        # calculating drop in stellar flux due to active regions.
        spotted_flux= np.sum(bin_flux)
        unspotted_flux= np.sum(stellar_spec)
        flux_norm= spotted_flux/ unspotted_flux # Normalising by unspotted stellar flux sets the peak flux to 1.0. This might not be the case on the basis of your normalisation. 
        # flux_norm= spotted_flux#/ np.median(spotted_flux)
               
        return flux_norm, contamination_factor, star_map_out

def stellar_inc(active_cord, stellar_inclination= 0.0 * u.deg):

    '''
    This function adds the effect of stellar inclination for active regions on the star. 

    Geometry: The observer is located at Z -> + np.inf. Thus, the plane of the sky is X-Y. 
    The stellar spin axis is inclined w.r.t to the y-axis. 
    So, for star_i= 90 deg. The observer in Z axis is looking at the north pole of the star. 
    While, for star_i = 0 deg. The star is spinning face-on. 

    Input: Stellar inclination [in deg] (default= 0.0 deg), 
    Cartesian cordinate of active regions (arr[x, y, z]). Be careful with the order.  

    Output: Cartesian cordinate of active regions in the inclined stellar grid. 
    '''

    rot= rotation_matrix(stellar_inclination, 'x').T # Rotation along x-axis. 

    rotated_active_cord=np.dot(rot, active_cord)
    spx= rotated_active_cord[0][0]
    spy= rotated_active_cord[1][0]
    spz= rotated_active_cord[2][0]

    return spx, spy, spz


def stellar_rotation(active_cord, phase):
    """This function rotates the stellar grid with the axis of rotation set to y-axis. 
    The observer is located at Z -> np.inf. 

    Args:
        active_cord (array): Cartesian coordinates of active regions (arr[x, y, z]) on the stellar grid.
        phase (integer): Rotational angle [in deg]

    Returns:
        [float, float, float]: Rotated corrdinates of active regions. 
    """
    rot= rotation_matrix(phase, 'y').T
    rotated_active_cord= np.dot(rot, active_cord)
    spx= rotated_active_cord[0][0]
    spy= rotated_active_cord[1][0]
    spz= rotated_active_cord[2][0]

    return spx, spy, spz