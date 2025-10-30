###############
### Imports ###
###############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import emcee
from copy import deepcopy
from scipy.stats import norm, uniform, truncnorm
from multiprocessing import Pool
from scipy.optimize import least_squares
from scipy.stats import binned_statistic
import corner
import astropy.units as u
from astropy.timeseries import LombScargle
import arviz as az
import shutil
warnings.filterwarnings(action='ignore', category=RuntimeWarning)

# Loading sage 
import sys
import os
sys.path.append('/Users/samsonmercier/Desktop/Work/Master/2023-2024/SAGE/sage/sage_output') #Replace this with your own path
from sage import sage_class



################################
### Functions (DO NOT TOUCH) ###
################################
# defining MCMC funcs (log prior, log likelihood, log probability)
def lnprior(params,priors, args):
    lp = 0.0
    for parname in args['var_param_list']:
        pr = priors[parname]
        lp += pr.logpdf(params[parname])
    return lp

def eval_sage(params, time, args):
    # disintegrating the params to make it usable in sage. 
    #% Spots
    spot_long= []
    spot_lat=[]
    spot_size= []
    for spot_param, spot_list in zip(['lat', 'long', 'size'], [spot_lat, spot_long, spot_size]):
        for spname in args['spotnames']:
            if (f'{spname}_{spot_param}' in args['var_param_list']) or (f'{spname}_{spot_param}' in args['fix_param_list']):spot_list.append(params[f'{spname}_{spot_param}'])

    #% All others
    add_param_values = {'offset': 0.,'jitter': 0.,'Prot': 0.,'sp_ctrst': 0.}

    # Update values from args
    for key in add_param_values:
        if (key in args['var_param_list']) or (key in args['fix_param_list']):add_param_values[key] = params[key]

    # Extract final values
    offset = add_param_values['offset']
    jitt = add_param_values['jitter']
    prot = add_param_values['Prot']
    spotcontrast = add_param_values['sp_ctrst']

    # inclination= params[-1] # inclination of star
    inclination=90.

    # defining wavelength params
    wavelength= [5000]  # cal wavelength [pretty much useless]
    flux_hot=[1]        # immaculate photosphere
    flux_cold= [1 * spotcontrast]   #[0.2691] # contrast

    u1= 0.63
    u2= 0.15

    stellar_params=[0.0488,                                    # Radius-ratio   
                18.79,                                     # scaled semi-major axis 
                u1,                                        # U1
                u2,                                        # U2
                0.0,                                       # cosine of angular distance
                0.0,                                       # Intensity profile 
                inclination]                               

    planet_pixel_size= 10

    ve=(2*np.pi*0.744*696340)/(prot*24*3600)

    model_lightcurve = np.empty(len(time))

    execute_once = [[False for _ in range(args['spotnumber'])] for _ in range(args['flarenumber'])]
    for i, ti in enumerate(time):
        
        #Vary spot size if a flare step parameter has been included
        if args['flare_time_dic'] != {}:
            for fidx, flarenam in enumerate(args['flarenames']):
                for sidx, spotnam in enumerate(args['spotnames']):
                    flare_param = f'{flarenam}_{spotnam}_size'
                    if ((flare_param in args['var_param_list']) or (flare_param in args['fix_param_list'])) and (ti > args['flare_time_dic'][flare_param]) and (not execute_once[fidx][sidx]):
                        spot_size[sidx] += params[flare_param]
                        execute_once[fidx][sidx] = True

        phase_roti = ((2*np.pi)/prot) * (ti - time[0])

        star = sage_class(stellar_params, planet_pixel_size, wavelength, flux_hot, flux_cold, 
                spot_lat, spot_long, spot_size, ve, args['spotnumber'], 'multi-color', 5000, phases_rot=[np.rad2deg(phase_roti) * u.deg])

        flux_norm, _, _= star.rotate_star()

        model_lightcurve[i] = flux_norm 
    
    #Adding the offset in the event of a visit concatenation
    if args['concat_fit']:
        for iconcat in range(args['local_output_dic']['num_concat_LCs']):
            if f'LC_offset{iconcat+1}' in args['var_param_list']:
                LC_offset = params[args['var_param_list'].index(f'LC_offset{iconcat+1}')]
            elif f'LC_offset{iconcat+1}' in args['fix_param_list']:
                LC_offset = args['fix_param_values'][args['fix_param_list'].index(f'LC_offset{iconcat+1}')]
            time_mask = ((time <= args['local_output_dic']['concat_ts'][iconcat][-1]) & (time >= args['local_output_dic']['concat_ts'][iconcat][0]))
            model_lightcurve[time_mask] += LC_offset

    return model_lightcurve + offset, jitt # delete this when you are done with the code.

def ls_residual(params, time, flux, flux_err, args):

    mdl = eval_sage(params, time, args)[0]

    residual = (mdl - flux)/flux_err

    residual[(np.isnan(residual) | np.isinf(residual))] = 0.

    return residual

def lnlike(params, time, flux, flux_err, args):

    mdl, jitt = eval_sage(params, time, args)

    sigma2= flux_err**2 + mdl**2 * np.exp(2*jitt) # the corrected error in flux.

    return -0.5 * np.sum((flux - mdl)**2/sigma2 + np.log(sigma2) ) 

def lnprob(params, priors, day, flux, flux_err, args):

    #Re-arrange input parameters to be better
    p_step = {}
    for idx, var_param in enumerate(args['var_param_list']):
        p_step[var_param] = params[idx]
    for fix_param, fix_parval in zip(args['fix_param_list'], args['fix_param_values']):
        p_step[fix_param] = fix_parval

    lp = lnprior(p_step, priors, args)
    if np.isfinite(lp):
        ll = lnlike(p_step, day, flux, flux_err, args)
        if (not np.isnan(ll)):
            return lp+ll
        else:
            return -np.inf
    else:
        return -np.inf


def main():

    ###########################
    ### Defining parameters ###
    ###########################
    # Defining sectors to use and LCs for each sector 
    # syntax is sector_number : [sector_transit_number1, sector_transit_number1, ...]
    sectors_dic = {
                    1:[1,2],
                    }
    sectors = list(sectors_dic.keys())

    #Defining relevant directories
    ##Directory for output files - plots and fitting model storage
    output_dir = ''
    ##Directory for input files - your lightcurves - make then .csv files with 3 columns (time, flux, flux_err)
    input_dir = ''

    #Defining dictionary with planet information -> to remove their transits
    planet_propdic = {}
    planet_propdic['AUMicb']={}
    planet_propdic['AUMicb']['T0'] = 2458330.39051 #BJD
    planet_propdic['AUMicb']['period'] = 8.463000  #days
    planet_propdic['AUMicb']['T14'] = 3.4927/24    #days

    planet_propdic['AUMicc']={}
    planet_propdic['AUMicc']['T0'] = 2458342.2240 #BJD
    planet_propdic['AUMicc']['period'] = 18.85969  #days
    planet_propdic['AUMicc']['T14'] = 4.236/24    #days

    planet_list = list(planet_propdic.keys())

    #Defining the flare sigma-clipping thresholds
    # syntax is sector_number : [sigma clipping value for transit 1, sigma clipping value for transit 2, ...]
    sigma_dic = {
        1:[1.5, 2.5],
        }

    # Defining rolling median function
    def rolling_median(data, window):
        half_window = window // 2
        medians = np.full_like(data, np.nan)
        for i in range(half_window, len(data) - half_window):
            medians[i] = np.median(data[i - half_window:i + half_window + 1])
        return medians

    #Defining number of points to bin the LC to
    #This is necessary since only a certain number of points capture the LC behaviour
    #and using too many points makes the fitting really slow
    nbins = 500

    #Toggle to swap from individual to concatenated LC fits
    #i.e. toggle to fit all the LCs together
    concat_fit = True

    #Defining dictionary to store output from pre-processing of each LC
    output_dic = {}

    #Defining fitting related settings
    
    ##Number of cores to use
    numcores = 1
    
    ##Type of fitting method (mcmc or chi2)
    fitting_method = 'mcmc'

    ##Whether to fit the data or re-use previous fits to just make plots and post-process
    processing_method = 'use'

    #Defining least squares settings

    #Defining MCMC settings
    nsteps = 10
    nburn = 3
    nwalkers = 60

    # Defining time window of sectors to fit
    # same syntax as before, the time windows can either be 'all' i.e. fit the entire data or 
    # [time1, time2] where oyu fit only the data in that slice. times should be in whatever units the time array you passed is.
    time_window_dic = {
        'sector_1':{'LC_1':'all','LC_2':'all'},
                       }

    #Initial guesses
    #Prot in days
    #lat, long, size in degrees
    guess_dic = {
            'sector_1':{
                'LC_1':{
                    'spot1_lat': {'vary':True, 'guess':-30, 'bounds':[-50, -10]},
                    'spot2_lat': {'vary':True, 'guess':40, 'bounds':[20, 60]},
                    # 'spot3_lat': {'vary':True, 'guess':0, 'bounds':[-50, 50]},
                    'spot1_long': {'vary':True, 'guess':-100, 'bounds':[-120, -80]},
                    'spot2_long': {'vary':True, 'guess':80, 'bounds':[60, 100]},
                    # 'spot3_long': {'vary':True, 'guess':0, 'bounds':[-100, 100]},
                    'spot1_size': {'vary':False, 'guess':15, 'bounds':[1, 15]},
                    'spot2_size': {'vary':False, 'guess':25, 'bounds':[1, 15]},
                    # 'spot3_size': {'vary':True, 'guess':2, 'bounds':[1, 15]},
                    'offset': {'vary':True, 'guess':0.0187, 'bounds':[0.001, 0.1]},
                    'jitter': {'vary':True, 'guess':-10, 'bounds':[-12, 10]},
                    'Prot': {'vary':True, 'guess':'LS'},
                    'sp_ctrst': {'vary':True, 'guess':0.2,'bounds':[0.4, 0.8]},
                },
                'LC_2':{
                    'spot1_lat': {'vary':True, 'guess':-30, 'bounds':[-50, -10]},
                    'spot2_lat': {'vary':True, 'guess':40, 'bounds':[20, 60]},
                    # 'spot3_lat': {'vary':True, 'guess':0, 'bounds':[-50, 50]},
                    'spot1_long': {'vary':True, 'guess':-100, 'bounds':[-120, -80]},
                    'spot2_long': {'vary':True, 'guess':80, 'bounds':[60, 100]},
                    # 'spot3_long': {'vary':True, 'guess':0, 'bounds':[-100, 100]},
                    'spot1_size': {'vary':False, 'guess':15, 'bounds':[1, 15]},
                    'spot2_size': {'vary':False, 'guess':25, 'bounds':[1, 15]},
                    # 'spot3_size': {'vary':True, 'guess':2, 'bounds':[1, 15]},
                    'offset': {'vary':True, 'guess':0.0187, 'bounds':[0.001, 0.1]},
                    'jitter': {'vary':True, 'guess':-10, 'bounds':[-12, 10]},
                    'Prot': {'vary':True, 'guess':'LS'},
                    'sp_ctrst': {'vary':True, 'guess':0.2,'bounds':[0.4, 0.8]},
                },
            },
    }

    #Priors
    #% For analysis of individual LCs
    priors_dic = {
            'sector_1':{
                'LC_1':{
                    'spot1_lat': {'type':'uf', 'min':-80, 'max':0.},
                    'spot2_lat': {'type':'uf', 'min':0, 'max':80},
                    # 'spot3_lat': {'type':'uf', 'min':-80, 'max':80},
                    'spot1_long': {'type':'uf', 'min':-180, 'max':180},
                    'spot2_long': {'type':'uf', 'min':-180, 'max':180},
                    # 'spot3_long': {'type':'uf', 'min':-180, 'max':180},
                    # 'spot1_size': {'type':'uf', 'min':0, 'max':90},
                    # 'spot2_size': {'type':'uf', 'min':0, 'max':90},
                    # 'spot3_size': {'type':'uf', 'min':0, 'max':90},
                    'offset': {'type':'uf', 'min':-1.5, 'max':2.5},
                    'jitter': {'type':'uf', 'min':-15, 'max':15},
                    'Prot': {'type':'LS'},
                    'sp_ctrst': {'type':'uf', 'min':0., 'max':1.}
                },
                'LC_2':{
                    'spot1_lat': {'type':'uf', 'min':-80, 'max':0.},
                    'spot2_lat': {'type':'uf', 'min':0, 'max':80},
                    # 'spot3_lat': {'type':'uf', 'min':-80, 'max':80},
                    'spot1_long': {'type':'uf', 'min':-180, 'max':180},
                    'spot2_long': {'type':'uf', 'min':-180, 'max':180},
                    # 'spot3_long': {'type':'uf', 'min':-180, 'max':180},
                    # 'spot1_size': {'type':'uf', 'min':0, 'max':90},
                    # 'spot2_size': {'type':'uf', 'min':0, 'max':90},
                    # 'spot3_size': {'type':'uf', 'min':0, 'max':90},
                    'offset': {'type':'uf', 'min':-1.5, 'max':2.5},
                    'jitter': {'type':'uf', 'min':-15, 'max':15},
                    'Prot': {'type':'LS'},
                    'sp_ctrst': {'type':'uf', 'min':0., 'max':1.}
                },
            },
    }


    ######################
    ### Preparing data ###
    ######################
    #Making sure the output directory exists
    if not os.path.isdir(output_dir):os.makedirs(output_dir)
    
    #Looping over sectors
    for sector in sectors:
        #Getting sector name
        sector_name = f'sector_{sector}'
        sector_dir = output_dir+'/'+sector_name
        if not os.path.isdir(sector_dir):os.makedirs(sector_dir)
        output_dic[sector_name]={}

        #Looping over each transit
        for transitidx, transitnum in enumerate(sectors_dic[sector]):
            
            #Getting transit name
            LC_name = f'LC_{transitnum}'
            LC_dir = sector_dir+'/'+LC_name
            if not os.path.isdir(LC_dir):os.makedirs(LC_dir)
            output_dic[sector_name][LC_name]={}

            #Print
            print(f'SECTOR N.{sector}, LC N.{transitnum}')
            plt.figure(figsize=[14, 8])

            #Retrieving data
            print('RETRIEVING DATA')
            pd_data= pd.read_csv(input_dir + f'/Sector_{sector}/LC_{sector}_{transitnum}.csv')

            #Retrieving raw data
            mask_finite_flux = np.isfinite(pd_data['flux'])
            t = pd_data['time'][mask_finite_flux] + 2457000 #converting TESS BJD to BJD
            flux = pd_data['flux'][mask_finite_flux]
            flux_err = pd_data['flux_err'][mask_finite_flux]

            #Removing transit of planet considered   
            print('REMOVING TRANSIT')
            #% Defining mask to figure out the timestamps of in-transit exposures
            IT_mask = np.zeros(len(t), dtype=bool)

            #% Looping over all known planets in the system just in case
            for planet in planet_list:
                #% Getting planet properties 
                planet_dic = planet_propdic[planet]

                #% Number of transits that happened during epoch considered
                n_low  = np.round((min(t) - planet_dic['T0'])/planet_dic['period'], 0)
                n_high = np.round((max(t) - planet_dic['T0'])/planet_dic['period'], 0)
                #% Predicting transit midpoints during epoch considered
                predicted_T0s = planet_dic['T0'] + np.arange(n_low, n_high+1, 1)*planet_dic['period']

                #%Looping over all transits in epoch considered and updating mask to remove them
                for predicted_T0 in predicted_T0s:
                    IT_mask |= (t>predicted_T0-(planet_dic['T14']/2)) & (t<predicted_T0+(planet_dic['T14']/2))

            #Plotting transits before and after
            plt.errorbar(t, flux, yerr=flux_err, fmt='.', color='green', label='Transit')
            t = t[~IT_mask]
            flux = flux[~IT_mask]
            flux_err = flux_err[~IT_mask]

            #Removing flares
            print('REMOVING FLARES')

            #% compute rolling median of the LC
            rolling_med = rolling_median(flux, window=300)

            #% "Normalize" LC
            raw_norm_flux = flux/rolling_med
            norm_flux = raw_norm_flux[~np.isnan(raw_norm_flux)]

            #% Sigma clip to remove flares
            sigma=sigma_dic[sector][transitidx]
            flare_mask = ((raw_norm_flux < np.median(norm_flux) - sigma*np.std(norm_flux)) | (raw_norm_flux > np.median(norm_flux) + sigma*np.std(norm_flux)))  

            #Masking points
            plt.errorbar(t, flux, yerr=flux_err, fmt='.', color='red', label='Flares')
            t=t[~flare_mask]
            flux=flux[~flare_mask]
            flux_err=flux_err[~flare_mask]


            #Binning data to make MCMC easier
            print('BINNING DATA')
            binned_t = binned_statistic(t, t, statistic='mean', bins=nbins)[0]
            binned_flux = binned_statistic(t, flux, statistic='mean', bins=nbins)[0]
            binned_flux_err = binned_statistic(t, flux, statistic='std', bins=nbins)[0]

            #Final cleaned result
            binned_flux = binned_flux[~np.isnan(binned_t)]
            binned_flux_err = binned_flux_err[~np.isnan(binned_t)]
            binned_t = binned_t[~np.isnan(binned_t)]
            output_dic[sector_name][LC_name]['t'] = binned_t
            output_dic[sector_name][LC_name]['flux'] = binned_flux
            output_dic[sector_name][LC_name]['flux_err'] = binned_flux_err
            plt.errorbar(t, flux, yerr=flux_err, fmt='.', color='blue', label='Cleaned')
            plt.errorbar(binned_t, binned_flux, yerr=binned_flux_err, color='yellow', linestyle='-', label='Binned')        
            plt.legend()
            plt.xlabel('Time (BJD)')
            plt.ylabel('Flux')
            plt.savefig(LC_dir+'/input_LC.pdf')
            plt.close()


    ####################################
    # Swapping to one LC fit per visit #
    ####################################
    if concat_fit:
        old_output_dic = deepcopy(output_dic)
        output_dic = {}
        for sector in sectors:
            #Plot the concatenated LCs
            plt.figure(figsize=[14, 8])
            plt.title(f'Sector {sector} concatenated LCs')
            for LC in sectors_dic[sector]:
                LC_name = f'LC_{LC}'
                t = old_output_dic[f'sector_{sector}'][LC_name]['t']
                flux = old_output_dic[f'sector_{sector}'][LC_name]['flux']
                flux_err = old_output_dic[f'sector_{sector}'][LC_name]['flux_err']
                plt.errorbar(t, flux, yerr=flux_err, fmt='.', label=LC_name)
            plt.xlabel('Time (BJD)')
            plt.ylabel('Flux')
            plt.savefig(output_dir+f'/sector_{sector}/concat_LC.pdf')
            plt.close()

            #Ensure the user defined the right format for the guess and priors dictionaries
            if (len(guess_dic[f'sector_{sector}']) != 1) or (len(priors_dic[f'sector_{sector}']) != 1) or (len(time_window_dic[f'sector_{sector}']) != 1):
                raise ValueError(f"Guess, prior, and time window dictionaries must have only one LC per sector when using concatenated LCs. Found {len(guess_dic[f'sector_{sector}'])}, {len(priors_dic[f'sector_{sector}'])}, and {len(time_window_dic[f'sector_{sector}'])} respectively.")

            #Store the output directory of the concatenated LCs
            output_dic[f'sector_{sector}'] = {}
            output_dic[f'sector_{sector}']['LC_1'] = {}
            output_dic[f'sector_{sector}']['LC_1']['num_concat_LCs'] = len(sectors_dic[sector])
            output_dic[f'sector_{sector}']['LC_1']['concat_ts'] = [old_output_dic[f'sector_{sector}'][f'LC_{i+1}']['t'] for i in range(output_dic[f'sector_{sector}']['LC_1']['num_concat_LCs'])]
            for entry in ['t', 'flux', 'flux_err']:
                output_dic[f'sector_{sector}'][f'LC_1'][entry] = np.concatenate([old_output_dic[f'sector_{sector}'][f'LC_{i+1}'][entry] for i in range(output_dic[f'sector_{sector}']['LC_1']['num_concat_LCs'])])

            sectors_dic[sector] = [1] #Only one LC per sector

            #Delete the old directory
            for subdir in os.listdir(output_dir+f'/sector_{sector}'):
                if ('LC_' in subdir) and (subdir != 'LC_1'):shutil.rmtree(output_dir+f'/sector_{sector}/'+subdir)


    ####################
    ### Fitting data ###
    ####################
    for sector in sectors:
        sector_name = f'sector_{sector}'
        sector_dir = output_dir+'/'+sector_name
        #Looping over each transit
        for transitidx, transitnum in enumerate(sectors_dic[sector]):
            LC_name = f'LC_{transitnum}'
            LC_dir = sector_dir+'/'+LC_name
            print(f'SECTOR N.{sector}, LC N.{transitnum}')

            #Loading data
            local_guess_dic = guess_dic[sector_name][LC_name].copy()
            local_priors_dic = priors_dic[sector_name][LC_name].copy()
            local_output_dic = output_dic[sector_name][LC_name].copy()

            #Inirtializing lists
            spot_params=[]
            spot_priors= {}
            spot_labels= []

            #Finding the number of spots
            spotnames = []
            for key in local_guess_dic.keys():
                if ('spot' in key) and ('flare' not in key):spotnames.append(key.split('_')[0])
            spotnumber = len(np.unique(spotnames))

            #Finding the number of flares
            flarenames = []
            for key in local_guess_dic.keys():
                if ('flare' in key):flarenames.append(key.split('_')[0])
            flarenumber = len(np.unique(flarenames))

            #Parameter check
            for param in local_guess_dic.keys():
                if (local_guess_dic[param]['vary']) and (param not in local_priors_dic.keys()):
                    raise ValueError(f"{param} is free and must therefore be in priors_dic.")
                if (not local_guess_dic[param]['vary']) and (param in local_priors_dic.keys()):
                    raise ValueError(f"{param} is fixed and must therefore not be in priors_dic.")
            
            #Initiallizing important lists
            var_param_list = [key for key in local_guess_dic.keys() if local_guess_dic[key]['vary']]
            fix_param_list = [key for key in local_guess_dic.keys() if not local_guess_dic[key]['vary']]
            fix_param_values = [local_guess_dic[key]['guess'] for key in fix_param_list]
            add_args={}
            add_args['var_param_list'] = var_param_list
            add_args['ndim'] = len(var_param_list)
            add_args['fix_param_list'] = fix_param_list
            add_args['fix_param_values'] = fix_param_values
            add_args['spotnumber'] = spotnumber
            add_args['spotnames'] = np.unique(spotnames)
            add_args['flarenumber'] = flarenumber
            add_args['flarenames'] = np.unique(flarenames)
            add_args['concat_fit'] = concat_fit
            add_args['local_output_dic'] = local_output_dic
                
            #Prot specific check
            if ('Prot' not in var_param_list) and ('Prot' not in fix_param_list):
                raise ValueError('Rotation period must be fixed or free to continue')
            
            #Spot size and contrast warning
            for param in var_param_list:
                if ('sp_ctrst' in var_param_list) and ('_size' in param):
                    print('WARNING: Spot size and contrast are both free parameters. This is not recommended.')
                    break
            
            #Retrieving spot properties
            for param in ['lat', 'long', 'size']:
                for num in range(spotnumber):
                    propname = f'spot{num+1}_{param}'
                    if propname in var_param_list:
                        #Adding parameter
                        spot_params.extend([local_guess_dic[propname]['guess']])
                        #Adding priors
                        if local_priors_dic[propname]['type'] == 'uf':
                            spot_priors.update({propname : uniform(loc=local_priors_dic[propname]['min'], scale=local_priors_dic[propname]['max']-local_priors_dic[propname]['min'])})
                        elif local_priors_dic[propname]['type'] == 'gauss':
                            spot_priors.update({propname : norm(loc=local_priors_dic[propname]['val'], scale=local_priors_dic[propname]['s_val'])})
                        else:raise ValueError(f"Prior type {local_priors_dic[propname]['type']} not recognized.")
                        #Adding labels
                        spot_labels.extend([propname])

            #Retrieving flare properties
            add_args['flare_time_dic']={}
            for flarenam in add_args['flarenames']:
                for spotnam in add_args['spotnames']:
                    propname = f'{flarenam}_{spotnam}_size'
                    if propname in var_param_list:
                        spot_params.extend([local_guess_dic[propname]['guess']])
                        #Adding priors
                        if local_priors_dic[propname]['type'] == 'uf':
                            spot_priors.update({propname : uniform(loc=local_priors_dic[propname]['min'], scale=local_priors_dic[propname]['max']-local_priors_dic[propname]['min'])})
                        elif local_priors_dic[propname]['type'] == 'gauss':
                            spot_priors.update({propname : norm(loc=local_priors_dic[propname]['val'], scale=local_priors_dic[propname]['s_val'])})
                        else:raise ValueError(f"Prior type {local_priors_dic[propname]['type']} not recognized.")
                        #Adding labels
                        spot_labels.extend([propname])

                    #Retrieve the timing - needed whether we have fixed or fitted properties
                    if (propname in var_param_list) or (propname in fix_param_list):
                        add_args['flare_time_dic'][propname] = local_guess_dic[propname]['timing']

            #Compute Lomb-Scargle periodogram - only if we fit period and use LS to initialize it
            min_period = 0.5
            max_period = 10.0
            frequency = np.linspace(1/max_period, 1/min_period, 10000)
            ls = LombScargle(local_output_dic['t'], local_output_dic['flux'])
            power = ls.power(frequency)
            best_frequency = frequency[np.argmax(power)]
            best_period = 1 / best_frequency
            if ('Prot' in fix_param_list) and (local_guess_dic['Prot']['guess'] == 'LS'):
                add_args['fix_param_values'][fix_param_list.index('Prot')]=best_period

            #Checking there is the correct number of offsets for the concatenated fit
            if concat_fit:
                var_LC_offset_list = [param for param in var_param_list if 'LC_offset' in param]
                fix_LC_offset_list = [param for param in fix_param_list if 'LC_offset' in param]
                if (len(var_LC_offset_list) + len(fix_LC_offset_list) != local_output_dic['num_concat_LCs']):
                    raise ValueError(f'When fitting concatenated LCs, LC offsets must be equal to number of LCs concatenated. Found {len(var_LC_offset_list) + len(fix_LC_offset_list)} LC offsets but {local_output_dic["num_concat_LCs"]} LCs.')
            else:
                var_LC_offset_list = []
                fix_LC_offset_list = []

            #Checking that there aren't too many offsets
            if concat_fit:
                if ('offset' in var_param_list) or ('offset' in fix_param_list and fix_param_values[fix_param_list.index('offset')] != 0.):
                    print('Careful, you might be adding too many offsets in your fit.')

            params= spot_params
            priors = spot_priors
            for param in ['offset', 'jitter', 'Prot', 'sp_ctrst'] + var_LC_offset_list:
                if param in var_param_list:
                    if (param == 'Prot') and (local_guess_dic['Prot']['guess'] == 'LS') and (local_priors_dic['Prot']['type'] == 'LS'):
                        params.append(best_period)
                        priors.update({'Prot' : norm(loc=best_period, scale=.2)})
                    else:
                        params.append(local_guess_dic[param]['guess'])
                        if local_priors_dic[param]['type'] == 'uf':
                            priors.update({param : uniform(local_priors_dic[param]['min'], local_priors_dic[param]['max']-local_priors_dic[param]['min'])})
                        elif local_priors_dic[param]['type'] == 'gauss':
                            priors.update({param : norm(loc=local_priors_dic[param]['val'], scale=local_priors_dic[param]['s_val'])})
                        else:raise ValueError(f"Prior type {local_priors_dic[param]['type']} not recognized.")
                    spot_labels.append(param)

            #Restricting the fit data to the time window defined
            if time_window_dic[sector_name][LC_name] != 'all':
                print('DEFINING TIME WINDOWS')
                time_window = time_window_dic[sector_name][LC_name]
                if type(time_window) == str:
                    if time_window == 'rot':
                        start_time = local_output_dic['t'][0]
                        end_time = start_time + best_period
                    else:raise KeyError(f'Time window {time_window} not recognized. Must be "all","rot", a tuple with start and end time, or a sliding window list with 3 elements (start, end, step).')
                elif type(time_window) == tuple:
                    start_time, end_time = time_window
                elif type(time_window) == list:
                    if len(time_window) == 3:
                        start_time, end_time, step_time = time_window
                    else:raise ValueError(f'Time window {time_window} not recognized. Must be "all","rot", a tuple with start and end time, or a sliding window list with 3 elements (start, end, step).')
                else:raise KeyError(f'Time window {time_window} not recognized. Must be "all","rot", a tuple with start and end time, or a sliding window list with 3 elements (start, end, step).')
                
                if (type(time_window) == str) or (type(time_window) == tuple):
                    sliding_masks = [(local_output_dic['t'] >= start_time) & (local_output_dic['t'] <= end_time)]
                elif type(time_window) == list:
                    sliding_masks = []
                    loc_start_time = start_time
                    loc_end_time = end_time
                    while loc_end_time < local_output_dic['t'][-1]:
                        print(f'Adding sliding mask from {loc_start_time} to {loc_end_time}')
                        sliding_masks.append((local_output_dic['t'] >= loc_start_time) & (local_output_dic['t'] <= loc_end_time))
                        #Updating time window
                        loc_start_time += step_time
                        loc_end_time += step_time
            else:
                print('NO TIME WINDOW DEFINED, USING ALL DATA')
                sliding_masks = [(np.ones(len(local_output_dic['t']), dtype=bool))]

            #Looping over sliding masks i.e. looping over time windows to fit
            for imask, sliding_mask in enumerate(sliding_masks):
                
                print(f'RESTRICTING TIME WINDOW USING MASK {imask+1}/{len(sliding_masks)}')
                #Restricting time window
                local_output_dic['t'] = np.copy(output_dic[sector_name][LC_name]['t'])[sliding_mask]
                local_output_dic['flux'] = np.copy(output_dic[sector_name][LC_name]['flux'])[sliding_mask]
                local_output_dic['flux_err'] = np.copy(output_dic[sector_name][LC_name]['flux_err'])[sliding_mask]
                print('Number of points after and before time window restriction:', len(local_output_dic['t']),'/', len(output_dic[sector_name][LC_name]['t']))

                #Defining mask specific directories for output of fitting results
                if len(sliding_masks) > 1:
                    mask_dir = LC_dir+'/'+f'mask_{imask+1}'
                    if not os.path.isdir(mask_dir):os.makedirs(mask_dir)
                else:mask_dir = LC_dir

                #Actually fitting the data
                print('FITTING')
                if fitting_method == 'ls':
                    print('Fitting with least squares')
                    # Run least-squares optimization
                    result = least_squares(ls_residual, params, args=(local_output_dic['t'],local_output_dic['flux'],local_output_dic['flux_err'], add_args))

                    # Best-fit parameters
                    ls_best_fit = result.x
                    print("Best-fit parameters:")
                    print('\n')
                    for best_par_val, best_par_name in zip(ls_best_fit, add_args['var_param_list']):
                        print(f'{best_par_name} : {best_par_val}')
                        print('\n')

                elif fitting_method == 'mcmc':
                    if processing_method == 'use': 
                        print('Fitting with MCMC')
                        print('DEFINING PRIORS')
                        pos = np.zeros((nwalkers, add_args['ndim']), dtype=float)
                        for i in range(add_args['ndim']):
                            if var_param_list[i]=='Prot':pos[:, i] = np.random.normal(loc=best_period, scale=1., size=nwalkers)
                            else:pos[:, i] = np.random.uniform(low=local_guess_dic[var_param_list[i]]['bounds'][0], high=local_guess_dic[var_param_list[i]]['bounds'][1], size=nwalkers)
                        for position in pos:
                            test_position = {}
                            for varidx, varparam in enumerate(add_args['var_param_list']):
                                test_position[varparam] = position[varidx]
                            assert np.isfinite(lnprior(test_position,priors,add_args)), f"lnprior of parameters: {test_position} is not finite"

                        fig,ax = plt.subplots(1,add_args['ndim'], figsize=(15,5))
                        ax = ax.reshape(-1)

                        print('STARTING MCMC')
                        if numcores > 1:
                            pool_proc = Pool(processes=numcores)
                            sampler = emcee.EnsembleSampler(nwalkers, 
                                                            add_args['ndim'], 
                                                            lnprob, 
                                                            args=(priors,local_output_dic['t'],local_output_dic['flux'],local_output_dic['flux_err'], add_args),
                                                            pool = pool_proc)
                            sampler.run_mcmc(pos, nsteps, progress=True)
                            pool_proc.close()
                            pool_proc.join()

                        else:
                            sampler = emcee.EnsembleSampler(nwalkers, 
                                                            add_args['ndim'], 
                                                            lnprob, 
                                                            args=(priors,local_output_dic['t'],local_output_dic['flux'],local_output_dic['flux_err'], add_args))
                            sampler.run_mcmc(pos, nsteps, progress=True)

                        raw_chain=sampler.chain
                        logprob=sampler.get_log_prob()

                        # Storing chains
                        np.save(mask_dir+'/chains.npy', raw_chain)
                        np.save(mask_dir+'/logprob.npy', logprob)

                    elif processing_method == 'reuse':
                        print('Retrieving MCMC chains')
                        raw_chain = np.load(mask_dir+'/chains.npy')
                        logprob = np.load(mask_dir+'/logprob.npy')
                        old_nwalkers, old_nsteps, old_ndim = raw_chain.shape
                        for old_val, new_val, name in zip([old_nwalkers, old_nsteps, old_ndim],[nwalkers, nsteps, add_args['ndim']],['walkers','steps','dimensions']):
                            if old_val != new_val:
                                raise ValueError(f'Incoherent number of {name}. Old is {old_val}, new is {new_val}. Cannot go further.')

                    # Plotting each chain
                    print('Chain plots')
                    del_chain = []
                    for idx, label in enumerate(spot_labels):
                        #Initialize plot
                        fig, ax = plt.subplots(figsize=[14, 4])
                        #Loop over walkers
                        param_median = np.median(raw_chain[:,:,idx])
                        param_std = np.std(raw_chain[:,:,idx])
                        for iwalk in range(nwalkers):
                            if (np.abs(raw_chain[iwalk, -1, idx] - param_median) > 2*param_std):
                                del_chain.append(iwalk)
                                ax.plot(np.arange(nsteps), raw_chain[iwalk, :, idx], alpha=0.5, color="red")
                            else:
                                ax.plot(np.arange(nburn), raw_chain[iwalk, :nburn, idx], alpha=0.5, color="red")
                                ax.plot(np.arange(nburn, nsteps), raw_chain[iwalk, nburn:, idx], alpha=0.5, color="blue")
                        
                        ax.axhline(np.median(raw_chain[:, :, idx]), color='black', linestyle='dashed')
                        HDI=az.hdi(raw_chain[:, :, idx].flatten(), hdi_prob=.68)
                        for val in HDI:ax.axhline(val, color='black', linestyle='dotted')

                        ax.set_title(f"MCMC Chains for {label}")
                        ax.set_xlabel("Step")
                        ax.set_ylabel(label)
                        plt.savefig(mask_dir+'/Chain_'+str(label)+'.pdf')
                        plt.close()
                    
                    #% Removing chains
                    del_chain = np.unique(del_chain)
                    print(f"Removing {len(del_chain)} walkers from the chain due to large deviations.")
                    if len(del_chain) > 0:
                        chain = np.delete(raw_chain, del_chain, axis=0)
                    else:chain = raw_chain

                    #Burning and merging chains
                    chain = chain[:, nburn:, :]
                    chain = np.reshape(chain, (chain.shape[0]*chain.shape[1], chain.shape[2]))

                    # Plotting the corner plot
                    print('Corner plot')
                    fig = corner.corner(chain, labels=spot_labels)
                    plt.savefig(mask_dir+'/corner.pdf')
                    plt.close()
                
                else:
                    raise ValueError(f"Fitting method {fitting_method} not recognized. Please use 'ls' or 'mcmc'.")
                
                # Plotting the best fit
                #Retrieving highest probability parameters for MCMC
                print('Best fit plot')

                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[3,1]}, figsize=[14, 4])
                ax1.errorbar(local_output_dic['t'], local_output_dic['flux'], yerr=local_output_dic['flux_err'], fmt='.', color='blue', label='Data', alpha=0.5)
                if fitting_method == 'mcmc':
                    #Highest logprob
                    max_step, max_walker = np.unravel_index(np.argmax(logprob), logprob.shape)
                    best_fit_params = raw_chain[max_walker, max_step, :]

                    # Median             
                    median_params = np.median(chain, axis=0)
                    median_dic = {varparname:varparval for varparname, varparval in zip(add_args['var_param_list'], median_params)}
                    median_dic.update({fixparname:fixparval for fixparname, fixparval in zip(add_args['fix_param_list'], add_args['fix_param_values'])})
                    median_res = eval_sage(median_dic, local_output_dic['t'], add_args)[0]
                    ax1.plot(local_output_dic['t'], median_res, color='yellow', label='Median Fit')
                    ax2.errorbar(local_output_dic['t'], (local_output_dic['flux'] - median_res) * 1e6, yerr=local_output_dic['flux_err'], fmt='.', color='yellow')
                
                elif fitting_method == 'ls':
                    best_fit_params = ls_best_fit
                best_fit_dic = {varparname:varparval for varparname, varparval in zip(add_args['var_param_list'], best_fit_params)}
                best_fit_dic.update({fixparname:fixparval for fixparname, fixparval in zip(add_args['fix_param_list'], add_args['fix_param_values'])})
                bestfit_res = eval_sage(best_fit_dic, local_output_dic['t'], add_args)[0]
                ax1.plot(local_output_dic['t'], bestfit_res, color='orange', label='Best Fit')
                ax2.errorbar(local_output_dic['t'], (local_output_dic['flux'] - bestfit_res) * 1e6, yerr=local_output_dic['flux_err'], fmt='.', color='orange')
                if add_args['flare_time_dic'] != {}:
                    for key in add_args['flare_time_dic']:
                        ax1.axvline(add_args['flare_time_dic'][key], color='black', linestyle='dashed', label=key)
                ax1.set_ylabel("Flux")
                ax2.set_xlabel("Time (BJD)")
                ax2.set_ylabel("Residuals (ppm)")
                ax1.legend()
                plt.tight_layout()
                plt.savefig(mask_dir+'/best_fit.pdf')
                plt.close()

if __name__ == "__main__":
    main()