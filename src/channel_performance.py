#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module computes channel performance metrics for the points of a mobile path. 

@author: pablo belzarena
"""


import numpy as np
import copy
import fading as fad
import angles as ang
from time import perf_counter


class ChannelPerformance:
  """This class computes different performance metrics of the wireless channel"""
  def __init__(self) :
    """ChannelPerformance constructor

    """ 
    self.H = None
    """The Fourier Transform of the channel matrix. Is an array of objects. 
    Each element of the array is the FT of the channel matrix of one MS. """ 
    
  def average_rx_power(self,fading):
    """ This method given the Tx power of the radiobase computes the 
    pathloss and shadowing of the scenario and return the average received power.

    @type fading: Fading3gpp Class.
    @param fading: The fading channel model object in one point of the path.
    @return: scenario.Ptx_db - pathloss - shadowing
    """
    d3D= np.sqrt((fading.MS_pos[0] - fading.scenario.BS_pos[0] )**2+(fading.MS_pos[1] - fading.scenario.BS_pos[1] )**2+(fading.MS_pos[2] - fading.scenario.BS_pos[2] )**2)
    rxPow = fading.scenario.Ptx_db
    if fading.los:
      rxPow -= fading.scenario.get_loss_los(d3D,fading.MS_pos[2])
    else:
      rxPow -= fading.scenario.get_loss_nlos(d3D,fading.MS_pos[2])
    if (fading.scenario.shadow_enabled):
      rxPow -= fading.scenario.get_shadowing_db(fading.MS_pos,0)
    return rxPow

  def compute_snr(self,fading,freq_band,wMS,wBS,MSAntenna,BSAntenna):
    """ This method computes the snr and other intermidate performance metrics for one point in the mobile path.
    
    The method computes the pathloss, the shadowing, the Fourier transform of the channel matrix,
    the beamforming gain and other variables. Using this information computes the average snr and the
    received power spectral density.

    @type fading: Fading Class.
    @param fading: The fading channel model object in one point of the path.
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @type wMS: numpy array.
    @param wMS: the beamforming vector at the MS.
    @type wBS: numpy array.
    @param wBS: the beamforming vector at the BS. 
    @return: snr (10 * np.log10(np.sum(rxPsd.psd) /np.sum(noisePsd.psd))), rxPsd (txPsd* 10**(-pathloss-shadowing) * (np.abs(beamforming gain)**2)), the channel matrix Fourier Transform, the beamforming gain, the linear path loss, the snr taking into account only the path loss,
    the spectral efficiency of the channel, the snr taking into account pathloss and shadow.
    """
    rxPsd = copy.deepcopy(freq_band.txpsd)  
    d3D= np.sqrt((fading.MS_pos[0] - fading.scenario.BS_pos[0] )**2+(fading.MS_pos[1] - fading.scenario.BS_pos[1] )**2+(fading.MS_pos[2] - fading.scenario.BS_pos[2] )**2)
    if fading.scenario.is_los_cond(fading.MS_pos):
      ploss_db = fading.scenario.get_loss_los(d3D,fading.MS_pos[2])
    else:
      ploss_db = fading.scenario.get_loss_nlos(d3D,fading.MS_pos[2])
    ploss_linear = pow(10.0, -ploss_db / 10.0)
    ploss_linear_shadow = ploss_linear
    if (fading.scenario.shadow_enabled):
      shadow = fading.scenario.get_shadowing_db(fading.MS_pos,1)  
      ploss_db_shadow = ploss_db+ shadow
      ploss_linear_shadow = pow(10.0, -ploss_db_shadow / 10.0)
    ploss_linear = pow(10.0, -ploss_db / 10.0)
    rxPsd *= ploss_linear
    snr_pl = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    rxPsd = rxPsd/ploss_linear* ploss_linear_shadow
    snr_pl_shadow = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))

    H = self.compute_Tfourier_H(fading,freq_band,MSAntenna,BSAntenna)
    G = self.compute_beamforming_gain(fading,freq_band.n_prbs,H,wMS,wBS,MSAntenna,BSAntenna)
    rxPsd = rxPsd*(np.abs(G)**2)

    snr = 10 * np.log10(np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    spectral_eff=  np.log(1+np.sum(rxPsd) /np.sum(freq_band.noisepsd))
    return snr,rxPsd,H,G,ploss_linear,snr_pl,spectral_eff,snr_pl_shadow

  def compute_beamforming_gain(self,fading,n_prbs,H_f,wMS,wBS,MSAntenna,BSAntenna):
    """ This method given the Tx power spectral density, the Fourier Tranform of the 
    impulse response of the channel and the MS and BS beamforming vectors,
    computes the beamforming gain for each prb of the OFDM specrum.
 
    @type fading: Fading Class.
    @param fading: The fading channel model object in one point of the path.
    @type n_prbs: int.
    @param n_prbs: number of  prbs in the frequency band.
    @type H_f: 3d numpy array.
    @param H_f: the channel matrix fourier transform for each prb, each MS antenna element,
    and each BS antenna element.
    @type wMS: numpy array.
    @param wMS: the beamforming vector at the MS.
    @type wBS: numpy array.
    @param wBS: the beamforming vector at the BS. 
    @return: bemaforming Gain for each prb (wMS . H_f[prb].wBS)
    """

    n_BS = BSAntenna.get_number_of_elements()
    n_MS = MSAntenna.get_number_of_elements()
    G_f = np.zeros((n_prbs),dtype=complex)
    for prb in range(n_prbs):
      for i in range(n_MS):
        for j in range(n_BS):
         G_f[prb]= G_f[prb] + wMS[i]*H_f[prb][i][j]*wBS[j]
    return G_f

  def compute_Tfourier_H(self,fading,freq_band,MSAntenna,BSAntenna):
    """ This method computes the fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.

    @type fading: Fading3gpp Class.
    @param fading: The fast fading 3gpp channel model object in one point of the path.
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @return: The fourier transform of the channel matrix for each prb,
    each MS antenna element and each BS antenna element.
    """
    H_usn = fading.H_usn
    n_clusters = H_usn.shape[2]
    n_BS = BSAntenna.get_number_of_elements()
    n_MS = MSAntenna.get_number_of_elements()
    H_us_f = np.zeros((freq_band.n_prbs,n_MS,n_BS),dtype=complex)
    for prb in range(freq_band.n_prbs):
      f =freq_band.fc_prbs[prb]
      for i in range(n_MS):
        for j in range(n_BS):
          for k  in range(n_clusters):
            tau = -2 * np.pi * f * fading.tau[k]
            H_us_f[prb][i][j]= H_us_f[prb][i][j] + H_usn[i][j][k]* np.exp(complex(0, tau))
    return H_us_f
    
  def compute_path(self,fading,freq_band,antennaTx,antennaRx,n_mspositions,n_times,force_los,path,mode,phi_addTx=0,theta_addTx=0,phi_addRx=0,theta_addRx=0,fix_beamforming=False):               
    """ This method computes the wireless channel performance in a mobile path given by the veector of positions and times. This is the
    main method to compute the channel performnce in a path.
    
    The method computes the performance metrics (snr,rxpsd,H,G,linear_losses, distances,snr_pl,sp_eff) in mobile path.
    For each point of the path it computes the beamforming vector for the LOS direction, using the 3gpp fading model for that point. This method calls the compute_snr method to compute the performance parameters. 
    In accordance with the spatial consistency distance, the method recalculates the fading model or updates it only.  
    IMPORTANT: for multiples MS, to have spatial consistency of ssps between the MSs it is necessary to use mode =1. In this mode the fading model generates
    a grid off ssps parameters and its values are interpolated for each MS position.

    @type scenario: Scnario3GPP Class.
    @param scenario: The 3GPP scenario model.    
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @type antennaTx: Class AntennaArray3gpp
    @param antennaTx: The BS antenna array model.
    @type antennaRx: Class AntennaArray3gpp
    @param antennaRx: The MS antenna array model.
    @type n_mspositions: Array.
    @param n_mspositions: Is an array of dimension the number of MS. Each element of the array contains the positions of the MS in its path.
    @type n_times: Array.
    @param n_times: Is an array of dimension the number of MS. Each element of the array contains the simulation times in seconds corresponding to each position of the MS.
    @type force_los: int.
    @param force_los: 0 if LOS is forced to False, 1 if LOS is forced to True and 2 if LOS is computed from the scenario probability model.
    @type phi_addTx: float.
    @param phi_addTx: Additional Tx azimuth rotation angle to the LOS angle in the beamforming vector.
    @type theta_addTx: float.
    @param theta_addTx: additional Tx inclination rotation angle to the LOS angle in the beamforming vector.
    @type phi_addRx: float.
    @param phi_addRx: Additional Rx azimuth rotation angle to the LOS angle in the beamforming vector.
    @type theta_addRx: float.
    @param theta_addRx: additional Rx inclination rotation angle to the LOS angle in the beamforming vector.
    @type path: string
    @param path: The directory path to save the results of the simulation.
    @type fix_beamforming: Boolean
    @param fix_beamforming: False if the AOA and AOD for the beamforming vector is computed using the LOS between the BS and MS positions. True if AOD and AOA are fixed with phi_addTx,theta_addTx and phi_addRx, theta_addRx .
    # @return: snr (10 * np.log10(np.sum(rxPsd.psd) /np.sum(noisePsd.psd))), rxPsd (txPsd* 10**(-pathloss-shadowing) * (np.abs(beamforming gain)**2)), the channel matrix Fourier Transform, the beamforming gain, the linear path loss, the snr taking into account only the path loss,
    # the spectral efficiency of the channel, the snr taking into account pathloss and shadow.
    """
    scenario = fading.scenario
    scenario.force_los = force_los
    freq_band.compute_tx_psd (scenario.Ptx_db)
    nMSs = n_mspositions.shape[0]
    np.savetxt(path+'/nMS.csv', [nMSs], delimiter=',')
    self.aMS = antennaRx
    self.aBS = antennaTx
    #fading = fading = fad.Fading3gpp(scenario, antennaRx,antennaTx,scatters_move,move_probability,v_min_scatters,v_max_scatters)
    self.H = np.empty(shape=(nMSs),dtype = object)
    for ms in range(nMSs):
        times = n_times[ms]
        mspositions = n_mspositions[ms]
        points_in_paths = mspositions.shape[0]

        snr = np.zeros((points_in_paths))
        snr_pl = np.zeros((points_in_paths))
        sp_eff = np.zeros((points_in_paths))
        snr_pl_shadow = np.zeros((points_in_paths))
        los = np.zeros((points_in_paths))
        distances = np.zeros((points_in_paths))
        n_BS = antennaTx.get_number_of_elements()
        n_MS = antennaRx.get_number_of_elements()
        
        H = np.zeros((points_in_paths,freq_band.n_prbs,n_MS,n_BS),dtype=complex)
        G = np.zeros((points_in_paths,freq_band.n_prbs),dtype=complex)
        rxpsd = np.empty((points_in_paths,freq_band.n_prbs))
        linear_losses= np.empty((points_in_paths))
        i = 0
        pos_ini = mspositions[0]
        pos0 = mspositions[0]
        los_ant = True
        for pos in mspositions:
            if fix_beamforming:
                wBS = antennaTx.compute_phase_steering (0,np.pi/2,phi_addTx,theta_addTx)
                wMS = antennaRx.compute_phase_steering (0,np.pi/2,phi_addRx,theta_addRx)
            else:
                dAngle = ang.Angles(0,0)
                dAngle.get_angles_vectors(pos,scenario.BS_pos)
                aAngle = ang.Angles(0,0)
                aAngle.get_angles_vectors(scenario.BS_pos,pos)
                wBS = antennaTx.compute_phase_steering (dAngle.phi,dAngle.theta)
                wMS = antennaRx.compute_phase_steering (aAngle.phi,aAngle.theta)
            los_condition = scenario.is_los_cond(pos)
            distances[i] = np.sqrt(pos[0]**2+pos[1]**2)
            if i == 0:
                msvel = [0,0,0]
            else:
                dist =pos -pos_ini       
                msvel = dist/(times[i]-times[i-1]) 
            distance_ini = pos -pos0
            if los_condition:
                corr_distance = scenario.corr_ssp_LOS
            else:
                corr_distance = scenario.corr_ssp_NLOS
            if mode == 0 or mode ==1 or i == 0 or distance_ini[0] > corr_distance or distance_ini[1] > corr_distance or los_condition != los_ant:
                t0 = times[i]
                fading.compute_ch_matrix(pos,msvel,self.aMS,self.aBS,times[i],mode)
                pos_ini = pos
                pos0 = pos
            else:
                d2d = np.linalg.norm(pos_ini-scenario.BS_pos)
                d3d = d2d
                fading.update(pos,msvel,self.aMS,self.aBS,t0,times[i],d2d,d3d)
                t0 = times[i]
                pos_ini = pos
            fading.save(path,i,ms)
            los_ant = los_condition
            #G = G[i]
            los[i] = fading.scenario.is_los_cond(pos)
            snr[i],rxpsd[i],H[i],G[i],linear_losses[i],snr_pl[i],sp_eff[i],snr_pl_shadow[i]= self.compute_snr(fading,freq_band,wMS,wBS,self.aMS,self.aBS)
        
            i = i+1
        self.H[ms] = H
        # curr_time = time.localtime() 
        # hour = time.strftime("%b_%d_%Y_%H_%M_%S",curr_time)
        self.save_path(los,snr,rxpsd,H,G,linear_losses, distances,path,snr_pl,sp_eff,snr_pl_shadow,ms,times)     
        return snr,rxpsd,H,G,linear_losses, distances,snr_pl,sp_eff
        
  def compute_point(self,fading,freq_band,antennaTx,antennaRx,MSposition,MSvelocity,time,force_los,mode,phi_addTx=0,theta_addTx=0,phi_addRx=0,theta_addRx=0,fix_beamforming=False,update =False,t0=0):               
    """ This method computes the wireless channel performance in the position of a mobile with a given velocity and in a given simulation time.This is the
    main method to compute the channel performnce in a point of the path of one mobile.
    
    The method computes the performance metrics (snr,rxpsd,H,G,linear_losses, distances,snr_pl,sp_eff) in mobile position.
    For one point of the path it computes the beamforming vector for the LOS direction, using the 3gpp fading model for that point. This method calls the compute_snr method to compute the performance parameters. 
    In accordance with the spatial consistency distance, the method recalculates the fading model or updates it only.  
    
    @type scenario: Scnario3GPP Class.
    @param scenario: The 3GPP scenario model.    
    @type freq_band: FrequencyBnad Class.
    @param freq_band: The frequency band containing the transmited power spectral density and the noise power spctral density for each prb. 
    @type antennaTx: Class AntennaArray3gpp
    @param antennaTx: The BS antenna array model.
    @type antennaRx: Class AntennaArray3gpp
    @param antennaRx: The BS antenna array model.
    @type MSposition: Array.
    @param MSposition: The position of the MS in its path.
    @type time: float.
    @param time:The simulation time in seconds corresponding to the position of the MS.
    @type force_los: int.
    @param force_los: 0 if LOS is forced to False, 1 if LOS is forced to True and 2 if LOS is computed from the scenario probability model.
    @type phi_addTx: float.
    @param phi_addTx: Additional Tx azimuth rotation angle to the LOS angle in the beamforming vector.
    @type theta_addTx: float.
    @param theta_addTx: additional Tx inclination rotation angle to the LOS angle in the beamforming vector.
    @type phi_addRx: float.
    @param phi_addRx: Additional Rx azimuth rotation angle to the LOS angle in the beamforming vector.
    @type theta_addRx: float.
    @param theta_addRx: additional Rx inclination rotation angle to the LOS angle in the beamforming vector.
    @type path: string
    @param path: The directory path to save the results of the simulation.
    @type fix_beamforming: Boolean
    @param fix_beamforming: False if the AOA and AOD for the beamforming vector is computed using the LOS between the BS and MS positions. True if AOD and AOA are fixed with phi_addTx,theta_addTx and phi_addRx, theta_addRx .
    # @return: snr (10 * np.log10(np.sum(rxPsd.psd) /np.sum(noisePsd.psd))), rxPsd (txPsd* 10**(-pathloss-shadowing) * (np.abs(beamforming gain)**2)), the channel matrix Fourier Transform, the beamforming gain, the linear path loss, the snr taking into account only the path loss,
    # the spectral efficiency of the channel, the snr taking into account pathloss and shadow.
    """
   

    self.aMS = antennaRx
    self.aBS = antennaTx
    fading.scenario.force_los = force_los
    freq_band.compute_tx_psd (fading.scenario.Ptx_db)

    n_BS = antennaTx.get_number_of_elements()
    n_MS = antennaRx.get_number_of_elements()
    
    H = np.zeros((freq_band.n_prbs,n_MS,n_BS),dtype=complex)
    G = np.zeros((freq_band.n_prbs),dtype=complex)
    rxpsd = np.empty((freq_band.n_prbs))

    if fix_beamforming:
        wBS = antennaTx.compute_phase_steering (0,np.pi/2,phi_addTx,theta_addTx)
        wMS = antennaRx.compute_phase_steering (0,np.pi/2,phi_addRx,theta_addRx)
    else:
        dAngle = ang.Angles(0,0)
        dAngle.get_angles_vectors(MSposition,fading.scenario.BS_pos)
        aAngle = ang.Angles(0,0)
        aAngle.get_angles_vectors(fading.scenario.BS_pos,MSposition)
        wBS = antennaTx.compute_phase_steering (dAngle.phi,dAngle.theta)
        wMS = antennaRx.compute_phase_steering (aAngle.phi,aAngle.theta)
    # if los_condition:
    #     corr_distance = fadingscenario.corr_ssp_LOS
    # else:
    #     corr_distance = scenario.corr_ssp_NLOS
    # if mode == 0 or mode ==1 or i == 0 or distance_ini[0] > corr_distance or distance_ini[1] > corr_distance or los_condition != los_ant:
    #     t0 = times[i]
    #     fading.compute_ch_matrix(pos,msvel,times[i],mode)
    #     pos_ini = pos
    #     pos0 = pos
    # else:
    #     d2d = np.linalg.norm(pos_ini-scenario.BS_pos)
    #     d3d = d2d
    #     fading.update(pos,msvel,t0,times[i],d2d,d3d)
    #     t0 = times[i]
    #     pos_ini = pos
    # los_ant = los_condition
    #G = G[i]

    start_time = perf_counter()
    if update == False:
            fading.compute_ch_matrix(MSposition,MSvelocity,self.aMS,self.aBS,time,mode)
    else:
           d2d = np.linalg.norm(MSposition-fading.scenario.BS_pos)
           d3d = d2d
           fading.update(MSposition,MSvelocity,self.aMS,self.aBS,t0,time,d2d,d3d)
    midle_time = perf_counter()

    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.compute_snr(fading,freq_band,wMS,wBS,self.aMS,self.aBS)
    end_time = perf_counter()
    return snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow

  

  def save_path(self,los,snr,rxpsd,H,G,linear_losses, distances,path,snr_pl,sp_eff,snr_pl_shadow,ms,times):
    """ This method save to disk the performance mtrics of the path.

    @type los: 1D Array
    @param los: The LOS condition for each point of the path. Saved to path+'/los.csv'
    @type snr: 1D Array
    @param snr: The  average snr (average over the prbs) for each point of the path. Saved to path+'/snr.csv'
    @type rxpsd: 2D Array
    @param rxpsd: The recieved power spectral density for each prb and for each point of the path. Saved to path+'/rxpsd.csv'   
    @type H: 4D Array
    @param H: The Fourier Transform of the channel matrix for each prb, each TX-RX antenna pair and for each point of the path. Saved to path+'/H_f.npy'
    @type G: 2D Array
    @param G: The beamforming gain for each prb, and for each point of the path. Saved to path+'/G.csv',
    @type linear_losses: 2D Array
    @param linear_losses: The path loss in linear scale for each point of the path. Saved to path+'/linear_losses.csv'.
    @type distances: 1D Array.
    @param distances: The distance for each point in the path to the BS. Saved to path+'/positions.csv'
    @type path: string
    @param path: The directory path to save the data.
    @type snr_pl: 1D Array.
    @param snr_pl: The average snr of each point of the path using only the path losses and not using shadowing and fading. Saved to path+'/snr_pl.csv'
    @type sp_eff: 1D Array.
    @param sp_eff: The average spectral efficiency (bits/s/Hz) for each point of the path. Saved to path+'/spectral_eff.csv'
    @type snr_pl_shadow: 1D Array.
    @param snr_pl_shadow: The average snr of each point of the path using only the path losses and shadowing and not using the fast fading. Saved to path+'/snr_pl_shadow.csv'
    @type ms: int.
    @param ms: The number of the MS.
    @type times: 1D Array.
    @param times: The times of the simulation of the ms device.
    """ 

    np.savetxt(path+'/los'+str(ms)+'.csv', los, delimiter=',')
    np.savetxt(path+'/snr'+str(ms)+'.csv', snr, delimiter=',')
    np.savetxt(path+'/snr_pl'+str(ms)+'.csv', snr_pl, delimiter=',')
    np.savetxt(path+'/snr_pl_shadow'+str(ms)+'.csv', snr_pl_shadow, delimiter=',')
    np.savetxt(path+'/times'+str(ms)+'.csv', times, delimiter=',')

    np.savetxt(path+'/spectral_eff' +str(ms)+ '.csv', sp_eff, delimiter=',')
    np.savetxt(path+'/rxpsd' +str(ms)+ '.csv', rxpsd, delimiter=',')
    np.savetxt(path+'/linear_losses' +str(ms)+ '.csv', linear_losses, delimiter=',')
    np.savetxt(path+'/positions' +str(ms)+ '.csv', distances, delimiter=',')
    np.savetxt(path+'/G' +str(ms)+ '.csv', G, delimiter=',')
    
    with open(path+'/H_f' +str(ms)+ '.npy', 'wb') as f:
        np.save(f, H)
