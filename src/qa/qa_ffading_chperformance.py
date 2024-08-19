#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the fast_fading and channel_performance modules.

@author: pablobelzarena
"""

# qa_angles.py : unittest for angles.py
##### to run from the project directory
import sys,os
sys.path.append('./src')
sys.path.append('./src/gui')
sys.path.append('./src/graph')

###### to run from qa directory
sys.path.append('../')
sys.path.append('../gui')
sys.path.append('../graph')

import unittest
import scenarios as sc
import fading as fad
import antennas as antennas
import channel_performance as cp
import frequency_band as fb
import numpy as np
import matplotlib.pyplot as plt
import datetime



class FadingPerformanceTest(unittest.TestCase):
  """Unitest class for testing fading and channel performance.
  
    It is very difficult to test the fading module because all short scale parameters are build
    using random variables. Therefore, the channel matrix coefficients depend on the realization of
    many rvs like: clusters delays, clusters powers, clusters arrival and departure angles,
    cross polarizations and intial phases. For testing this module we override these random variables
    with deterministics values and we test the buiding of the channel matrix for these known scenarios.
    We also test the performance results using the channel_performance methods that are the interface( like an
    API) to access the results of the wireless channel simulations.

  """   
  def test_SISORayleighFading(self):
    fcGHz = 1
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 10
    BS_pos = [0,0,0]
    Ptx_db = 10
    sigma_shadow=5
    shadow_corr_distance=5
    pos_ini =[0,0,0]
    order=2
    scf=  sc.ScenarioSimpleLossModel(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db,order,sigma_shadow,shadow_corr_distance)
    MS_pos = [0,0,0]    
    MS_vel = [10,0,0]
        ######## Build the receive and transmit antenna for testing 
    nMS=1
    nBS = 1
    aeBS = antennas.Antenna3gpp3D(8)
    self.aBS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nBS, 0, 0, 0, aeBS, 1)
    """ Base station antenna array""" 
    aeMS  = antennas.AntennaIsotropic(8)
    self.aMS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nMS, 0, 0, 0, aeMS, 1)

    fading = fad.FadingSiSoRayleigh(scf,10)
    iterations = 100000
    z = np.zeros(iterations,dtype=complex)
    t= 0
    for i in range(iterations):
        t = t+0.001
        z[i] = fading.compute_ch_matrix(MS_pos,MS_vel,self.aMS,self.aBS,t)
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag) # convert to dB
    # Plot fading over time
    plt.plot(z_mag_dB)
    plt.show()
    count, bins_count = np.histogram(z_mag, bins=10)
  
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
  
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
  
    # plotting PDF and CDF
    #plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()
  
  def test_SISORicianFading(self):
    fcGHz = 1
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 10
    BS_pos = [0,0,0]
    Ptx_db = 10
    sigma_shadow=5
    shadow_corr_distance=5
    pos_ini =[0,0,0]
    order=2
    scf=  sc.ScenarioSimpleLossModel(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db,order,sigma_shadow,shadow_corr_distance)
    MS_pos = [0,0,0]    
    MS_vel = [10,0,0]
        ######## Build the receive and transmit antenna for testing 
    nMS=1
    nBS = 1
    aeBS = antennas.Antenna3gpp3D(8)
    self.aBS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nBS, 0, 0, 0, aeBS, 1)
    """ Base station antenna array""" 
    aeMS  = antennas.AntennaIsotropic(8)
    self.aMS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nMS, 0, 0, 0, aeMS, 1)

    fading = fad.FadingSiSoRician(scf,10,10)
    iterations = 10000
    z = np.zeros(iterations,dtype=complex)
    t= 0
    for i in range(iterations):
        t = t+0.001
        z[i] = fading.compute_ch_matrix(MS_pos,MS_vel,self.aMS,self.aBS,t)
    z_mag = np.abs(z) # take magnitude for the sake of plotting
    z_mag_dB = 10*np.log10(z_mag) # convert to dB
    # Plot fading over time
    plt.plot(z_mag_dB)
    plt.show()
    count, bins_count = np.histogram(z_mag, bins=10)
  
    # finding the PDF of the histogram using count values
    pdf = count / sum(count)
  
    # using numpy np.cumsum to calculate the CDF
    # We can also find using the PDF values by looping and adding
    cdf = np.cumsum(pdf)
  
    # plotting PDF and CDF
    #plt.plot(bins_count[1:], pdf, color="red", label="PDF")
    plt.plot(bins_count[1:], cdf, label="CDF")
    plt.legend()
    plt.show()
  
     #z/sqrt(k+1)+sqrt(K/(K+1)) e**j( 2 pi fd_LOS t + tita_LOS)  
 
    # iterations = 1000000
    # res = np.zeros(iterations)
    # res[0] = scf.get_shadowing_db(MS_pos,0)
    # for i in range(1,iterations):
    #     MS_pos[0]= MS_pos[0]+1
    #     res[i] = scf.get_shadowing_db(MS_pos,0)
    
  def config(self):
    """ This method configures the antennas, scenario, etc. 
    
    
    """ 

    ######## Build the receive and transmit antenna for testing 
    nMS=1
    nBS = 1
    aeBS = antennas.Antenna3gpp3D(8)
    self.aBS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nBS, 0, 0, 0, aeBS, 1)
    """ Base station antenna array""" 
    aeMS  = antennas.AntennaIsotropic(8)
    self.aMS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nMS, 0, 0, 0, aeMS, 1)
    """ Ms antenna array""" 
    #print(self.aBS.get_number_of_elements(), " numero de elementos "    )
     ###########################################################
    
    ######## Build the scenario for testing
    self.fcGHz = 30
    """ Scenario frequency in GHz""" 
    posx_min = -1000
    posx_max = 1000
    posy_min = -1000
    posy_max = 1000
    grid_number = 20
    BS_pos = np.array([0,0,20])
    Ptx_db = 30

    sigma_shadow=5
    shadow_corr_distance=5  
    order = 2
    self.LOS = True
    self.scf=  sc.ScenarioSimpleLossModel(self.fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db,order,sigma_shadow,shadow_corr_distance)
    """ Scenario for test Simple model of Losses """ 
    #self.scf=  sc.Scenario3GPPUmi(self.fcGHz, posx_min,posx_max, posy_min, posy_max, grid_number, BS_pos, Ptx_db,True,self.LOS)

    #####################################################
    
    ########## Build the OFDM frequency band for testing
    self.freq_band =  fb.FrequencyBand(fcGHz=self.fcGHz,number_prbs=81,bw_prb=10000000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    """ OFDM frequency band for test""" 
    self.freq_band.compute_tx_psd(tx_power_dbm=30)
    ###################################################
    
    #####################################################
    
    #### Build the channel performance object to get the results after the simulation
    self.performance  = cp.ChannelPerformance()
    """ Channel performance object"""
    #####################################################
  def test_SISO(self):   
    """ This method tests. 
    
    Hay que entender los saltos..... y ver si sortear una sin diferente es razonable....
    """ 
    MS_pos = [10,0,2]    
    MS_vel = [100,0,0]
    fading = fad.FadingSiSoRayleigh(self.scf,10)
    #fading = fad.Fading3gpp(self.scf)

    iterations = 1000
    snr = np.zeros(iterations)
    snr_pl = np.zeros(iterations)
    snr_pl_shadow= np.zeros(iterations)
    sp_eff = np.zeros(iterations)
    t = 0
    force_los =2
    mode = 0
    mode = 1 #for 3gpp
    #dt = datetime.datetime.now()
    #print(dt)
    for i in range(iterations):
        t = 0.001
        snr[i],rxpsd,H,G,linear_losses,snr_pl[i],sp_eff[i],snr_pl_shadow[i] = self.performance.compute_point(fading,self.freq_band,self.aBS,self.aMS,MS_pos,MS_vel,t,force_los,mode)               
        MS_pos[0]= MS_pos[0]+MS_vel[0] * t
        MS_pos[1]= MS_pos[1]+MS_vel[1] * t
        MS_pos[2]= MS_pos[2]+MS_vel[2] * t
    #dt1 = datetime.datetime.now()
    #print(dt1)
    #print(dt-dt1)

    MS_pos = [500,0,2]    
    MS_vel = [-1000,0,0] 
    snr2 = np.zeros(iterations)  
    snr_pl2 = np.zeros(iterations)
    snr_pl_shadow2 = np.zeros(iterations)
    sp_eff2 = np.zeros(iterations)

    t = 0
    force_los =2
    mode = 0
    
    for i in range(iterations):
        t = 0.001
        snr2[i],rxpsd,H,G,linear_losses,snr_pl2[i],sp_eff2[i],snr_pl_shadow2[i] = self.performance.compute_point(fading,self.freq_band,self.aBS,self.aMS,MS_pos,MS_vel,t,force_los,mode)               
        MS_pos[0]= MS_pos[0]+MS_vel[0] * t
        MS_pos[1]= MS_pos[1]+MS_vel[1] * t
        MS_pos[2]= MS_pos[2]+MS_vel[2] * t 

    fig, ax = plt.subplots()
    ax.plot(snr,'--b',label = "snr")
    ax.plot(snr_pl,'r',label="snr_pl")
    ax.plot(snr_pl_shadow,'g',label="snr_pl_shadow")
    ax.plot(snr2,'--c',label = "snr2")
    ax.plot(snr_pl_shadow2,'--y',label = "snr2 shadow")


    leg = ax.legend()
    plt.show()

    
    
    #print( "Resultados : " , snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow)
      
  
    
  def config3gpp(self):
    """ This method configures the antennas, scenario, etc. 
    
    
    """ 

    ######## Build the receive and transmit antenna for testing 
    nMS=2
    nBS = 2
    aeBS = antennas.Antenna3gpp3D(8)
    self.aBS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nBS, 0, 0, 0, aeBS, 1)
    """ Base station antenna array""" 
    aeMS  = antennas.AntennaIsotropic(8)
    self.aMS  = antennas.AntennaArray3gpp(0.5, 0.5, 1, nMS, 0, 0, 0, aeMS, 1)
    """ Ms antenna array""" 
    ###########################################################
    
    ######## Build the scenario for testing
    self.fcGHz = 30
    """ Scenario frequency in GHz""" 
    posx_min = -1000
    posx_max = 1000
    posy_min = -1000
    posy_max = 1000
    grid_number = 20
    BS_pos = np.array([0,0,20])
    Ptx_db = 30
    self.MS_pos = np.array([100.00,0,20])
    """ MS position""" 
    self.MS_vel = np.array([100,100,0])
    """ MS velocity""" 
    self.LOS = 0
    """ LOS condition""" 

    self.scf=  sc.Scenario3GPPUmi(self.fcGHz, posx_min,posx_max, posy_min, posy_max, grid_number, BS_pos, Ptx_db,True,self.LOS)
    """ Scenario for test 3gppUmi""" 

    #####################################################
    
    ########## Build the OFDM frequency band for testing
    self.freq_band =  fb.FrequencyBand(fcGHz=self.fcGHz,number_prbs=81,bw_prb=10000000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    """ OFDM frequency band for test""" 
    self.freq_band.compute_tx_psd(tx_power_dbm=30)
    ###################################################
    
    ###### Build the steering vectors for beamforming according the the BS and MS positions
    self.wBS = self.aBS.compute_phase_steering (0,np.pi/2,0,0)
    """ BS steering vector for baemforming""" 
    self.wMS = self.aMS.compute_phase_steering (np.pi,np.pi/2,0,0)
    """ MS steering vector for baemforming"""
    #####################################################
    
    #### Build the channel performance object to get the results after the simulation
    self.performance  = cp.ChannelPerformance()
    """ Channel performance object"""
    #####################################################
  def test_NLOS_1cluster(self):   
    """ This method tests for a NLOS condition with only one cluster with all power and other controlled ssp, the generation of the channel matrix coefficients, the channel matrix Fourier Transform, and the channel performance. 
    
    
    """ 

    ####### Build the fading object. This object generates all random short scale parameters
    ####### and computes the channel coefficients. This call checks that all method in fading
    ####### modules are executed without errors
    ##### First set the LOS condition of the scenario
    self.LOS = 0 ### Condition LOS
    self.scf.force_los = self.LOS
    fading = fad.Fading3gpp(self.scf)
    fading.compute_ch_matrix(self.MS_pos,self.MS_vel, self.aMS,self.aBS,0,mode=0)

    ############################################################
    
    ##### In the following instructions we override the short scale parameters and 
    ##### we compute the channel matrix coefficients for these ssps.
    ##### generateChannelCoeff() is a private method but for this testing we call it.
    self.set_deterministic_ssp(fading)
    fading.H_usn = fading._Fading3gpp__generateChannelCoeff()
    ##############################################################
    
    ###### Now, we call one of the main API functions of the channel performance
    ###### module to get the different channel performance mtrics and test that
    ###### results are OK.
    
    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.performance.compute_snr(fading,self.freq_band,self.wMS,self.wBS,self.aMS,self.aBS)
    u,s,vh = np.linalg.svd(np.average(H,axis=0))
    cond = max(s)/min(s)
    self.assertAlmostEqual(snr,13.0,1,"snr error", )
    self.assertAlmostEqual(np.average(rxpsd,axis=0),2.52e-19,20,"average over prbs rxpsd error", )
    ### The square module of H elements must be 16 because Rx and Tx antennas have 8db gain in the phi=0 theta = 90 direction
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(H)**2,axis=0)[0][0]),16.0,0," abs(H)** 2 in db error")
    #### The square module of the beamforming gain in db must be 22 db 16 of the antennas and 6 db for the 2x2 MIMO array
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(G)**2,axis=0)),22.0,0," abs(G)** 2 in db error")
    ######## In this case the condition number of the matrix is around 163 db
    self.assertAlmostEqual(10*np.log10(cond),163,0," matrix condition number in db error")
 
  def test_LOS_imaginary(self):   
    """ This method tests for a LOS condition with a distance between MS and BS that gives a channel coefficients with only imaginary part.The others ssp are controlled. 
    The method tests the generation of the channel matrix coefficients, the channel matrix Fourier Transform, and the channel performance. 
    
    """ 

    ####### Build the fading object. This object generates all random short scale parameters
    ####### and computes the channel coefficients. This call checks that all method in fading
    ####### modules are executed without errors
    ##### First set the LOS condition of the scenario
    self.LOS = 1 ### Condition LOS
    self.scf.force_los = self.LOS
    ##### In LOS condition the fase of the LOS delay it is important for the channel matrix coefficients
    ##### we set the MS position to have a pi/2 phase.
    ### The results are the similar with the ones obtained in test1 for NLOS but the channel coefficients
    ### with only imaginay part.
    self.MS_pos = np.array([100.0025,0,20])
    fading = fad.Fading3gpp(self.scf)
    fading.compute_ch_matrix(self.MS_pos,self.MS_vel, self.aMS,self.aBS,0,mode=0)
    ############################################################
    
    ##### In the following function we override the short scale parameters and 
    ##### we compute the channel matrix coefficients for these ssps.
    ##############################################################
    self.set_deterministic_ssp(fading)
    fading.H_usn = fading._Fading3gpp__generateChannelCoeff()

    
    ###### Now, we call one of the main API functions of the channel performance
    ###### module to get the different channel performance mtrics and test that
    ###### results are OK.
    
    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.performance.compute_snr(fading,self.freq_band,self.wMS,self.wBS,self.aMS,self.aBS)

    self.assertAlmostEqual(snr,28,0,"snr error", )
    self.assertAlmostEqual(np.average(rxpsd,axis=0),7.93e-18,19,"average over prbs rxpsd error", )
    ### The square module of H elements must be 16 because Rx and Tx antennas have 8db gain in the phi=0 theta = 90 direction
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(H)**2,axis=0)[0][0]),16.0,0," abs(H)** 2 in db error")
    #### The square module of the beamforming gain in db must be 22 db 16 of the antennas and 6 db for the 2x2 MIMO array
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(G)**2,axis=0)),22.0,0," abs(G)** 2 in db error")
    ######## In this case the coeffcients of the channel matrix has real part null.
    self.assertAlmostEqual(np.average(H,axis=0)[0][0],0-6.3j,1," channel coeffcients error")
    

  def test_NLOS_initialphase(self):
    """ This method tests for a NLOS condition with power in only one clsuter and initialphases not zero.The others ssps are controlled. 
    The method tests the generation of the channel matrix coefficients, the channel matrix Fourier Transform, and the channel performance. 
    
    """ 

    ####### Build the fast fading object. This object generates all random short scale parameters
    ####### and computes the channel coefficients. This call checks that all method in fading
    ####### modules are executed without errors
    ##### First set the LOS condition of the scenario
    self.LOS = 0 ### Condition NLOS
    self.scf.force_los = self.LOS
    self.MS_pos = np.array([100.0025,0,20])
    fading = fad.Fading3gpp(self.scf)
    fading.compute_ch_matrix(self.MS_pos,self.MS_vel, self.aMS,self.aBS,0,mode=0)
    ############################################################
    
    ##### In the following instructions we override the short scale parameters and 
    ##### we compute the channel matrix coefficients for these ssps.
    self.set_deterministic_ssp(fading)
    n_clus = fading.reduced_n_scatters
    m_rays = fading.scenario.raysPerCluster
    for i in range(n_clus):
      for j in range(m_rays):
         # For the moment we are using only vertical polarization. Then, the only
         #initalphase that has influence is the 0 one, \Phi_{theta,theta}.
         # For testing only access to the provate attributes
         fading._Fading3gpp__ini_phases[i][j][0]=np.pi/4
         fading._Fading3gpp__ini_phases[i][j][1]=0
         fading._Fading3gpp__ini_phases[i][j][2]=0
         fading._Fading3gpp__ini_phases[i][j][3]=0
    fading.H_usn = fading._Fading3gpp__generateChannelCoeff()
    ##############################################################
    
    ###### Now, we call one of the main API functions of the channel performance
    ###### module to get the different channel performance mtrics and test that
    ###### results are OK.
    
    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.performance.compute_snr(fading,self.freq_band,self.wMS,self.wBS,self.aMS,self.aBS)
    u,s,vh = np.linalg.svd(np.average(H,axis=0))
    self.assertAlmostEqual(snr,13,0,"snr error", )
    self.assertAlmostEqual(np.average(rxpsd,axis=0),2.52e-19,20,"average over prbs rxpsd error", )
    ### The square module of H elements must be 16 because Rx and Tx antennas have 8db gain in the phi=0 theta = 90 direction
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(H)**2,axis=0)[0][0]),16.0,0," abs(H)** 2 in db error")
    #### The square module of the beamforming gain in db must be 22 db 16 of the antennas and 6 db for the 2x2 MIMO array
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(G)**2,axis=0)),22.0,0," abs(G)** 2 in db error")
    ######## In this case the coeffcients of the channel matrix has the same ral and imaginary part.
    self.assertAlmostEqual(np.average(H,axis=0)[0][0],4.46+4.46j,2," channel coeffcients error")
 
  def test_NLOS_AOD(self):   
    """ This method tests for a NLOS condition with power in only one clsuter and an azimuth angle of departure not zero.The others ssps are controlled. 
    The method tests the generation of the channel matrix coefficients, the channel matrix Fourier Transform, and the channel performance. 
    
    """ 
      
    ####### Build the fast fading object. This object generates all random short scale parameters
    ####### and computes the channel coefficients. This call checks that all method in fast_fading
    ####### modules are executed without errors
    ##### First set the LOS condition of the scenario
    self.LOS = 0 ### Condition NLOS
    self.scf.force_los = self.LOS
    self.MS_pos = np.array([100.0025,0,20])
    fading = fad.Fading3gpp(self.scf)
    fading.compute_ch_matrix(self.MS_pos,self.MS_vel, self.aMS,self.aBS,0,mode=0)
    ############################################################
    
    ##### In the following instructions we override the short scale parameters and 
    ##### we compute the channel matrix coefficients for these ssps.
    self.set_deterministic_ssp(fading)
    n_clus = fading.reduced_n_scatters
    m_rays = fading.scenario.raysPerCluster
    for i in range(n_clus):
      for j in range(m_rays):
         # For the moment we are using only vertical polarization. Then, the only
         #initalphase that has influence is the 0 one, \Phi_{theta,theta}.
         fading._Fading3gpp__phiAOD_m_rad[i][j] = np.pi/4
    fading.H_usn = fading._Fading3gpp__generateChannelCoeff()
    ##############################################################
    
    ###### Now, we call one of the main API functions of the channel performance
    ###### module to get the different channel performance mtrics and test that
    ###### results are OK.
    
    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.performance.compute_snr(fading,self.freq_band,self.wMS,self.wBS,self.aMS,self.aBS)
    u,s,vh = np.linalg.svd(np.average(H,axis=0))

    self.assertAlmostEqual(snr,0.22,2,"snr error", )
    self.assertAlmostEqual(np.average(rxpsd,axis=0),1.32e-20,21,"average over prbs rxpsd error", )
    ### The square module of H elements must be 16 because Rx and Tx antennas have 8db gain in the phi=0 theta = 90 direction
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(H)**2,axis=0)[0][0]),10.2,1," abs(H)** 2 in db error")
    #### The square module of the beamforming gain in db must be 22 db 16 of the antennas and 6 db for the 2x2 MIMO array
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(G)**2,axis=0)),9.2,1," abs(G)** 2 in db error")
    ######## In this case the coeffcients of the channel matrix has the same ral and imaginary part.
    #self.assertAlmostEqual(np.average(H,axis=0)[0][0],4.46+4.46j,2," channel coeffcients error")
    # print(" ----------------------snr---------------" )
    # print(snr)
    # print(" ----------------------rxpsd average over prbs---------------" )
    # print(np.average(rxpsd,axis=0))
    # print(" ----------------------H average over prbs---------------" )
    # print(np.average(H,axis=0))
    # print(10*np.log10(np.average(np.abs(H)**2,axis=0)))
    # print(" ----------------------G average over prbs---------------" )
    # print(np.average(G,axis=0))
    # print(10*np.log10(np.average(np.abs(G)**2,axis=0)))
    # print(np.average(np.abs(G)**2,axis=0))
    # print(np.std(np.abs(G)**2,axis=0))
    # print(" ----------------------svd avgerage---------------" )
    # print(cond)
  def test_NLOS_2clusters(self): 
    """ This method tests for a NLOS condition with two clusters with the same power.The others ssps are controlled. 
    The method tests the generation of the channel matrix coefficients, the channel matrix Fourier Transform, and the channel performance. 
    
    """ 
    ####### Build the fast fading object. This object generates all random short scale parameters
    ####### and computes the channel coefficients. This call checks that all method in fast_fading
    ####### modules are executed without errors
    ##### First set the LOS condition of the scenario
    self.LOS = 0 ### Condition NLOS
    self.scf.force_los = self.LOS
    ##### In LOS condition the fase of the LOS delay it is important for the channel matrix coefficients
    ##### we set the MS position to have a pi/2 phase.
    ### The results are the similar with the ones obtained in test1 for NLOS but the channel coefficients
    ### with only imaginay part.
    self.MS_pos = np.array([100.0025,0,20])
    fading = fad.Fading3gpp(self.scf)
    fading.compute_ch_matrix(self.MS_pos,self.MS_vel, self.aMS,self.aBS,0,mode=0)
    ############################################################
    
    ##### In the following instructions we override the short scale parameters and 
    ##### we compute the channel matrix coefficients for these ssps.
    ##### For testing only access to private variables
    self.set_deterministic_ssp(fading)
    fading._Fading3gpp__P[0] = 0.5
    fading._Fading3gpp__P[1] = 0.5
    fading.H_usn = fading._Fading3gpp__generateChannelCoeff()
    ##############################################################
    
    ###### Now, we call one of the main API functions of the channel performance
    ###### module to get the different channel performance mtrics and test that
    ###### results are OK.
    
    snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow= self.performance.compute_snr(fading,self.freq_band,self.wMS,self.wBS,self.aMS,self.aBS)
    self.assertAlmostEqual(snr,16,0,"snr error", )
    self.assertAlmostEqual(np.average(rxpsd,axis=0),5.05e-19,20,"average over prbs rxpsd error", )
    ### The square module of H elements must be 16 because Rx and Tx antennas have 8db gain in the phi=0 theta = 90 direction
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(H)**2,axis=0)[0][0]),19.0,0," abs(H)** 2 in db error")
    #### The square module of the beamforming gain in db must be 22 db 16 of the antennas and 6 db for the 2x2 MIMO array
    self.assertAlmostEqual(10*np.log10(np.average(np.abs(G)**2,axis=0)),25.0,0," abs(G)** 2 in db error")
    ######## In this case the coeffcients of the channel matrix has the same ral and imaginary part.
    self.assertAlmostEqual(np.average(H,axis=0)[0][0],8.92+0j,1," channel coeffcients error")
    # print(" ----------------------snr---------------" )
    # print(snr)
    # print(" ----------------------rxpsd average over prbs---------------" )
    # print(np.average(rxpsd,axis=0))
    # print(" ----------------------H average over prbs---------------" )
    # print(np.average(H,axis=0))
    # print(10*np.log10(np.average(np.abs(H)**2,axis=0)))
    # print(" ----------------------G average over prbs---------------" )
    # print(np.average(G,axis=0))
    # print(10*np.log10(np.average(np.abs(G)**2,axis=0)))
    # print(np.average(np.abs(G)**2,axis=0))
    # print(np.std(np.abs(G)**2,axis=0))

  def test_scatter_move_nomove(self): 
    """ This method tests for a MS fixed in one position that if the scatters are not moving the channel matrix coefficients are the same
    all the time but if the scatters are moving the channel matrix change with time. 
    The method tests the compute path metod of the channel performance. 
   
    """ 
    ######## Build the scenario for testing
    self.fcGHz = 30
    """ Scenario frequency in GHz""" 
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 25
    BS_pos = np.array([0,0,20])
    Ptx_db = 30
    force_los = 2

    self.scf=  sc.Scenario3GPPUmi(self.fcGHz, posx_min,posx_max, posy_min, posy_max, grid_number, BS_pos, Ptx_db,True,force_los)
    """ Scenario for test 3gppUmi""" 

    #####################################################
    
    ########## Build the OFDM frequency band for testing
    self.freq_band =  fb.FrequencyBand(fcGHz=self.fcGHz,number_prbs=81,bw_prb=10000000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    """ OFDM frequency band for test""" 
    self.freq_band.compute_tx_psd(tx_power_dbm=30)
    ###################################################
    
    
    n_MS = 1
    positions = np.empty(shape=(n_MS),dtype = object)
    mspositions1= np.array(([10,10,2],[10,10,2]))
    positions[0] = mspositions1
    times  = np.empty(shape=(n_MS),dtype = object)
    timesMS1 = np.array(([0,0.01]))
    times[0] = timesMS1
    path = "./data" 
    #Scatters no move case
    fading = fad.Fading3gpp(self.scf, scatters_move=False,move_probability=0,v_min_scatters=0,v_max_scatters=10)

    self.performance.compute_path(fading, self.freq_band, self.aMS,self.aBS,positions,times,force_los,path,mode=2)               
    H = self.performance.H[0]

    self.assertTrue(np.allclose(H[0],H[1], rtol=1e-05, atol=1e-08)," Error : in the same point and the scatters not moving, H must be the same")
    #Scatters move case
    fading = fad.Fading3gpp(self.scf,scatters_move=True,move_probability=1,v_min_scatters=0,v_max_scatters=10)

    self.performance.compute_path(fading, self.freq_band, self.aMS,self.aBS,positions,times,force_los,path,mode=2)               
    H = self.performance.H[0]
    #print(H[0],H[1])
    self.assertTrue(np.all(np.not_equal(H[0],H[1]))," Error : in the same point and the scatters  moving, H must be different")

  def test_multiple_MS(self): 
    """ This method tests for multiple MS in the scenario. 
    
    """ 
    n_MS = 2
    positions = np.empty(shape=(n_MS),dtype = object)
    mspositions1= np.array(([10,10,2],[10,10,2]))
    mspositions2 = np.array(([10,10,2],[20,10,2]))
    positions[0] = mspositions1
    positions[1] = mspositions2
    times  = np.empty(shape=(n_MS),dtype = object)
    timesMS1 = np.array(([0,0.01]))
    timesMS2 = np.array(([0,0.01]))
    times[0] = timesMS1
    times[1] = timesMS2
    force_los = 2

    path = "./data" 
    #Scatters no move case
    fading = fad.Fading3gpp(self.scf,scatters_move=False,move_probability=0,v_min_scatters=0,v_max_scatters=10)

    self.performance.compute_path(fading, self.freq_band, self.aMS,self.aBS,positions,times,force_los,path,mode=1)               
    # self.assertTrue(np.all(np.equal(H[0],H[1]))," Error : in the same point and the scatters not moving, H must be the same")
    # #Scatters move case
    # snr,rxpsd,H,G,linear_losses, distances,snr_pl,sp_eff = self.performance.compute_path(self.scf, self.freq_band, self.aMS,self.aBS,mspositions,times,force_los,path,mode=1,scatters_move=True,move_probability=1,v_min_scatters=0,v_max_scatters=10)               
    # self.assertTrue(np.all(np.not_equal(H[0],H[1]))," Error : in the same point and the scatters  moving, H must be different")


  def set_deterministic_ssp(self,fading):
    """ This method sets the deterministics values of the short scale parameters.
    
    @type fading: Class Fading3gpp.
    @param fading: The fading object to override its ssps values.
    """ 
    ###### we access and assign the private atributes only for test
    fading.scenario.raysPerCluster =1
    fading.scenario.n_scatters = 2
    fading.reduced_n_scatters = 2
    n_clus = fading.reduced_n_scatters
    m_rays = fading.scenario.raysPerCluster
    fading._Fading3gpp__sigma_K = 100
    fading._Fading3gpp__P = np.zeros(n_clus)
    fading._Fading3gpp__P[0] = 1
    fading._Fading3gpp__P[1] = 0
    fading._Fading3gpp__P_LOS = np.zeros(n_clus)
    fading._Fading3gpp__P_LOS[0] = 1
    fading._Fading3gpp__P_LOS[1] = 0
    fading._Fading3gpp__tau = np.zeros(n_clus)
    fading._Fading3gpp__phiAOA = np.zeros(n_clus)
    fading._Fading3gpp__phiAOD = np.zeros(n_clus)
    fading._Fading3gpp__thetaAOA = np.ones(n_clus)*np.pi/2
    fading._Fading3gpp__thetaAOD = np.ones(n_clus)*np.pi/2
    fading._Fading3gpp__phiAOA_m_rad = np.zeros((n_clus,m_rays))
    fading._Fading3gpp__phiAOD_m_rad = np.zeros((n_clus,m_rays))
    fading._Fading3gpp__thetaAOA_m_rad = np.ones((n_clus,m_rays))*np.pi/2
    fading._Fading3gpp__thetaAOD_m_rad = np.ones((n_clus,m_rays))*np.pi/2
    fading._Fading3gpp__xpol = np.zeros((n_clus,m_rays)) 
    fading._Fading3gpp__ini_phases= np.zeros((n_clus,m_rays,4))
    fading.scenario.shadow_enabled = False
    fading._Fading3gpp__cluster1 = 0
    fading._Fading3gpp__cluster2 = 0
 
if __name__ == "__main__":
    
    test = FadingPerformanceTest()
    test.test_SISORayleighFading()
    test.test_SISORicianFading()
    test.config()
    test.test_SISO()
    
    test.config3gpp()
    test.test_NLOS_1cluster()
    test.test_LOS_imaginary()
    test.test_NLOS_initialphase()
    test.test_NLOS_AOD()
    test.test_NLOS_2clusters()
    test.test_scatter_move_nomove()
    test.test_multiple_MS() 


    
