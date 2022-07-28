#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements the OFDM frequency spectrum. 

@author: pablo belzarena
"""

import numpy as np
class FrequencyBand:
  """This class implements the OFDM frequency spectrum. 
  
  """  
  def __init__(self,fcGHz,number_prbs=100,bw_prb=180000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0): 
    """ FrequencyBand constructor
    
    Note: This version assumes that all prbs transport data.
    @type fcGHz: float.
    @param fcGHz: The carrier frequency of OFDM spectrum in GHz.
    @type number_prbs: int.
    @param number_prbs: The number of physical reseource blocks (PRB) in OFDM. Default 100.
    @type bw_prb: float.
    @param bw_prb: The bandwidth of each physical reseource blocks in OFDM. In Hertz.
    Default 180000.
    @type noise_figure_db: float.
    @param noise_figure_db :The noise figure in db.
    @type thermal_noise_dbm_Hz: float.
    @param thermal_noise_dbm_Hz :The thermal noise in dbm per Hertz.

   """
    self.n_prbs = number_prbs
    """ The number of reseource blocks in OFDM.  """ 
    self.bw_prb = bw_prb
    """ The bandwidth of each reseource blocks in OFDM. In Hertz.""" 
    self.fcGHz = fcGHz
    """ carrier frequency of OFDM spectrum in GHz.""" 
    self.noise_figure_db = noise_figure_db
    """ The noise figure in db.""" 
    self.thermal_noise_dbm_Hz = thermal_noise_dbm_Hz
    """The thermal noise in dbm per Hertz.""" 
    self.fc_prbs = np.zeros(self.n_prbs)
    """The frequency of each prb """
    f = fcGHz*1e9 - (self.n_prbs * self.bw_prb / 2.0);
    for nrb in range(self.n_prbs):
      f = f + self.bw_prb/2
      self.fc_prbs[nrb] = f
      f = f + self.bw_prb/2
    self.txpsd = np.ones(self.n_prbs)
    """ The transmited power spectral density of each prb """ 
    self.noisepsd = np.ones(self.n_prbs)
    """ The noise  power spectral density of each prb """     
    self.compute_noise_psd()
    
  def save(self,path):
    """ This method saves to disk the configuration of the frequency band.

    @type path: string.
    @param path: The dirctory to save the information.
    """ 
    np.savetxt(path+'/frequency_band.csv', [self.n_prbs,self.bw_prb,self.noise_figure_db,self.thermal_noise_dbm_Hz], delimiter=',')      
 
    
  def compute_tx_psd (self,tx_power_dbm):
    """ This method given the Tx power computes the Tx power spectral density.

    TODO: In this version the Tx power is divided equally between prbs.
    @type tx_power_dbm: float.
    @param tx_power_dbm :The Tx power of the BS in dbm.
    """
    tx_power_W = pow(10,(tx_power_dbm - 30) / 10) # 30 is to convert from mW to W
    tx_power_density = tx_power_W / (self.n_prbs * self.bw_prb)
    for prb in range(self.n_prbs): 
      self.txpsd[prb] = tx_power_density
    
  def compute_noise_psd(self):
    """ This method given the noise figure and the thermal noise, computes the 
    noise power spectral density.

    """
    th_noise_W = pow(10, (self.thermal_noise_dbm_Hz - 30) / 10)
    noise_figure = pow (10, self.noise_figure_db / 10)
    noise_psd_value =  th_noise_W * noise_figure
    for prb in range(self.n_prbs): 
      self.noisepsd[prb] = noise_psd_value

################# Default Values ##############################

# Stefania Sesia, Issam Toufik, Matthew Baker - LTE - The UMTS Long Term Evolution_ From Theory to Practice-Wiley (2011)
# pag 478 : In the LTE specifications the thermal noise density, kT , is defined to be −174 dBm/Hz where k is Boltzmann’s 
#constant (1.380662 × 10−23) and T is the temperature of the receiver (assumed to be 15◦C)
# pag 479:
#LTE defines an NF requirement of 9 dB for the UE, the same as UMTS. This is somewhat higher than the NF of a state-of-the-art 
#receiver, which would be in the region of 5–6 dB, with typically about 2.5 dB antenna filter insertion loss and an NF for the 
#receiver integrated circuit of 3 dB or less. Thus, a practical 3–4 dB margin is allowed. The eNodeB requirement is for an NF of 5 dB.


# 3GPP TR 36.942 version 13.0.0 Release 13
#Table 4.6: E-UTRA FDD and E-UTRA TDD reference base station parameters
#Maximum BS power: 43dBm for 1.25, 2.5 and 5MHz carrier, 46dBm for 10, 15 and 20MHz carrier
#Maximum power per DL traffic channel : 32dBm
#Noise Figure 5 db

# In 5G NR see: https://www.etsi.org/deliver/etsi_ts/138100_138199/138104/15.04.00_60/ts_138104v150400p.pdf
# In 5G NR there different base stations architectures (1-c,1-h 1-o,2-o ) and different bands fr1 and fr2.
# Each case has differentt max output power all in the range 30-50 dbm   
     
# kT_dBm_Hz  = -174.0 # dBm/Hz
# noiseFigure = 5.0; # noise figure in dB


# fb = FrequencyBand(1)
# fb.compute_tx_psd(46.5)
# print("tx psd",10*np.log10(fb.psd))
# print(fb.psd*180000)
# fb = FrequencyBand(1)
# fb.compute_noise_psd(noiseFigure,kT_dBm_Hz)
# print("noise psd",10*np.log10(fb.psd))





