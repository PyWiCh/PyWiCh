#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the frequency_band module.

@author: pablobelzarena
"""
##### to run from the project directory
import sys,os
sys.path.append('./src')
sys.path.append('./src/gui')
sys.path.append('./src/graph')

###### to run from qa directory
sys.path.append('../')
sys.path.append('../gui')
sys.path.append('../graph')

import frequency_band as fb
import numpy as np 
import unittest

class FrequencyBandTest(unittest.TestCase):
  """Unitest class for testing frequency_band."""  
  
  def test_psd(self):
    """ This method tests the noise and tx psd and OFDM frequencies. 

    """ 
    freq_band =  fb.FrequencyBand(fcGHz=1,number_prbs=100,bw_prb=180000,noise_figure_db=5.0,thermal_noise_dbm_Hz=-174.0) 
    freq_band.compute_tx_psd(tx_power_dbm=30)
    txpsd = freq_band.txpsd
    noisepsd = freq_band.noisepsd
    ts = np.ones(freq_band.n_prbs)*0.01
    self.assertTrue((txpsd*180000==ts).all(), "tx psd error" )

    th_noise_W = pow(10, (-174 - 30) / 10)
    noise_figure = pow (10, 5 / 10)
    npsd = 10*np.log10(th_noise_W*noise_figure)
    ns = np.ones(freq_band.n_prbs)*npsd
    self.assertTrue((10*np.log10(noisepsd) == ns).all(), "noise psd error" )

    fc_prb1 = freq_band.fc_prbs[int(freq_band.n_prbs/2)]
    fc_prb0 = freq_band.fc_prbs[int(freq_band.n_prbs/2)-1]
    self.assertEqual(int((fc_prb1+fc_prb0)/2), int(freq_band.fcGHz*1e9)," carrier frequency error" )

    self.assertEqual(int((fc_prb1-fc_prb0)), int(freq_band.bw_prb)," bw prb frequency error" )


if __name__ == "__main__":
    
    test = FrequencyBandTest()
    test.test_psd()
