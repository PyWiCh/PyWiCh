#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 09:46:15 2023

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
import csv



class ToFileTest(unittest.TestCase):
    """Unitest class for testing fading and channel performance.
    
      It is very difficult to test the fading module because all short scale parameters are build
      using random variables. Therefore, the channel matrix coefficients depend on the realization of
      many rvs like: clusters delays, clusters powers, clusters arrival and departure angles,
      cross polarizations and intial phases. For testing this module we override these random variables
      with deterministics values and we test the buiding of the channel matrix for these known scenarios.
      We also test the performance results using the channel_performance methods that are the interface( like an
      API) to access the results of the wireless channel simulations.
      
    """   



    def config(self):
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
        #print(self.aBS.get_number_of_elements(), " numero de elementos "    )
        ###########################################################
        
        ######## Build the scenario for testing
        self.fcGHz = 30
        """ Scenario frequency in GHz""" 
        self.posx_min = -300
        self.posx_max = 300
        self.posy_min = -300
        self.posy_max = 300
        grid_number = 30
        self.n_userequipments =5
        BS_pos = np.array([0,0,20])
        Ptx_db = 40
    
        sigma_shadow=5
        shadow_corr_distance=5  
        order = 2
        self.LOS = True
        #self.scf=  sc.ScenarioSimpleLossModel(self.fcGHz,self.posx_min,self.posx_max,self.posy_min,self.posy_max,grid_number,BS_pos, Ptx_db,order,sigma_shadow,shadow_corr_distance)
        """ Scenario for test Simple model of Losses """ 
        self.scf=  sc.Scenario3GPPUmi(self.fcGHz, self.posx_min,self.posx_max, self.posy_min, self.posy_max, grid_number, BS_pos, Ptx_db,True,self.LOS)
    
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
        #self.fading = fad.FadingSiSoRayleigh(self.scf,10)
        self.fading = fad.Fading3gpp(self.scf)
        self.title = "5gUmi-"+str(self.n_userequipments)+"_ues-5000ms"
        self.mycsv_snr = CSVFile('./snr-'+self.title+'.csv','w')
        self.mycsv_snr.get_writer()
        self.mycsv_pos = CSVFile('./positions-'+self.title+'.csv','w')
        self.mycsv_pos.get_writer()

    def gen_ues(self):
        self.ls_usreqs = []
        for i in range(self.n_userequipments):
            keys = ['name', 'pos', 'vel']
            values = ["UE-" + str(i+1), [0,0,0], [0,0,0]]
            ue = dict(zip(keys, values))
            x_t = np.random.uniform(self.posx_min, self.posx_max)
            y_t = np.random.uniform(self.posy_min, self.posy_max)
            pos = [x_t,y_t,2]
            ue['pos']=pos
            if np.abs(x_t) > self.posx_max/2:
                vx_t = np.random.uniform(0, 30)*-np.sign(x_t)
            else:
                vx_t = np.random.uniform(0, 30)*np.sign(x_t)
            if np.abs(y_t) > self.posy_max/2:
                vy_t = np.random.uniform(0, 30)*-np.sign(y_t)
            else:
                vy_t = np.random.uniform(0, 30)*np.sign(y_t)
            vel = [vx_t,vy_t,0]
            ue['vel']=vel
            self.ls_usreqs.append(ue)
        
    
    def test(self):   
        """ This method tests. 
        """ 
    
        iterations = 51
        snr = np.zeros(iterations)
        snr_pl = np.zeros(iterations)
        snr_pl_shadow= np.zeros(iterations)
        sp_eff = np.zeros(iterations)
        t = 0
        force_los =2
        mode = 0
        #mode = 1 #for 3gpp
        for i in range(iterations):
            delta_t = 0.001
            t = t+delta_t
            for j in range(self.n_userequipments):
                MS_pos = self.ls_usreqs[j]['pos']   
                MS_vel = self.ls_usreqs[j]['vel']
                self.mycsv_pos.write(self.ls_usreqs[j]['name'],str(t),''+str(MS_pos[0])+' '+str(MS_pos[1])+' '+str(MS_pos[2]))
                snr,rxpsd,H,G,linear_losses,snr_pl,sp_eff,snr_pl_shadow = self.performance.compute_point(self.fading,self.freq_band,self.aBS,self.aMS,MS_pos,MS_vel,t,force_los,mode)               
                self.mycsv_snr.write(self.ls_usreqs[j]['name'],str(t),str(snr)+' '+ '0.0'+' '+ '0.0')
                self.ls_usreqs[j]['pos'][0] = MS_pos[0]+MS_vel[0] * delta_t
                self.ls_usreqs[j]['pos'][1] = MS_pos[1]+MS_vel[1] * delta_t
                self.ls_usreqs[j]['pos'][2]= MS_pos[2]+MS_vel[2] * delta_t
                # write the data
            if i%10 == 0 :
                print("Iterations ",i)
        #self.graph_positions()
        self.mycsv_snr.close()
        self.mycsv_pos.close()

    def graph_positions(self):
        self.mycsv_pos = CSVFile('./positions-'+self.title+'.csv','w')
        self.mycsv_pos.get_reader()
        self.mycsv_pos.read()
        
    def to_cvs(self):
        self.f = open('csvfile.csv','w')

class CSVFile():

    def __init__(self, filename,r_w):
        self.filename = filename
        self.fp = open(self.filename, r_w, encoding='utf8')
    
    def get_reader(self):
        self.reader = csv.reader(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')
        
    def get_writer(self):
        self.writer = csv.writer(self.fp, delimiter=';', quotechar='"', quoting=csv.QUOTE_ALL, lineterminator='\n')

        
    def close(self):
        self.fp.close()

    def write(self, *args):
        self.writer.writerow(args)

    def size(self):
        return os.path.getsize(self.filename)

    def fname(self):
        return self.filename
    
    def read(self):
        data_pos =[[]]
        line_count = 0
        for row in self.reader:
            print(row[0],row[1])
            res = [float(ele) for ele in row[2].split()]
            print(res)
            line_count += 1
        print(f'Processed {line_count} lines.')

           
  
        
  
if __name__ == "__main__":
    
    test = ToFileTest()
    test.config()
    test.gen_ues()
    test.test()

        
     