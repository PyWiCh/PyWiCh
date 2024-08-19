#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is the main User interface of the Python wireless channel simulator

@author: pablobelzarena
"""
##### to run from the project directory
import sys,os
sys.path.append('./src')
sys.path.append('./src/gui')
sys.path.append('./src/graph')

###### to run from gui directory
sys.path.append('../')
sys.path.append('../gui')
sys.path.append('../graph')


import tkinter as tk
import tkinter.font as tkfont
import gui.gui_antennas as ga
import gui.gui_scenarios as gs
import gui.gui_select_LSP as glsp
import gui.gui_select_SSP as gssp
import gui.gui_user_message as gum
import gui.gui_frequency_band as gfb
import gui.gui_name_input as gni
import gui.gui_select_point as gsp
import gui.gui_select_point_prb as gspp

import frequency_band as fband
import scenarios as sc
import fading as fad

import graph.graph_scenarios as grs
import channel_performance as cp
import numpy as np
from tkinter import filedialog
import errno


class AppGraph():
    """ This class is the main form of the Python wireless channel simulator.
    
    """
    def __init__(self,window,title):
        """The constructor of the AppNameInput Class
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type title: string
        @param title: The title of the form.

        """ 
        self.freq_band = None
        """ The FrequencyBand object. """ 
        self.cname = None
        """ The list of scenarios. """ 
        self.scenario = None
        """ The Scenario Object object. """ 
        self.mspos = None
        """ The start position of the MS route. """ 
        self.msvel = None
        """The velocity of the MS. """ 
        self.force_los = None
        """ A variable that incdicates if is forced to LOS or NLOS condition or if the LOS condition is computed form the probability model. """ 
        self.antennaTx = None
        """ The antenna Tx object. """ 
        self.antennaRx = None
        """ The antenna Rx object.""" 
        self.ant_type = None
        """ The antenna type. """ 
        self.scenario_update = True
        """ Boolean variable that indictes if the scnario has changed. """ 
        self.directory_name  = None
        """ The directory to save the configuration data. """ 
        self.points_in_paths = 20
        """ The number of points in the MS route."""
        self.__window  = window
        """ The main window of this form. """ 
        self.__window.title(title)
        self.move_probability = 0
        """ The probability that each scatter is moving.""" 
        self.v_min_scatters = 0
        """ The minimum velocity of one scatter movement.""" 
        self.v_max_scatters = 0
        """ The maximum velocity of one scatter movement. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the main form to enter the data.
        
        """         
        width, height = window.winfo_screenwidth(), window.winfo_screenheight()
        self.__FONT_SIZE = int(width /1.25/ 1920*25)
        #window.geometry('%dx%d+0+0' % (width/1.1,height/1.1))

        self.__window.rowconfigure((0,1,2,3,4,5,6,7), weight=1)  # make buttons stretch when

        self.__window.columnconfigure((0,1,2,3,4), weight=1)# columnconfigure([0, 1, 2, 3, 4, 5], minsize=50, weight=1) 
        self.gui_simultion_config()
        
    def gui_simultion_config(self):
        """This method builds the buttons to enter the configuration, to run the simulation and to analyze the results of the simulations.
        
        """         
             
        font = tkfont.Font(family="Helvetica", size=self.__FONT_SIZE, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/1.)
        
        lbl_config = tk.Label(self.__window, text="Configuration of the \n simulation scenario: ",fg = "dark blue",font = font)#Verdana 18 bold")
        lbl_config.grid(row=0, column=2, columnspan=1, sticky='EWNS')

       
        aux0 = tk.Button(self.__window, text="Select \n Frequency \n Band",font=font, compound=tk.CENTER, command=self.cmd_freq_band)
        aux0.grid(row=2, column=3, columnspan=1, sticky='EWNS') #padx=10)
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))
        
        aux0 = tk.Button(self.__window, text="Select Tx \n Antenna", font=font, compound=tk.CENTER,command=self.cmd_tx_antenna)
        aux0.grid(row=2, column=0, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))
        
        aux0 = tk.Button(self.__window, text="Select Rx \n Antenna", font=font, compound=tk.CENTER,command=self.cmd_rx_antenna)
        aux0.grid(row=2, column=1, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        aux0 = tk.Button(self.__window, text="Select \n Scenario", font=font, compound=tk.CENTER,command=self.cmd_scenario)
        aux0.grid(row=2, column=2, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        frm_runsim = tk.Frame(master=self.__window)
        lbl_runsim = tk.Label(master=frm_runsim, text="Run the simulation : ",fg = "dark blue",font = font)
        lbl_runsim.grid(row=0, column=0, sticky="w")
        frm_runsim.grid(row=3, column=2, padx=10)

        aux0 = tk.Button(self.__window, text="Graph \nscenario\n and Tx \npower map",font=font, compound=tk.CENTER, command=self.cmd_tx_pm)
        aux0.grid(row=2, column=4, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))
               
        aux0 = tk.Button(self.__window, text="Run \n Simulation", font=font, compound=tk.CENTER,command=self.cmd_run)
        aux0.grid(row=4, column=2,columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        frm_graph = tk.Frame(master=self.__window)
        lbl_graph = tk.Label(master=frm_graph, text="Analysis of \n simulation results : ",fg = "dark blue",font = font)
        lbl_graph.grid(row=0, column=0, sticky="w")
        frm_graph.grid(row=5, column=2, padx=10)

        aux0 = tk.Button(self.__window, text="Select the \n directory with\n simulation \n results" , font=font, compound=tk.CENTER,command=self.cmd_directory)
        aux0.grid(row=6, column=2, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        aux0 = tk.Button(self.__window, text="Channel \n Matrix \n Analysis", font=font, compound=tk.CENTER,command=self.cmd_ch_matrix)
        aux0.grid(row=7, column=0, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        aux0 = tk.Button(self.__window, text="Multipah \n LSP \n Analysis",font=font, compound=tk.CENTER, command=self.cmd_multipath_lsp)
        aux0.grid(row=7, column=1,columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

        aux0 = tk.Button(self.__window, text="Multipath \n SSP \n Analysis"  ,font=font, compound=tk.CENTER, command=self.cmd_multipath_ssp)
        aux0.grid(row=7, column=2,columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))
  
        aux0 = tk.Button(self.__window, text="System \n Performance \n Analysis",font=font, compound=tk.CENTER, command=self.cmd_ch_performance)
        aux0.grid(row=7, column=3, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))
        
        aux0 = tk.Button(self.__window, text="Path \n Performance \n Analysis",font=font, compound=tk.CENTER, command=self.cmd_path_performance)
        aux0.grid(row=7, column=4, columnspan=1, sticky='EWNS')
        #aux0.config(width=int(square_size*1.5), height=int(square_size/2))

    ##### comand functions
        
    def cmd_ch_matrix(self): 
        """ This method is called when the user select to graph the channel matrix. 
        
        """ 
        if self.directory_name is not None:     
            try:
                freq_band = np.loadtxt(self.directory_name+'/frequency_band.csv',delimiter=',')
                self.__window_spoint = tk.Tk()
                """ The tk.TK() window for the AppSelectPointPrb form. """ 
                ap = gspp.AppSelectPointPrb(self.__window_spoint, self.select_point_H, "Select the MS, path point and Prb for the analysis", " OK", self.points_in_paths,int(freq_band[0]))
                self.__window_spoint.mainloop()
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )
       
        
    def cmd_multipath_lsp(self):  
        """ This method is called when the user select to graph LSP. 
        
        """ 
        self.__window_lsp = tk.Tk()
        """ The tk.TK() window for the AppLSP form. """ 
        app = glsp.AppLSP(self.__window_lsp,self.lsp_multipath_analysis," Select Large Scale Parameter" ,"OK" )
        self.__window_lsp.mainloop()
        
    def cmd_multipath_ssp(self):    
        """ This method is called when the user select to graph SSP. 
        
        """ 
        self.__window_ssp = tk.Tk()
        """ The tk.TK() window for the AppSSP form. """ 
        app = gssp.AppSSP(self.__window_ssp,self.ssp_multipath_analysis," Select Small Scale Parameter and the point in the path" ,"OK",self.points_in_paths )
        self.__window_ssp.mainloop()

    def cmd_ch_performance(self): 
        """ This method is called when the user select to graph the channel performance. 
        
        """ 
        if self.directory_name is not None:
            try:
                self.__window_spoint = tk.Tk()
                """ The tk.TK() window for the AppSelectPoint form. """ 
                ap = gsp.AppSelectPoint(self.__window_spoint, self.select_point_per, "Select the point of the path or the analysis", " OK", self.points_in_paths)
                self.__window_spoint.mainloop()
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )
   
    def cmd_run(self):
        """ This method is called when the user select to run the simulation. 
        
        """ 
        if self.scenario is not None and self.antennaTx is not None and self.antennaRx is not None and self.freq_band is not None:
            if self.scenario_update:
                self.__window_run = tk.Tk()
                """ The tk.TK() window for the AppNameInput form. """ 
                ap = gni.AppNameInput(self.__window_run, self.run, "Select the name of the simlation", " OK", "Simulation name","name")
                self.__window_run.mainloop()              
            else:
                gum.AppUserMsg("System message", "The scenario is updated. " )
        else:
            gum.AppUserMsg("Error message", " Scenario, Frequency band, AntennaTx or AntennaRx are not configured.\n Please configure all of them. " )

    def cmd_directory(self):
        """ This method is called when the user select the simulation directory. 
        
        """ 
        try:
            directory_name = filedialog.askdirectory(initialdir="./data",title='Please select the source directory')

            if directory_name is not None and directory_name != "" :
                self.directory_name = directory_name
                self.nMS = np.loadtxt(self.directory_name+'/nMS.csv',delimiter=',')
                """The quantity of MS in the simulation."""
                self.__window_MS = tk.Tk()
                """ The tk.TK() window for the AppNameInput form. """ 
                value = 0
                ap = gni.AppNameInput(self.__window_MS, self.select_MS, "Select the number of the MS to analyze", " OK", "MS number",value )
                self.__window_MS.mainloop()              

            else:
                gum.AppUserMsg('Error!','You must select a Directory. Please try again')

        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )

    def cmd_path_performance(self):
        """ This method is called when the user select to graph the MS route performance. 
        
        """ 
        if self.directory_name is not None:     
            try:
                freq_band = np.loadtxt(self.directory_name+'/frequency_band.csv',delimiter=',')
                self.__window_path_prb = tk.Tk()
                """ The tk.TK() window for the AppSelectPoint form. """  
                ap = gsp.AppSelectPoint(self.__window_path_prb, self.select_path_prb, "Select the prb for matrix H analysis along the path ", " OK",int(freq_band[0]))
                self.__window_path_prb.mainloop()
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )

    def cmd_tx_pm(self):
        """ This method is called when the user select to graph the power map. 
        
        """ 
        self.graph_powermap()
        
    def cmd_freq_band(self):
        """ This method is called when the user select to configure the frequency band. 
        
        """ 
        if self.scenario is None:
            gum.AppUserMsg("Error message", "Please select the scenario first" )
        else:
            self.__window_fb = tk.Tk()
            """ The tk.TK() window for the AppFreqBand form. """ 
            if self.freq_band is None:
                app = gfb.AppFreqBand(self.__window_fb,self.function_freq_band,"Frequency Band Specification" ,"OK")
            else:
                app = gfb.AppFreqBand(self.__window_fb,self.function_freq_band,"Frequency Band Specification" ,"OK",self.freq_band.n_prbs,self.freq_band.bw_prb,self.freq_band.noise_figure_db,self.freq_band.thermal_noise_dbm_Hz)
            self.__window_fb.mainloop()
       
    def cmd_tx_antenna(self):
        """ This method is called when the user select to configure the Tx antenna. 
        
        """ 
        self.__window_txa = tk.Tk()
        """ The tk.TK() window for the AppAntenna Tx form. """ 
        if self.antennaTx == None:
            app = ga.AppAntenna(self.__window_txa,self.function_tx_antenna,"Tx Antenna Specification","antennaTx")
        else:
            if self.ant_typeTx == 1:
                app = ga.AppAntenna(self.__window_txa,self.function_tx_antenna,"Tx Antenna Specification","antennaTx",self.antennaTx.n_rows,self.antennaTx.n_cols,self.antennaTx.d_h,self.antennaTx.d_v,self.antennaTx.alpha,self.antennaTx.beta,self.antennaTx.gamma, self.phi_addTx,self.theta_addTx, self.antennaTx.antenna_element.gaindb )
            else:
                app = ga.AppAntenna(self.__window_txa,self.function_tx_antenna,"Tx Antenna Specification","antennaTx",self.antennaTx.n_rows,self.antennaTx.n_cols,self.antennaTx.d_h,self.antennaTx.d_v,self.antennaTx.alpha,self.antennaTx.beta,self.antennaTx.gamma,self.phi_addTx,self.theta_addTx,8,self.antennaTx.antenna_element.maxgaindb,self.antennaTx.antenna_element.SLA_v,self.antennaTx.antenna_element.A_max,self.antennaTx.antenna_element.beamwidth,2 ) 
        self.__window_txa.mainloop()

    def cmd_rx_antenna(self):
        """ This method is called when the user select to configure the Rx antenna. 
        
        """ 
        self.__window_rxa = tk.Tk()
        """ The tk.TK() window for the AppAntenna Rx form. """ 
        if self.antennaRx is None:
           app = ga.AppAntenna(self.__window_rxa,self.function_rx_antenna,"Rx Antenna Specification","antennaRx" )
        else:
            if self.ant_typeRx == 1:
                app = ga.AppAntenna(self.__window_rxa,self.function_rx_antenna,"Rx Antenna Specification","antennaRx",self.antennaRx.n_rows,self.antennaRx.n_cols,self.antennaRx.d_h,self.antennaRx.d_v,self.antennaRx.alpha,self.antennaRx.beta,self.antennaRx.gamma,self.phi_addRx,self.theta_addRx,self.antennaRx.antenna_element.gaindb )
            else:
                app = ga.AppAntenna(self.__window_rxa,self.function_rx_antenna,"Rx Antenna Specification","antennaRx",self.antennaRx.n_rows,self.antennaRx.n_cols,self.antennaRx.d_h,self.antennaRx.d_v,self.antennaRx.alpha,self.antennaRx.beta,self.antennaRx.gamma,self.phi_addRx,self.theta_addRx,8,self.antennaRx.antenna_element.maxgaindb,self.antennaRx.antenna_element.SLA_v,self.antennaRx.antenna_element.A_max,self.antennaRx.antenna_element.beamwidth,2 )      
        self.__window_rxa.mainloop()
      
    def cmd_scenario(self):
        """ This method is called when the user select to configure the scenario. 
        
        """ 
        self.__window_sc = tk.Tk()
        """ The tk.TK() window for the AppScenario form. """ 

        if self.scenario is None:
            app = gs.AppScenarios(self.__window_sc,self.function_select_scenario," Select Scenario Specification" ,"OK")
        else:
            app = gs.AppScenarios(self.__window_sc,self.function_select_scenario," Select Scenario Specification" ,"OK",self.cname,self.scenario.fcGHz,self.scenario.posx_min,self.scenario.posx_max,self.scenario.posy_min,self.scenario.posy_max,self.scenario.grid_number,self.scenario.BS_pos,self.scenario.Ptx_db,self.force_los,self.move_probability,self.v_min_scatters,self.v_max_scatters)
        self.__window_sc.mainloop()

    #########################################################
    ####################callback functions###################    
    #########################################################

    def select_MS(self,ms):
        """This is the callback function of the form used to select the number of the MS to graph.
        
        @type ms: int
        @param ms: the snumber of the MS.
         
        """
        self.__window_MS.destroy()
        self.MS_number = int(ms)
        """ The number of the MS selected to graph."""
        try:
            if self.MS_number >= self.nMS or self.MS_number < 0:
                gum.AppUserMsg('Error' , ' The number of MS is invalid, try again')
    
            else:
                self.positions = np.loadtxt(self.directory_name+'/positions'+ms+'.csv',delimiter=',')
                self.points_in_paths = self.positions.size
            
        except BaseException as error:
           gum.AppUserMsg('Exception occurred!','{}'.format(error))
         

    def select_point_H(self,point,prb):
        """ This is the callback function of the AppSelectPointPrb form. 
        
        @type point: int
        @param point: the selected point number.
        @type prb: int.
        @param prb: The selected prb.
        """
        self.__window_spoint.destroy()
        self.graph_H_f(point,prb)

    def select_point_per(self,point):
        """ This is the callback function of the AppSelectPoint form. 
        
        @type point: int
        @param point: the selected point number.
        """ 
        self.__window_spoint.destroy()
        self.performance_analysis(point)
     
    def run(self,name): 
        """ This is the callback function of the AppNameInput form to run the simulation. 
        
        @type name: string
        @param name: the selected name of the simulation.
        """ 
        performance  = cp.ChannelPerformance()
        path = './data/'+name 
        try: 
            self.__window_run.destroy()
            os.mkdir(path) 
        except OSError as e:
            if e.errno == errno.EEXIST:
                gum.AppUserMsg('Error: Directory exists.' , ' Directory not created. Please select anothe name or delete the directory')
                return
            else:
                gum.AppUserMsg('Exception!',str(e)+' occurred.' )
                return
        self.scenario.save(path)
        self.antennaTx.save(path)
        self.antennaRx.save(path)
        self.freq_band.save(path)
        fading = fad.Fading3gpp(self.scenario,self.scatters_move,self.move_probability,self.v_min_scatters,self.v_max_scatters)

        performance.compute_path(fading,self.freq_band,self.antennaTx,self.antennaRx,self.positions,self.times,self.force_los,path,self.sspmode,self.phi_addTx,self.theta_addTx,self.phi_addRx,self.theta_addRx) 
        self.scenario_update = False

         
    def select_path_prb(self,prb):
        """ This is the callback function of the AppSelectPoint form to select the prb to analize the MS route.
        
        @type prb: int
        @param prb: the selected prb number.
        """      
        self.__window_path_prb.destroy()
        self.path_performance_analysis(prb)
          
    def graph_powermap(self):    
        """This method is called to graph the power map.
        
        """ 
        if self.scenario is not None and self.antennaTx is not None:
            shadow = True
            grs.graph_Txpower_map(self.antennaTx,self.scenario,shadow,self.positions,self.phi_addTx,self.theta_addTx)
        else:
            gum.AppUserMsg("Error message", "Please selct one scenario from the scenarios list and configure antenna Tx " )
             
    def function_select_scenario(self,cname,fcGHz,xmin,xmax,ymin,ymax,grid,bspos,ptx_db,force_los,MS_move,scatters_move,move_probability,v_min_scatters,v_max_scatters,sspmode):
        """ This is the callback function of the AppScenarios.
        
        @type cname: list.
        @param cname: A list with the slected scenarios.
        @type fcGHz: float .
        @param fcGHz: Frequency in GHz of the carrier frequency of the scenario.
        @type xmin: float .
        @param xmin: The minimum limit of the x coordinate in the scenario. 
        @type xmax: float .
        @param xmax: The maximum limit of the x coordinate in the scenario. 
        @type ymin: float .
        @param ymin: The minimum limit of the y coordinate in the scenario. 
        @type ymax: float .
        @param ymax: The maximum limit of the y coordinate in the scenario. 
        @type grid: int .
        @param grid: For calculating the spacial distribution of the parameters of the scenario, 
        the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate. 
        @type bspos: 3d array or list .
        @param bspos: The position of the Base Satation in the scenario in the coordinates system [x,y,z].
        @type ptx_db: float.
        @param ptx_db: The power transmited by the base station in dbm. 
        @type force_los: int
        @param force_los: This parameter can take value 0 if LOS condition is forced to NLOS, 1 if is forcd to LOS, and 2 if iti is calculated from the probability model.
        @type MS_move: Array
        @param MS_move: An array of the number of MSs. Each element has the position, velocity, path length and samples.
        @type move_probability: float
        @param move_probability: The probability that each scatter is moving.
        @type v_min_scatters: float
        @param v_min_scatters: The minimum velocity of one scatter movement.
        @type v_max_scatters: float
        @param v_max_scatters: The maximum velocity of one scatter movement.
        @type sspmode: int (0,1,2).
        @param sspmode: The ssp's spatial consistency  mode.
        
        """ 
        self.scenario_update = True
        self.cname = cname
        self.nMS = MS_move.shape[0]
        self.sspmode = sspmode
        """The ssp's spatial consistency  mode."""
        self.scatters_move = scatters_move
        """Boolean indicating if the scatters move or not in this simulation."""
        self.move_probability = move_probability
        self.v_min_scatters = v_min_scatters
        self.v_max_scatters = v_max_scatters
        self.positions = np.empty(shape=(self.nMS),dtype = object)
        """ An array of objects, where each object is the path of one MS."""
        self.times = np.empty(shape=(self.nMS),dtype = object)
        """ An array of objects, where each object is the times sequence of one MS."""
        max_x = xmin
        min_x = xmax
        max_y = ymin
        min_y = ymax
        for ms in range(self.nMS):
            mspos = MS_move[ms][0]
            msvel = MS_move[ms][1]
            length = MS_move[ms][2]
            """ The length of the MS route in meters.  """ 
            samples = MS_move[ms][3]
            """ The distance between sample points in the MS route. In meters. """ 
            points_in_paths = int(length/samples)
            mspositions = np.zeros((points_in_paths,3))
            """An array with the 3D positions of the MS in its route. """ 
            mstimes = np.zeros(points_in_paths)
            """ The times in second for each point in the MS route. """ 
            time = 0
            for i in range(points_in_paths):
                if(np.linalg.norm(msvel) != 0):
                    mspositions[i] = mspos + i*samples*np.array([np.cos(np.arctan2(msvel[1],msvel[0])),np.sin(np.arctan2(msvel[1],msvel[0])),0])
                    mstimes[i] = time
                    time = time +samples/np.linalg.norm(msvel)
                else:
                    mspositions[i] = mspos 
                    mstimes[i] = time
                    time = time + 0.01 # if the mobile is fixed, I take an arbitrary Delta_t
            self.positions[ms] = mspositions
            
            if  max(mspositions[:,0])> max_x:
                max_x = max(mspositions[:,0])
            if  max(mspositions[:,1])> max_y:
                max_y = max(mspositions[:,1])
            if  min(mspositions[:,0])< min_x:
                min_x = min(mspositions[:,0])
            if  min(mspositions[:,1])> min_y:
                min_y = min(mspositions[:,1])
            self.times[ms] = mstimes
        self.force_los = force_los
        if min_x >= 0:
            min_x = -1
        if max_x <= 0:
            max_x = 1
        if min_y >= 0:
            min_y = -1
        if max_y <= 0:
            max_y = 1
        xmin = min_x
        xmax = max_x
        ymin = min_y
        ymax = max_y

        if cname[0] == 0:
            self.scenario =  sc.Scenario3GPPInDoor(fcGHz,xmin,xmax,ymin,ymax,grid,bspos,ptx_db,True,self.force_los) 
        elif cname[0] == 1:
            self.scenario =  sc.Scenario3GPPUma(fcGHz,xmin,xmax,ymin,ymax,grid,bspos,ptx_db,True,self.force_los) 
        else:
            self.scenario =  sc.Scenario3GPPUmi(fcGHz,xmin,xmax,ymin,ymax,grid,bspos,ptx_db,True,self.force_los)             
        self.__window_sc.destroy()
        #self.function_scenario(self.cname,self.fcGHz,self.xmin,self.xmax,self.ymin,self.ymax,self.grid,self.bspos,self.ptx_db)
        
    def function_tx_antenna(self,antenna,ant_type,phi_add,theta_add):
        """ This is the callback function of the AppAntenna to configure the TX antenna.

        @type antenna: Antenna Class.
        @param antenna: The Tx antenna object.
        @type ant_type: string
        @param ant_type: the type of the antenna.
        @type phi_add: float
        @param phi_add: The additional azimnuth rotation angle for beamforming.
        @type theta_add: float
        @param theta_add: The additional inclination rotation angle for beamforming.      
        """ 
        self.antennaTx = antenna
        self.ant_typeTx = ant_type
        """ The type of the antenna Tx element. """ 
        self.theta_addTx = theta_add
        """ Additional Tx inlination rotation angle for beamforming. """ 
        self.phi_addTx = phi_add
        """ Additional Tx azimuth rotation angle for beamforming. """ 
        self.scenario_update = True
        self.__window_txa.destroy()

    def function_rx_antenna(self,antenna,ant_type,phi_add,theta_add):
        """ This is the callback function of the AppAntenna to configure the RX antenna.

        @type antenna: Antenna Class.
        @param antenna: The Rx antenna object.
        @type ant_type: string
        @param ant_type: the type of the antenna.
        @type phi_add: float
        @param phi_add: The additional azimnuth rotation angle for beamforming.
        @type theta_add: float
        @param theta_add: The additional inclination rotation angle for beamforming.       
        """ 
        self.antennaRx = antenna
        self.ant_typeRx = ant_type
        """ the type of the antenna Rx element. """ 
        self.theta_addRx = theta_add
        """ Additional Rx azimuth rotation angle for beamforming. """ 
        self.phi_addRx = phi_add
        """ Additional Rx inlination rotation angle for beamforming. """ 
        self.scenario_update = True
        self.__window_rxa.destroy()
            
    def function_freq_band(self,nprb,bwprb,noisefig,thnoise):
        """ This is the callback function of the AppAFreqBand to configure the OFDM frequency band.
        
        @type nprb: int.
        @param nprb: The number of physical reseource blocks (PRB) in OFDM. Default 100.
        @type bwprb: float.
        @param bwprb: The bandwidth of each physical reseource blocks in OFDM. In Hertz.
        Default 180000.
        @type noisefig: float.
        @param noisefig :The noise figure in db.
        @type thnoise: float.
        @param thnoise:The thermal noise in dbm per Hertz.
        """ 
        self.scenario_update = True
        self.freq_band = fband.FrequencyBand(self.scenario.fcGHz,nprb,bwprb,noisefig,thnoise)
        self.__window_fb.destroy()

        
    def graph_H_f(self,point,prb):
        """ This is the method to graph the channl matrix properties. 
        
        @type point: int
        @param point: the selected point number.
        @type prb: int.
        @param prb: The selected prb.
        """
        if self.directory_name is not None:
            try:
                scenario_params = np.loadtxt(self.directory_name+'/scenario.csv',delimiter=',')
                antennaTx = np.loadtxt(self.directory_name+'/antennaTx.csv',delimiter=',')
                antennaRx = np.loadtxt(self.directory_name+'/antennaRx.csv',delimiter=',')
                text_file = open(self.directory_name+'/antenna_elementantennaTx_type.csv', "r")
                Tx_element = text_file.read() 
                text_file.close()
                text_file = open(self.directory_name+'/antenna_elementantennaRx_type.csv', "r")
                Rx_element = text_file.read() 
                text_file.close()
                
                aTx_n_elements = antennaTx[0]
                aRx_n_elements = antennaRx[0]                
                fc = scenario_params[0]
                ptxdb = scenario_params[6]
                text_file = open(self.directory_name+'/scenario_name.csv', "r")
                sc_name = text_file.read() 
                text_file.close()
                lsp = np.loadtxt(self.directory_name+'/lsp_'+str(self.MS_number)+'_'+str(point)+'.csv', delimiter=',')
                LOS = lsp[0]
                
                H = np.load(self.directory_name+'/H_f'+str(self.MS_number)+'.npy')
                u,s,vh = np.linalg.svd(H[point,prb,:,:])
                if LOS == 1:
                    los_cond = " LOS "
                else:
                    los_cond = " NLOS "
                subtitle = "\n Scenario: " + sc_name+ ", fc (GHz): " + str(fc) + ", PTx(dbm): " +str(ptxdb)+ ", LOS condition:  " + los_cond + "\n Antenna Tx: "+ Tx_element + ", Tx number of elements: " + str(aTx_n_elements) + "\n Antenna Rx: "+ Rx_element + ", Rx number of elements: " + str(aRx_n_elements)
                grs.graph_H_f(s, prb, point,subtitle)
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
            gum.AppUserMsg("Error message", " Please, first select the directory of the simulation " )

        
    def lsp_multipath_analysis(self,lsp_param):
        """ This is the callback function to graph the LSP parameters.
        
        @type lsp_param: int.
        @param lsp_param: The number of the LSP selected.
        """         
        self.__window_lsp.destroy()
        param_name = ["Sadowing in db" , "Ricean K in db " , " Delay Spread in log10(DS/1s) ", "Azimuth Departure Angle in log10(ADA/1º) ", "Azimuth Arrival Angle in log10(AAA/1º) ", " Zenith Departure Angle in log10(ZDA/1º) ", " Zenith Arrival Angle in log10(ZAA/1º) ", "LOS condition" ]
        #title_end = " for los condition: "
        if self.directory_name is not None:
            try:
                scenario_params = np.loadtxt(self.directory_name+'/scenario.csv',delimiter=',')
                posx_min = scenario_params[1]
                posx_max = scenario_params[2]
                posy_min = scenario_params[3]
                posy_max = scenario_params[4]
                grid_number = int(scenario_params[5])
                
                stepx = (posx_max-posx_min)/grid_number
                stepy = (posy_max-posy_min)/grid_number
                x = np.linspace(posx_min,posx_max+stepx,grid_number+1) # 2*self.grid_number)*(self.posx_max-self.posx_min)/(2*self.grid_number-1)+self.posx_min
                y = np.linspace(posy_min,posy_max+stepy,grid_number+1) #np.arange(0, 2*self.grid_number)*(self.posy_max-self.posy_min)/(2*self.grid_number-1)+self.posy_min
                X, Y = np.meshgrid(x, y) 
                fc = scenario_params[0]
                text_file = open(self.directory_name+'/scenario_name.csv', "r")
                sc_name = text_file.read() 
                text_file.close()
                if lsp_param <7:
                    gridlsp_los = np.load(self.directory_name+'/scenario_gridLSP.npy')
                    param = gridlsp_los[lsp_param]
                    title = param_name[lsp_param]#+title_end
                else:
                    param = np.load(self.directory_name+'/scenario_gridLOS.npy')
                    title = param_name[lsp_param]#+title_end
                # else:    
                #     gridlsp_nlos = np.load(self.directory_name+'/scenario_gridLSP_NLOS.npy') 
                #     if lsp_param == 1:
                #         gum.AppUserMsg("Error message", " Ricean K is not defined in NLOS scenario" )
                #     else:
                #         if lsp_param == 0:
                #             param = gridlsp_nlos[lsp_param ]
                #         else:
                #             param = gridlsp_nlos[lsp_param -1]
                #         title = param_name[lsp_param]+title_end+ " NLOS"             
                grs.graph_params_scenario(X,Y,param,title,fc,sc_name)
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )

            
    def ssp_multipath_analysis(self,param,point):   
        """ This is the callback function to graph the SSP parameters.
        
        @type param: int.
        @param param: The number of the SSP selected.
        @type point: int.
        @param point: The number of the point in the MS route.
        """                 
        self.__window_ssp.destroy()       
        if self.directory_name is not None:
           try: 
              scenario_params = np.loadtxt(self.directory_name+'/scenario.csv',delimiter=',')
              antennaTx = np.loadtxt(self.directory_name+'/antennaTx.csv',delimiter=',')
              antennaRx = np.loadtxt(self.directory_name+'/antennaRx.csv',delimiter=',')
              text_file = open(self.directory_name+'/antenna_elementantennaTx_type.csv', "r")
              Tx_element = text_file.read() 
              text_file.close()
              text_file = open(self.directory_name+'/antenna_elementantennaRx_type.csv', "r")
              Rx_element = text_file.read() 
              text_file.close()      
              aTx_n_elements = antennaTx[0]
              aRx_n_elements = antennaRx[0]                
              fc = scenario_params[0]
              ptxdb = scenario_params[6]
              text_file = open(self.directory_name+'/scenario_name.csv', "r")
              sc_name = text_file.read() 
              text_file.close()
              lsp = np.loadtxt(self.directory_name+'/lsp_'+str(self.MS_number)+'_'+str(point)+'.csv', delimiter=',')
              LOS = lsp[0]
              if LOS == 1:
                  los_cond = " LOS "
              else:
                  los_cond = " NLOS "
              subtitle = "\n Scenario: " + sc_name+ ", fc (GHz): " + str(fc) + ", PTx(dbm): " +str(ptxdb)+ ", LOS condition:  " + los_cond + "\n Antenna Tx: "+ Tx_element + ", Tx number of elements: " + str(aTx_n_elements) + "\n Antenna Rx: "+ Rx_element + ", Rx number of elements: " + str(aRx_n_elements)

              if param == 0:
                  tau = np.loadtxt(self.directory_name+'/tau_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                  n = tau.size
                  grs.graph_ssp('Cluster delay for point '+str(self.MS_number)+'_'+str(point)+ subtitle," relative delay",n,tau )
              if param == 1: 
                  if LOS == 0:
                      P = np.loadtxt(self.directory_name+'/PDP_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                      n = P.size
                  else:
                      P = np.loadtxt(self.directory_name+'/PDP_LOS_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                      n = P.size
                       
                  grs.graph_ssp('Power Delay Profile for point '+str(self.MS_number)+'_'+str(point)+ subtitle," Power (normalized to 1)",n,P )
              if param == 2:
                  phiAOA = np.loadtxt(self.directory_name+'/PHI_AOA_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                  n = phiAOA.size
                  grs.graph_ssp('Cluster Azimuth angle of arrival for point '+str(self.MS_number)+'_'+str(point)+ subtitle," Azimuth AOA in degrees",n,phiAOA )
              if param == 3:
                  phiAOD = np.loadtxt(self.directory_name+'/PHI_AOD_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                  n = phiAOD.size
                  grs.graph_ssp('Cluster Azimuth angle of departure for point '+str(self.MS_number)+'_'+str(point)+subtitle," Azimuth AOD in degrees",n,phiAOD )
              if param == 4:
                  thetaAOA = np.loadtxt(self.directory_name+'/THETA_AOA_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                  n = thetaAOA.size
                  grs.graph_ssp('Cluster Inclination angle of arrival for point '+str(self.MS_number)+'_'+str(point)+subtitle," Inclination AOA in degrees",n,thetaAOA )
              if param == 5:
                  thetaAOD = np.loadtxt(self.directory_name+'/THETA_AOD_'+str(self.MS_number)+'_'+str(point)+'.csv',delimiter=',')
                  n = thetaAOD.size
                  grs.graph_ssp('Cluster Inclination angle of departure for point '+str(self.MS_number)+'_'+str(point)+subtitle," Inclination AOD in degrees",n,thetaAOD )
           except BaseException as error:
               gum.AppUserMsg('Exception occurred!','{}'.format(error))
        else:
           gum.AppUserMsg("Error message", " Please, first select the directory of the simulation " )
       
    def performance_analysis(self,point):   
       """ This is the callback function to graph the performance analysis in one point of the MS route.

       @type point: int.
       @param point: The number of the point in the MS route.
       """                  
       if self.directory_name is not None:
            try:
                scenario_params = np.loadtxt(self.directory_name+'/scenario.csv',delimiter=',')
                antennaTx = np.loadtxt(self.directory_name+'/antennaTx.csv',delimiter=',')
                antennaRx = np.loadtxt(self.directory_name+'/antennaRx.csv',delimiter=',')
                text_file = open(self.directory_name+'/antenna_elementantennaTx_type.csv', "r")
                Tx_element = text_file.read() 
                text_file.close()
                text_file = open(self.directory_name+'/antenna_elementantennaRx_type.csv', "r")
                Rx_element = text_file.read() 
                text_file.close()
                
                aTx_n_elements = antennaTx[0]
                aRx_n_elements = antennaRx[0]                
                fc = scenario_params[0]
                ptxdb = scenario_params[6]
                text_file = open(self.directory_name+'/scenario_name.csv', "r")
                sc_name = text_file.read() 
                text_file.close()
                lsp = np.loadtxt(self.directory_name+'/lsp_'+str(self.MS_number)+'_'+str(point)+'.csv', delimiter=',')
                LOS = lsp[0]
 
                H = np.load(self.directory_name+'/H_f'+str(self.MS_number)+'.npy')
                rxpsd = np.loadtxt(self.directory_name+'/rxpsd'+str(self.MS_number)+'.csv',delimiter=',')
                n_BS = int(antennaTx[0])
                n_MS = int(antennaRx[0])
                if LOS == 1:
                    los_cond = " LOS "
                else:
                    los_cond = " NLOS "
                subtitle = "\n Scenario: " + sc_name+ ", fc (GHz): " + str(fc) + ", PTx(dbm): " +str(ptxdb)+ ", LOS condition:  " + los_cond + "\n Antenna Tx: "+ Tx_element + ", Tx number of elements: " + str(aTx_n_elements) + "\n Antenna Rx: "+ Rx_element + ", Rx number of elements: " + str(aRx_n_elements)

                grs.graph_performance(point, rxpsd, n_MS, n_BS, H,subtitle)
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
       else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )

    def path_performance_analysis(self,prb): 
       """ This is the callback function to graph the path performance analysis.

       @type prb: int.
       @param prb: The number of the prb in the OFDM band.
       """                  
       if self.directory_name is not None:
            try:
                scenario_params = np.loadtxt(self.directory_name+'/scenario.csv',delimiter=',')
                antennaTx = np.loadtxt(self.directory_name+'/antennaTx.csv',delimiter=',')
                antennaRx = np.loadtxt(self.directory_name+'/antennaRx.csv',delimiter=',')
                text_file = open(self.directory_name+'/antenna_elementantennaTx_type.csv', "r")
                Tx_element = text_file.read() 
                text_file.close()
                text_file = open(self.directory_name+'/antenna_elementantennaRx_type.csv', "r")
                Rx_element = text_file.read() 
                text_file.close()
                
                aTx_n_elements = antennaTx[0]
                aRx_n_elements = antennaRx[0]                
                fc = scenario_params[0]
                ptxdb = scenario_params[6]
                text_file = open(self.directory_name+'/scenario_name.csv', "r")
                sc_name = text_file.read() 
                text_file.close()
                los = np.loadtxt(self.directory_name+'/los'+str(self.MS_number)+'.csv',delimiter=',')
                H = np.load(self.directory_name+'/H_f'+str(self.MS_number)+'.npy')
                snr = np.loadtxt(self.directory_name+'/snr'+str(self.MS_number)+'.csv',delimiter=',')
                linear_losses = np.loadtxt(self.directory_name+'/linear_losses'+str(self.MS_number)+'.csv',delimiter=',')
                snr_pl = np.loadtxt(self.directory_name+'/snr_pl'+str(self.MS_number)+'.csv',delimiter=',')
                snr_pl_shadow = np.loadtxt(self.directory_name+'/snr_pl_shadow'+str(self.MS_number)+'.csv',delimiter=',')
                times = np.loadtxt(self.directory_name+'/times'+str(self.MS_number)+'.csv',delimiter=',')

                sp_eff = np.loadtxt(self.directory_name+'/spectral_eff'+str(self.MS_number)+'.csv',delimiter=',')                
                positions = np.loadtxt(self.directory_name+'/positions'+str(self.MS_number)+'.csv',delimiter=',')
                n_BS = int(antennaTx[0])
                n_MS = int(antennaRx[0])
                subtitle = "\nMS number: " +str(self.MS_number) + ", Scenario: " + sc_name+ ", fc (GHz): " + str(fc) + ", PTx(dbm): " +str(ptxdb)+ "\n Antenna Tx: "+ Tx_element + ", Tx number of elements: " + str(aTx_n_elements) + "\n Antenna Rx: "+ Rx_element + ", Rx number of elements: " + str(aRx_n_elements)

                grs.graph_path_performance(los,positions, snr, H, n_MS, n_BS, prb,snr_pl,sp_eff,subtitle,linear_losses,snr_pl_shadow,times)
            except BaseException as error:
                gum.AppUserMsg('Exception occurred!','{}'.format(error))
       else:
            gum.AppUserMsg('Error','Please, first select the directory of the simulation to analyze' )

   

if __name__ == "__main__":
    window = tk.Tk()

  
    app = AppGraph(window,"PyWiCh")
    try:
        window.mainloop()
    except:
        print('Exception!', sys.exc_info()[0],'occurred.' )


