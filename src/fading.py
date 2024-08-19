#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module computes the fading model of differents scenarios.

@author: pablo belzarena
"""

import numpy as np
import angles as ang
from time import perf_counter


class Fading:
  """This class implments the base fading model.
  
  This class defins differents methods common to all fading models.
  """  
  def __init__(self,scenario):
    """
    The constructor method of the Fading Class.
    This method sets the scenario of the fading model and generates an empty short scale parameters grid.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    """
    self.scenario = scenario
    """ The scenario for the fading model.""" 
    self.__ssp_grid = np.empty(shape=(self.scenario.X[0].size,self.scenario.Y[:,0].size), dtype=object)
    """ An array of objects with one element for each point of the grid. Each element is an SSPs object. """

    
  def compute_ssps(self,pos):
        """
        This method computes the ssps for this position. It is an abstract method. It must be implemented
        in the child classes.
        
        @type pos: array.
        @param pos: The position of the MS.
        """ 
   
  def set_correlated_ssps(self,pos,ret_ssp):
    """ This method using the ssps grid, computes the ssps for this position in the scenario. It
    also sets the ssps for this fading object.

    TODO: uncouple LSP and ssp grids.
    @type pos: Array.
    @param pos: the psoition in the scenario to interpolate.
    @type ret_ssp: SSP object.
    @param ret_ssp: the SSP object to return the SSP parameters of the position pos.
    """
    i_x1,i_x2,i_y1,i_y2 = self.find_point_meshgrid(pos,self.scenario.X,self.scenario.Y)
    ########
    los = self.scenario.is_los_cond(pos)
    ######
    p00 = [self.scenario.X[0][i_x1],self.scenario.Y[i_y1][0],pos[2]]
    p01 = [self.scenario.X[0][i_x1],self.scenario.Y[i_y2][0],pos[2]]
    p10 = [self.scenario.X[0][i_x2],self.scenario.Y[i_y1][0],pos[2]]
    p11 = [self.scenario.X[0][i_x2],self.scenario.Y[i_y2][0],pos[2]]
    los_p00 = self.scenario.is_los_cond(p00)
    los_p01 = self.scenario.is_los_cond(p01)
    los_p10 = self.scenario.is_los_cond(p10)
    los_p11 = self.scenario.is_los_cond(p11)
    points = []
    ssps_grid = []

    if los_p00 == los: 
        if self.__ssp_grid[i_x1][i_y1]==None:
            self.__ssp_grid[i_x1][i_y1] = self.compute_ssps(p00)
        points.append(p00)
        ssps_grid.append((self.__ssp_grid[i_x1][i_y1]))
        
    if los_p01 == los:
        if self.__ssp_grid[i_x1][i_y2]==None:
            self.__ssp_grid[i_x1][i_y2] = self.compute_ssps(p01)
        points.append(p01)
        ssps_grid.append((self.__ssp_grid[i_x1][i_y2]))
    if los_p10 == los:
        if self.__ssp_grid[i_x2][i_y1]==None:
            self.__ssp_grid[i_x2][i_y1] = self.compute_ssps(p10)            
        points.append(p10)
        ssps_grid.append((self.__ssp_grid[i_x2][i_y1]))
    if los_p11 == los:
        if self.__ssp_grid[i_x2][i_y2]==None:
            self.__ssp_grid[i_x2][i_y2] = self.compute_ssps(p11)
        points.append(p11)
        ssps_grid.append((self.__ssp_grid[i_x2][i_y2]))
    if len(ssps_grid) > 0:
        ####################################################
        #ret_ssp = SSPs3GPP(15,ssps_grid[0].n_scatters)
        ####################################################
        for i in range(ssps_grid[0].number_sps):
            values = []
            for ssp in ssps_grid:
                values.append(ssp.ssp_array[i])    
            ret_ssp.ssp_array[i] = self.inverse_distance_interpol(pos, np.array(points),np.array(values))
    else:
            ret_ssp = self.compute_ssps(pos)
    return ret_ssp  

  def inverse_distance_interpol(self,point, XY,values, p = 2):
    """
    This method interpolates one point with the values in the XY points using inverse distance interpolation with module p.
    
    @type point: array.
    @param point: The point for the interpolation.
    @type XY: array
    @param XY: a matrix where each row is a point of the grid.
    @type values: array
    @param values: an array with the values in each XY point.
    @return: the value interpolated.
    """
    d = np.sqrt ((point[0] - XY[:,0]) ** 2 +(point[1] - XY[:,1]) ** 2) ** p
    if d.min () == 0:
      ret = values[np.unravel_index(d.argmin(), d.shape)]
    else:
      w = 1.0 / d
      shape = values.shape
      aux = (values.T * w).T #[:,None]
      aux = aux.reshape(shape)
      sumcols = np.sum (aux,axis=0)
      ret = sumcols / np.sum (w)
    return ret
 
  def find_point_meshgrid(self,pos,X,Y):
    """ This method given a position in the scenario, finds the  square (the four vertices)
    of the grid where the point is inside.

    @type pos: array.
    @param pos: The position of the MS in the scenario.
    @type X: array
    @param X: a matrix of the x coordinates of the grid. results of X,Y = np.meshgrid(x,y)
    @type Y: array
    @param Y: a matrix of the y coordinates of the grid. results of X,Y = np.meshgrid(x,y)
    @return: x1,x2,y1,y2 - the x and y coordinates of the four vertices.
    """       
    index_x1 = np.argmin(np.abs(X[0]-pos[0]))
    if pos[0] < X[0][index_x1]:
        if index_x1 == 0:
            index_x2 = 0
        else:
            index_x2 = index_x1
            index_x1 = index_x1 -1
    elif pos[0] > X[0][index_x1]:
       if index_x1 == X[0].size-1:
           index_x2 = index_x1
       else:
           index_x2 = index_x1+1
    else:
        index_x2 = index_x1
   
    index_y1 = np.argmin(np.abs(Y[:,0]-pos[1]))
    if pos[1] < Y[:,0][index_y1]:
        if index_y1 == 0:
            index_y2 = 0
        else:
            index_y2 = index_y1
            index_y1 = index_y1 -1
    elif pos[1] > Y[:,0][index_y1]:
       if index_y1 == Y[:,0].size-1:
           index_y2 = index_y1
       else:
           index_y2 = index_y1+1
    else:
        index_y2 = index_y1
    return(index_x1,index_x2,index_y1,index_y2)


class FadingSiSoRayleigh(Fading):
  """This class implments a simple SISO Rayleigh fading model using sum of sinusoids.
  
  """  
  def __init__(self,scenario,number_sin):
    """
    The constructor method of the  Fading Rayleigh Class.
    This method sets the scenario and generates the ssp object.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    """
    super().__init__(scenario)
    
    self.number_sin = number_sin # number of sinusoids to sum
    self.ssp = self.compute_ssps([0,0,0])
    
  def compute_ssps(self,pos):
    """
    This method computes the ssps for this position.
    
    It calls the methods to compute the Small Scale Parameters. The ssps parameters are three in this model.
    The number of sinusoids, the alpha and phi angles of each sinusoid. The position is not used in this model.
    
    @type pos: array.
    @param pos: The position of the MS.
    """ 
    ssp = SSPsRayleigh(3,self.number_sin) 
    return ssp

  def compute_ch_matrix(self,MS_pos,MS_vel,aMS,aBS,t,mode=0):
    """ 
    This method computes  the channel matrix according to the calculation mode.
    
    This method sets the MS_pos, the MS_vel, and the simulation time.
    The amplitud of each sinusoid is an independent random variable for each point and each sinusoid. 

    @type MS_pos: 3D array or list.
    @param MS_pos: The position of the mobile device in the scenario.
    @type MS_vel: 3D array or list.
    @param MS_vel: the velocity vector of the mobile device in the scenario.
    @type aMS: Class Antenna.
    @param aMS: The MS antenna.
    @type aBS: Class Antenna.
    @param aBS: The BS antenna.  
    @type t: float.
    @param t: The cuurent time of the simulation in seconds.
    @type mode: int.
    @param mode: If mode =0 the ssps are generated for the MS_pos interpolating in the correlated ssp grid. If mode = 1
    uses the same alpha and phi for all points. If mode = 2 the alpha and phi are generated with independet random variables
    for each point.

    """ 

    self.MS_pos = MS_pos
    self.MS_vel = MS_vel
    self.MS_t = t
    self.H_usn = np.zeros((1,1,1), dtype=complex)  
    self.tau = np.zeros(1)
    self.tau[0] =  np.sqrt(MS_pos[0]**2+MS_pos[1]**2+MS_pos[2]**2)/3e8
    # Simulation Params, feel free to tweak these
    v = np.sqrt(MS_vel[0]**2+MS_vel[1]**2+MS_vel[2]**2)
    fd = v*self.scenario.fcGHz/3e8 # max Doppler shift
    #t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)
    x = 0 #np.zeros(len(t))
    y = 0 #np.zeros(len(t))
    if mode == 0:
        ssps = self.compute_ssps([0,0,0])
        self.ssp = self.set_correlated_ssps(MS_pos,ssps)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
        # ssp = self.compute_ssps(MS_pos)
        # alpha[self.number_sin-1] = ssp.alpha[self.number_sin-1]
        # phi[self.number_sin-1] = ssp.phi[self.number_sin-1]

    elif mode == 1:    
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    else:
        self.ssp = self.compute_ssps(MS_pos)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    for i in range(self.number_sin):
        #x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)
        #y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)
        x = x + (np.random.randn()+1)* np.cos(2 * np.pi * fd * t * np.cos(alpha[i])) * np.cos( phi[i])
        y = y + (np.random.randn()+1)* np.sin(2 * np.pi * fd * t * np.cos(alpha[i])) * np.sin(phi[i])
    self.H_usn[0] = (np.sqrt(2/self.number_sin)) * (x + 1j*y)
    return self.H_usn   


class FadingSiSoRician(Fading):
  """This class implments a simple SISO Rician fading model using sum of sinusoids
  """  
  def __init__(self,scenario,number_sin,K_LOS):
    """
    The constructor method of the  Fading Rician Class.
    This method sets the scenario and generates the ssp object.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    @type K_LOS: Cfloat
    @param K_LOS: The K cosntant of the Rice model.
     
    """
    super().__init__(scenario)

    
    self.number_sin = number_sin # number of sinusoids to sum
    self.ssp = self.compute_ssps([0,0,0])
    self.K = K_LOS
    
  def compute_ssps(self,pos):
    """
    This method computes the ssps for this position.
    
    It calls the methods to compute the Small Scale Parameters. The ssps parameters are three in this model.
    The number of sinusoids, the alpha and phi angles of each sinusoid. The position is not used in this model.
    
    @type pos: array.
    @param pos: The position of the MS.
    """ 
    ssp = SSPsRayleigh(3,self.number_sin) 
    return ssp


  def compute_ch_matrix(self,MS_pos,MS_vel,aMS,aBS,t,mode=0,phase_LOS=0,phase_ini=0):
    """ 
    This method computes  the channel matrix according to the calculation mode.
    
    This method sets the MS_pos, the MS_vel, and the simulation time.
    The amplitud of each sinusoid is an independent random variable for each point and each sinusoid. 

    @type MS_pos: 3D array or list.
    @param MS_pos: The position of the mobile device in the scenario.
    @type MS_vel: 3D array or list.
    @param MS_vel: the velocity vector of the mobile device in the scenario.
    @type aMS: Class Antenna.
    @param aMS: The MS antenna.
    @type aBS: Class Antenna.
    @param aBS: The BS antenna.  
    @type t: float.
    @param t: The cuurent time of the simulation in seconds.
    @type mode: int.
    @param mode: If mode =0 the ssps are generated for the MS_pos interpolating in the correlated ssp grid. If mode = 1
    uses the same alpha and phi for all points. If mode = 2 the alpha and phi are generated with independet random variables
    for each point.
    @type phase_LOS: float in [−pi,pi).
    @param phase_LOS: The arrival phase of the LOS ray.
    @type phase_ini:: float in [−pi,pi).
    @param phase_ini: the initial phase of the LOS ray. Tipically is a random variable uniformly distributed over [−pi,pi). 

    
    """ 

    self.MS_pos = MS_pos
    self.MS_vel = MS_vel
    self.MS_t = t
    self.H_usn = np.zeros((1,1,1), dtype=complex)  
    self.tau = np.zeros(1)
    self.tau[0] =  np.sqrt(MS_pos[0]**2+MS_pos[1]**2+MS_pos[2]**2)/3e8
    # Simulation Params, feel free to tweak these
    v = np.sqrt(MS_vel[0]**2+MS_vel[1]**2+MS_vel[2]**2)
    fd = v*self.scenario.fcGHz/3e8 # max Doppler shift
    #t = np.arange(0, 1, 1/Fs) # time vector. (start, stop, step)
    x = 0 #np.zeros(len(t))
    y = 0 #np.zeros(len(t))
    if mode == 0:
        ssps = self.compute_ssps([0,0,0])
        self.ssp = self.set_correlated_ssps(MS_pos,ssps)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
        # ssp = self.compute_ssps(MS_pos)
        # alpha[self.number_sin-1] = ssp.alpha[self.number_sin-1]
        # phi[self.number_sin-1] = ssp.phi[self.number_sin-1]

    elif mode == 1:    
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    else:
        self.ssp = self.compute_ssps(MS_pos)
        alpha = self.ssp.alpha
        phi = self.ssp.phi
    for i in range(self.number_sin):
        #x = x + np.random.randn() * np.cos(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)
        #y = y + np.random.randn() * np.sin(2 * np.pi * fd * t * np.cos(self.alpha) + self.phi)
        x = x + (np.random.randn()+1)* np.cos(2 * np.pi * fd * t * np.cos(alpha[i])) * np.cos( phi[i])
        y = y + (np.random.randn()+1)* np.sin(2 * np.pi * fd * t * np.cos(alpha[i])) * np.sin(phi[i])
    z = (np.sqrt(2/self.number_sin)) * (x + 1j*y) # this is what you would actually use when simulating the channel
    w = np.exp(1j*(2 * np.pi * fd * t * np.cos(phase_LOS)+phase_ini))
    self.H_usn[0] = z/np.sqrt(self.K+1)+w*np.sqrt(self.K/(self.K+1))
    return self.H_usn  



class Fading3gpp(Fading):
  """This class implments the 3gpp fading model
  """  
  def __init__(self,scenario,scatters_move=False,move_probability =0,v_min_scatters=0,v_max_scatters=0):
    """
    The constructor method of the  Fading3gpp Class.
    This method sets the scenario, and the antennas. It also creates an empty a ssps grid
    used for interpolation mode.
    
    @type scenario: Class Scenario.
    @param scenario: The scenario for the fast fading model.
    @type scatters_move: Boolean
    @param scatters_move: If the scatters can move or not
    @type move_probability: float between 0 and 1.
    @param move_probability: If the scatters can move, the probability that each scatter moves or not.
    @type v_min_scatters: float.
    @param v_min_scatters: The minimum velocity of the scatters if they can move.
    @type v_max_scatters: float.
    @param v_max_scatters: The maximum velocity of the scatters if they can move.

    """
    super().__init__(scenario)
    self.scatters_move = scatters_move
    """ If the scatters can move or not """
    self.move_probability = move_probability
    """ If the scatters can move, the probability that each scatter moves or not. """ 
    self.v_min_scatters = v_min_scatters
    """The minimum velocity of the scatters if they can move. """ 
    self.v_max_scatters = v_max_scatters
    """ The maximum velocity of the scatters if they can move. """ 

  @property  
  def tau(self):
    """ Gets the array of relative delays of each cluster """ 
    return self.__tau 

  def compute_ch_matrix(self,MS_pos,MS_vel,MS_antenna,BS_antenna,t=0,mode=0):
    """ 
    This method computes the ssps and the channel matrix according to the calculation mode.

    This method sets the MS_pos, the MS_vel, and the simulation time.
    and sets the channel matrix for each cluster and each antenna elements pairs.If mode is equal to
    zero it calls compute_ssps and generates a new set of ssps for this point. If mode is equal to one
    it calls set_correlated_ssps and interpolates ssps  grid for this point.

    @type MS_pos: 3D array or list.
    @param MS_pos: The position of the mobile device in the scenario.
    @type MS_vel: 3D array or list.
    @param MS_vel: the velocity vector of the mobile device in the scenario.
    @type MS_antenna: Class Antenna.
    @param MS_antenna: The MS antenna.
    @type BS_antenna: Class Antenna.
    @param BS_antenna: The BS antenna.  
    @type t: float.
    @param t: The cuurent time of the simulation in seconds.
    @type mode: int.
    @param mode: If mode =0 the ssps are generated for the MS_pos using the 3gpp standard equations for that point. If mode = 1
    the ssps are generated for the MS_pos interpolating for this position with the ssps generateed for a grid in the scenario.
    """ 
    self.MS_antenna = MS_antenna
    """ The MS antenna model."""
    self.BS_antenna = BS_antenna
    """The BS antenna model """ 
    time1 = perf_counter()
    self.__set_lsps(MS_pos)
    self.MS_pos  = MS_pos
    """The MS current position """ 
    self.MS_vel = MS_vel
    """ The MS current velocity vector""" 
    self.t = t
    """ The current time of the simulation """  
    self.filter_scatters = True
    time2 = perf_counter()
    if mode == 0:
        self.compute_ssps(MS_pos)
    else:
        ############################################
        self.filter_scatters = False
        ##############################################
        ssps = SSPs3GPP(17)
        ssps.n_scatters = self.scenario.n_scatters
        ssps.reduced_n_scatters = self.scenario.n_scatters
        ret = self.set_correlated_ssps(MS_pos,ssps)
        self.__filter_scatters(ret)
        self.set_ssps(ret)
    time3 = perf_counter()
    c1,c2 = self.__find_strongest_clusters()
    self.__cluster1 = c1
    """The cluster with the strongest power """ 
    self.__cluster2 = c2
    """ The cluster with the second strogest power """ 
    self.H_usn = self.__generateChannelCoeff()
    time4 = perf_counter()

    """Channel matrix for each Tx-Rx antenna elements pair and for each cluster """
    if self.scatters_move:#if scatters move in the following updates the velocity vector of the scatters is substracted to the MS velocity vector 
        self.__set_scatters_move()
    time5 = perf_counter()
          
          
  def __set_scatters_move(self):
    """ This method sets the velocity of each cluster according to the probability model.
    
    """
    self.vel_scatters = np.zeros((self.reduced_n_scatters,3))
    """ The array with the velocities of each scatter""" 
    move_rvs = np.random.uniform(low=0, high=1, size=self.reduced_n_scatters)
    v = np.random.uniform(low=self.v_min_scatters, high=self.v_max_scatters, size=self.reduced_n_scatters)
    for i in range(self.reduced_n_scatters):
        if(move_rvs[i] < self.move_probability):
            self.vel_scatters[i] = np.random.uniform(-1,1,3)
            self.vel_scatters[i] = self.vel_scatters[i]/np.linalg.norm(self.vel_scatters[i])*v[i]
        else:
            self.vel_scatters[i] = np.zeros(3)

  



  def __set_lsps(self,pos):
    self.__los = self.scenario.is_los_cond(pos)
    lsp = self.scenario.generate_correlated_LSP_vector(pos,0)
    if self.__los:
      self.__sigma_shadow  = lsp[0]  
      """Standard deviation of shadowing. """ 
      self.__sigma_K = lsp[1]
      """Standard deviation of Ricean K. """
      self.__sigma_tau = 10**lsp[2]
      """Standard deviation of the delay spread. """
      self.__sigma_AOD_AZS = 10**lsp[3]
      """Standard deviation of the azimuth departure angle spread. """
      self.__sigma_AOA_AZS = 10**lsp[4]
      """Standard deviation of the azimuth arrival angle spread. """
      self.__sigma_AOD_ELS = 10**lsp[5]
      """Standard deviation of the inclination departure angle spread. """
      self.__sigma_AOA_ELS = 10**lsp[6]
      """Standard deviation of the inclination arrival angle spread. """
    else:
      self.__sigma_shadow  = lsp[0]     
      self.__sigma_tau = 10**lsp[1]
      self.__sigma_AOD_AZS = 10**lsp[2]
      self.__sigma_AOA_AZS = 10**lsp[3]
      self.__sigma_AOD_ELS = 10**lsp[4]
      self.__sigma_AOA_ELS = 10**lsp[5]
      self.__sigma_K = 0
    
    self.__sigma_AOD_AZS = min(self.__sigma_AOD_AZS, 104.0) 
    self.__sigma_AOA_AZS = min(self.__sigma_AOA_AZS, 104.0)
    self.__sigma_AOD_ELS = min(self.__sigma_AOD_ELS, 52.0);
    self.__sigma_AOD_ELS = min(self.__sigma_AOD_ELS, 52.0);
    self.__attenuation_dB = 0  
    """ The attenuation in case of blockage. Not implemented yet. """ 
   

  def compute_ssps(self,pos):
        """
        This method computes the ssps for this position.
        
        This method sets the LOS condition for the position of the MS , and sets 
        the correlated Large Scale Parameters (LSP) for this MS position. 
        The method using the LSP parameters for the MS position, assigns the variance of the
        delay spread, the angles spread, Riccean K and shadowing. Finally it calls all methods 
        to computes the Small Scale Parameters.
        @type pos: array.
        @param pos: The position of the MS.
        @type filter_scatters: Boolean.
        @param filter_scatters: If the scatters must be filtered (reduced) or not according to its power .        
        """ 
        # self.scenario.set_los_positions(pos)
        # d2D= np.sqrt((self.scenario.BS_pos[0] - pos[0] )**2+(self.scenario.BS_pos[1] - pos[1])**2)
        # self.scenario.set_params(d2D,pos[2])
        ssp = SSPs3GPP(17) 
        ssp.n_scatters = self.scenario.n_scatters
        ssp.reduced_n_scatters = self.scenario.n_scatters
        """ The number of clusters is reduced after PDP calculation to filter the clusters with power less than a threshold  """ 
        ssp.tau,ssp.tau_LOS,ssp.tau_min = self.__gen_clusters_delays()
        self.__tau = ssp.tau
        """ Array of relative delays of each cluster """ 
        self.__tau_LOS = ssp.tau_LOS
        """ Array of relative delays (in the LOS ccondition) of each cluster """ 
        self.__tau_min = ssp.tau_min
        """ The minimum delay of the clusters """ 
        ssp.P,ssp.P_LOS = self.__gen_clusters_powers(self.filter_scatters)
        self.__P = ssp.P
        """ Array of powers recieved of each cluster . Power delay profile (PDP) """ 
        self.__P_LOS = ssp.P_LOS
        """ Array of powers recieved of each cluster in the LOS condition . Power delay profile (PDP) """ 
        ssp.phiAOA,ssp.phiAOD,ssp.thetaAOA,ssp.thetaAOD = self.__gen_cluster_angles(pos)
        self.__phiAOA = ssp.phiAOA
        """The array with the azimuth angle of arrival for each cluster.In degrees.""" 
        self.__phiAOD = ssp.phiAOD
        """The array with the azimuth angle of departure for each cluster.In degrees. """ 
        self.__thetaAOA = ssp.thetaAOA
        """The array with the inclination angle of arrival for each cluster.In degrees. """ 
        self.__thetaAOD = ssp.thetaAOA
        """The array with the inclination angle of departure for each cluster.In degrees. """ 
        ssp.phiAOA_m_rad,ssp.phiAOD_m_rad,ssp.thetaAOA_m_rad,ssp.thetaAOD_m_rad = self.__generate_rays_angles()
        self.__phiAOA_m_rad = ssp.phiAOA_m_rad 
        """ A matrix (cluster number x ray number) with the azimuth angle of arrival for each ray in each cluster. In radians. """ 
        self.__phiAOD_m_rad = ssp.phiAOD_m_rad
        """ A matrix (cluster number x ray number) with the azimuth angle of departure for each ray in each cluster. In radians. """ 
        self.__thetaAOA_m_rad = ssp.thetaAOA_m_rad
        """ A matrix (cluster number x ray number) with the inclination angle of arrival for each ray in each cluster. In radians. """ 
        self.__thetaAOD_m_rad = ssp.thetaAOD_m_rad
        """ A matrix (cluster number x ray number) with the inclination angle of departure for each ray in each cluster. In radians. """     
        self.__tau_prev = np.copy(self.__tau)
        """ Auxiliary variable to store the tau vector of the previous point in the path. """
        if self.__los:
            self.__tau = self.__tau_LOS
        ssp.xpol= self.__generate_xpolarization()
        self.__xpol = ssp.xpol 
        """ Cross polarization array """ 
        ssp.iniphases = self.__generate_initial_phases()
        self.__ini_phases = ssp.iniphases 
        """ Initial phases """ 
        return ssp
 
  def set_ssps(self,ssp):
    """ This method given a SSPs object, sets the ssp for this fading object.
    
    @type ssp: SSPs Clsss
    @param ssp: The SSPs object to assign to the ssps of this Fading object.
    """ 
      
    self.__tau = ssp.tau
    """ Array of relative delays of each cluster """ 
    self.__tau_LOS = ssp.tau_LOS
    """ Array of relative delays (in the LOS ccondition) of each cluster """ 
    self.__tau_min = ssp.tau_min
    """ The minimum delay of the clusters """ 
    self.__P = ssp.P
    """ Array of powers recieved of each cluster . Power delay profile (PDP) """ 
    self.__P_LOS = ssp.P_LOS
    """ Array of powers recieved of each cluster in the LOS condition . Power delay profile (PDP) """ 
    self.__phiAOA = ssp.phiAOA
    """The array with the azimuth angle of arrival for each cluster.In degrees.""" 
    self.__phiAOD = ssp.phiAOD
    """The array with the azimuth angle of departure for each cluster.In degrees. """ 
    self.__thetaAOA = ssp.thetaAOA
    """The array with the inclination angle of arrival for each cluster.In degrees. """ 
    self.__thetaAOD = ssp.thetaAOA
    """The array with the inclination angle of departure for each cluster.In degrees. """ 
    self.__phiAOA_m_rad = ssp.phiAOA_m_rad 
    """ A matrix (cluster number x ray number) with the azimuth angle of arrival for each ray in each cluster. In radians. """ 
    self.__phiAOD_m_rad = ssp.phiAOD_m_rad
    """ A matrix (cluster number x ray number) with the azimuth angle of departure for each ray in each cluster. In radians. """ 
    self.__thetaAOA_m_rad = ssp.thetaAOA_m_rad
    """ A matrix (cluster number x ray number) with the inclination angle of arrival for each ray in each cluster. In radians. """ 
    self.__thetaAOD_m_rad = ssp.thetaAOD_m_rad
    """ A matrix (cluster number x ray number) with the inclination angle of departure for each ray in each cluster. In radians. """     
    if self.__los:
        self.__tau = self.__tau_LOS
    self.__tau_prev = np.copy(self.__tau)
    """ Auxiliary variable to store the tau vector of the previous point in the path. """

    self.__xpol = ssp.xpol 
    """ Cross polarization array """ 
    self.__ini_phases = ssp.iniphases 
    """ Initial phases """ 
    self.reduced_n_scatters = ssp.reduced_n_scatters
 
  
  def update(self,mspos,msvel,MS_antenna,BS_antenna,t0,time,d2d,d3d):
    """ This method updates the ssp parameters and the channel matrix, for a points of the path with spatial consistency with the previous point of the path. 
    
    The method using the previous fading parameters updates the ssp parameters acording to procedure A in 7.6.3.2 of 33GPP TR 38.901 version 14.0.0 Release 14.
    @type mspos: 3D array or list.
    @param mspos: The position of the mobile device in the scenario.
    @type msvel: 3D array or list.
    @param msvel: the velocity vector of the mobile device in the scenario.
    @type MS_antenna: Class Antenna.
    @param MS_antenna: The MS antenna.
    @type BS_antenna: Class Antenna.
    @param BS_antenna: The BS antenna.  
    @type t0: float.
    @param t0: The previous time of the simulation in seconds.
    @type time: float.
    @param time: The cuurent time of the simulation in seconds.
    @type d2d: float.
    @param d2d: The 2d distance of the previous point in the path with the BS.
    @type d3d: float.
    @param d3d: The 3d distance of the previous point in the path with the BS.
    """ 
    self.MS_antenna = MS_antenna
    """ The MS antenna model."""
    self.BS_antenna = BS_antenna
    """The BS antenna model """ 
    self.__set_lsps(mspos)
    self.MS_pos = mspos
    self.MS_vel = msvel
    msvel = np.ones((self.reduced_n_scatters,3))*msvel
    if self.scatters_move:
        msvel = msvel - self.vel_scatters 
        msvel = msvel[0:self.reduced_n_scatters]
    self.t = time
    self.__tau = np.copy(self.__tau_prev[0:self.reduced_n_scatters])
    self.__phiAOA = self.__phiAOA[0:self.reduced_n_scatters]
    self.__phiAOD = self.__phiAOD[0:self.reduced_n_scatters]
    self.__thetaAOA = self.__thetaAOA[0:self.reduced_n_scatters]
    self.__thetaAOD = self.__thetaAOD[0:self.reduced_n_scatters]    
    self.scenario.n_scatters = len(self.__tau)
    self.__tau = self.__tau + self.__tau_min
    # I have a doubt of the sign (add or sustract to tau)
    self.__tau  = self.__tau + (time-t0)*(np.sin(self.__thetaAOA/180*np.pi) * np.cos(self.__phiAOA/180*np.pi) * msvel[:,0]+ 
                      np.sin(self.__thetaAOA/180*np.pi) * np.sin(self.__phiAOA/180*np.pi) * msvel[:,1]+
                      np.cos(self.__thetaAOA/180*np.pi) * msvel[:,2])/3e8  
    np.where(self.__tau<0, 0, self.__tau)
  
    self.__tau_min = min(self.__tau)
    self.__tau = self.__tau - min(self.__tau)

    # Equation 7.5.3 and 7.5.4   
    # if self.__los:
    #     K = self.__sigma_K
    #     D = 0.7705-0.0433*K+0.0002*K**2+0.000017*K**3
    #     self.__tau_LOS = np.copy(self.__tau/D)
    #self.__P,self.__P_LOS = self.gen_clusters_powers()
    self.__phiAOA = self.__phiAOA[0:self.reduced_n_scatters]
    self.__phiAOD = self.__phiAOD[0:self.reduced_n_scatters]
    self.__thetaAOA = self.__thetaAOA[0:self.reduced_n_scatters]
    self.__thetaAOD = self.__thetaAOD[0:self.reduced_n_scatters]

    v = np.linalg.norm(msvel,axis=1)
    phi_v = np.zeros(self.reduced_n_scatters)
    phi_prime_AOA = np.zeros(self.reduced_n_scatters)
    phi_prime_AOD = np.zeros(self.reduced_n_scatters)
    theta_prime_AOA = np.zeros(self.reduced_n_scatters)
    theta_prime_AOD = np.zeros(self.reduced_n_scatters)

    for i in range(self.reduced_n_scatters):
        if v[i] != 0:
            phi_v[i] = np.arcsin(np.divide(msvel[i,1],v[i]))
            phi_prime_AOA[i] = np.random.uniform(-np.pi,np.pi) #, size=len(self.__phiAOA))
            phi_prime_AOD[i] = np.random.uniform(-np.pi,np.pi)# size=len(self.__phiAOD))
            theta_prime_AOA[i] = np.random.uniform(-np.pi/2,np.pi/2)# size=len(self.__thetaAOA))
            theta_prime_AOD[i] = np.random.uniform(-np.pi/2,np.pi/2) # size=len(self.__thetaAOD))

    if self.__los:
        phi_prime_AOA[0] = 0
        phi_prime_AOD[0] = 0
        theta_prime_AOA[0] = 0
        theta_prime_AOD[0] = 0

    self.__phiAOA = self.__phiAOA-v*(time-t0)/d2d*np.sin(phi_v-self.__phiAOA+phi_prime_AOA)*180/np.pi
    self.__phiAOD = self.__phiAOD+v*(time-t0)/d2d*np.sin(phi_v-self.__phiAOD+phi_prime_AOD)*180/np.pi
    self.__thetaAOA = self.__thetaAOA-v*(time-t0)/d3d*np.sin(phi_v-self.__phiAOA+theta_prime_AOA)*180/np.pi
    self.__thetaAOD = self.__thetaAOD-v*(time-t0)/d3d*np.sin(phi_v-self.__phiAOD+theta_prime_AOD)*180/np.pi

    for i in range(self.reduced_n_scatters):
       while (self.__phiAOA[i] > 360):
         self.__phiAOA[i] = self.__phiAOA[i] - 360
       while (self.__phiAOD[i] > 360):
         self.__phiAOD[i] = self.__phiAOD[i] - 360
       while (self.__thetaAOA[i] > 360):
         self.__thetaAOA[i] = self.__thetaAOA[i] - 360
       while (self.__thetaAOD[i] > 360):
         self.__thetaAOD[i] = self.__thetaAOD[i] - 360
       while (self.__phiAOA[i] < 0):
         self.__phiAOA[i] = self.__phiAOA[i] + 360
       while (self.__phiAOD[i] < 0):
         self.__phiAOD[i] = self.__phiAOD[i] + 360
       while (self.__thetaAOA[i] < 0):
         self.__thetaAOA[i] = self.__thetaAOA[i] + 360
       while (self.__thetaAOD[i] < 0):
         self.__thetaAOD[i] =self.__thetaAOD[i] + 360
       if (self.__thetaAOD[i] > 180):
         self.__thetaAOD[i] = 360 -self.__thetaAOD[i] 
       if (self.__thetaAOA[i] > 180):
         self.__thetaAOA[i] = 360 -self.__thetaAOA[i] 

    phiA_ant = np.copy(self.__phiAOA_m_rad)
    phiD_ant = np.copy(self.__phiAOD_m_rad)
    thetaA_ant = np.copy(self.__thetaAOA_m_rad)
    thetaD_ant = np.copy(self.__thetaAOD_m_rad)
    
    self.__phiAOA_m_rad,self.__phiAOD_m_rad,self.__thetaAOA_m_rad,self.__thetaAOD_m_rad = self.__generate_rays_angles(shuffle=False)
    for i in range(self.reduced_n_scatters):
        if v[i] == 0:
            self.__phiAOA_m_rad[i] = phiA_ant[i]
            self.__phiAOD_m_rad[i] = phiD_ant[i]  
            self.__thetaAOA_m_rad[i] = thetaA_ant[i] 
            self.__thetaAOD_m_rad[i] = thetaD_ant[i] 
            
    self.__tau_prev = np.copy(self.__tau)
    self.__cluster1,self.__cluster2 = self.__find_strongest_clusters()
    self.H_usn = self.__generateChannelCoeff()
    

  def __gen_clusters_delays(self):
    """ This method computes the random cluster delays acording to  3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model,step 5.
    
    @return: the cluster delays and the cluster delays in LOS condition.
    """ 
    tau_tmp = np.zeros(self.scenario.n_scatters) 
    tau =np.zeros(self.scenario.n_scatters)
    tau_LOS =np.zeros(self.scenario.n_scatters)
    for i in range(self.scenario.n_scatters):
        x = np.random.uniform()
        #Equation 7.5.1
        tau_tmp_ele = -self.scenario.rTau*self.__sigma_tau*np.log(x)
        tau_tmp[i] = tau_tmp_ele
    # Equation 7.5.2
    tau_min = min(tau_tmp)
    tau = tau_tmp - min(tau_tmp)
    tau = np.sort(tau)
    # Equation 7.5.3 and 7.5.4   
    if self.__los:
      K = self.__sigma_K
      D = 0.7705-0.0433*K+0.0002*K**2+0.000017*K**3
      tau_LOS = tau/D
    # Equation 7.5.4 tau_LOS is not used for cluster power generation
    # After cluster power generation tau must be assigned with tau_LOS in the 
    # case of LOS condition.
    return tau,tau_LOS,tau_min

  def __gen_clusters_powers(self,filter_scatters=True):
    """ This method computes and returns the random cluster powers acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 6.
    
    @return: the cluster powers and the cluster powers in LOS condition.
    """ 
    P_tmp = np.zeros(self.scenario.n_scatters)
    for i in range(self.scenario.n_scatters):
      Z = np.random.normal(0,1)*self.scenario.perClusterShadowingStd
      v = 10**(-Z/10)
      gamma = (self.scenario.rTau - 1)/(self.scenario.rTau*self.__sigma_tau)
      exptau = np.exp(-self.__tau[i]*gamma)
      # Equation 7.5.5
      P_tmp[i] = exptau*v
    # Equation 7.5.6
    P = P_tmp / sum(P_tmp)
    #Case LOS condition Eq 7.5.7 and 7.5.8 . These
    # power values are used only in equations (7.5-9) and (7.5-14), but not in equation (7.5-22).
    P_LOS = np.zeros(self.scenario.n_scatters)    
    for i in range(self.scenario.n_scatters):
      if self.__los:
        KR = np.power(10,self.__sigma_K/10)
        if i == 0:
          P_LOS[i] = P[i] / (1 + KR) + KR / (1 + KR) 
        else:
          P_LOS[i] = P[i] / (1 + KR)
      else:
        P_LOS[i] = P[i]    
    powerMax = max(P_LOS)
    if filter_scatters:
        #Remove clusters with less than -25 dB power compared to the maximum cluster power. 
        #The scaling factors need not be changed after cluster elimination.
        thresh = np.power(10, -2.5)
        index = []
        for i in range(self.scenario.n_scatters):
          if (P_LOS[i] < thresh * powerMax ):
            index.append(i)
        P_LOS = np.delete(P_LOS,index)
        P = np.delete(P,index)
        self.__tau = np.delete(self.__tau,index)
        self.__tau_LOS =  np.delete(self.__tau_LOS,index)
    self.reduced_n_scatters = self.__tau.size     
    return P,P_LOS

  def __filter_scatters(self,ssp):
    """This method removes the scatters  with power less than -25 db of the power of the scatter with maximum power.
 
    @type ssp: SSPs Class.
    @param ssp: the SSPs object to filter.
    @return: The SSPs object filtered.
    """ 
    powerMax = max(ssp.P_LOS)
    #Remove clusters with less than -25 dB power compared to the maximum cluster power. 
    #The scaling factors need not be changed after cluster elimination.
    thresh = np.power(10, -2.5)
    index = []
    for i in range(ssp.n_scatters):
      if (ssp.P_LOS[i] < thresh * powerMax ):
        index.append(i)
    ssp.P_LOS = np.delete(ssp.P_LOS,index)
    ssp.P = np.delete(ssp.P,index)
    ssp.tau =  np.delete(ssp.tau,index)
    ssp.tau_LOS =  np.delete(ssp.tau_LOS,index)
    ssp.phiAOA = np.delete(ssp.phiAOA,index)
    ssp.phiAOD= np.delete(ssp.phiAOD,index)
    ssp.thetaAOA = np.delete(ssp.thetaAOA,index)
    ssp.thetaAOD = np.delete(ssp.thetaAOD,index)
    ssp.phiAOA_m_rad = np.delete(ssp.phiAOA_m_rad,index,axis=0)
    ssp.phiAOD_m_rad = np.delete(ssp.phiAOD_m_rad,index,axis=0)
    ssp.thetaAOA_m_rad = np.delete(ssp.thetaAOA_m_rad,index,axis=0)
    ssp.thetaAOD_m_rad = np.delete(ssp.thetaAOD_m_rad,index,axis=0)
    ssp.xpol = np.delete(ssp.xpol,index,axis=0)
    ssp.iniphases = np.delete(ssp.iniphases,index,axis=0)
    ssp.reduced_n_scatters = ssp.tau.size     
    return ssp

  def __gen_cluster_angles(self,pos):
    """ This method computes and returns for each cluster the arrival and departures angles (azimuth and zenith), acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 7.
    
    @return: phiAOA,phiAOD,thetaAOA,thetaAOD.
    """
    # Table 7.5.2 Scaling factors for AOA, AOD generation
    C_PHI_TABLE = {2:0.779,4:0.779, 5:0.860, 8:1.018, 10:1.090, 11:1.123,12:1.146, 14:1.190, 15:1.211, 16:1.226,19:1.273, 20:1.289}
    # Table 7.5-4: Scaling factors for ZOA, ZOD generation
    C_THETA_TABLE = {2:0.889,8:0.889,10:0.957,11:1.031,12:1.104,15:1.1088,19:1.184,20:1.178}
    if self.__los:
      # Modifies the scaling factors of tables 7.5.2 ans 7.5.4 for LOS condition
      K = self.__sigma_K
      C_PHI = C_PHI_TABLE[self.scenario.n_scatters]
      # Equation 7.5.10
      C_PHI = C_PHI*(1.1035 - 0.028*K-0.002*K**2+0.0001*K**3)
      C_THETA = C_THETA_TABLE[self.scenario.n_scatters]
      #Table 7.5-4: Scaling factors for ZOA, ZOD generation
      C_THETA = C_THETA*(1.3086 + 0.0339*K-0.0077*K**2+0.0002*K**3)
    else:
      C_PHI = C_PHI_TABLE[self.scenario.n_scatters]
      C_THETA = C_THETA_TABLE[self.scenario.n_scatters]
    
    phiAOA = np.zeros(self.reduced_n_scatters)
    phiAOD = np.zeros(self.reduced_n_scatters)
    thetaAOA = np.zeros(self.reduced_n_scatters)
    thetaAOD = np.zeros(self.reduced_n_scatters)
    for i in range(self.reduced_n_scatters):
      if self.__los:
          logtmp = -np.log(self.__P_LOS[i]/max(self.__P_LOS))
      else:
          logtmp = -np.log(self.__P[i]/max(self.__P))
      angletmp = 2 * np.sqrt(logtmp) / 1.4 / C_PHI
      # Equation 7.5.9
      phiAOA[i] = self.__sigma_AOA_AZS * angletmp
      phiAOD[i] = self.__sigma_AOD_AZS* angletmp
      angletmp = logtmp / C_THETA
      # Equation 7.5.14
      thetaAOA[i] = self.__sigma_AOA_ELS * angletmp
      thetaAOD[i] = self.__sigma_AOD_ELS * angletmp
    dAngle = ang.Angles(0,0)
    dAngle.get_angles_vectors(pos,self.scenario.BS_pos)
    aAngle = ang.Angles(0,0)
    aAngle.get_angles_vectors(self.scenario.BS_pos,pos)
    for i in range(self.reduced_n_scatters):
      # Equation 7.5.11
      X = np.random.choice([-1,1])
      phiAOA[i] = X*phiAOA[i]+np.random.normal(0,1)*self.__sigma_AOA_AZS/7+aAngle.get_azimuth_degrees()
      X = np.random.choice([-1,1])
      phiAOD[i] = X*phiAOD[i]+np.random.normal(0,1)*self.__sigma_AOD_AZS/7+dAngle.get_azimuth_degrees()
      # Equation 7.5.16
      X = np.random.choice([-1,1])
      if (self.scenario.O2I):
        thetaAOA[i] = X*thetaAOA[i]+np.random.normal(0,1)*self.__sigma_AOA_ELS/7+90
      else:
        thetaAOA[i] = X*thetaAOA[i]+np.random.normal(0,1)*self.__sigma_AOA_ELS/7+ aAngle.get_inclination_degrees()
      X = np.random.choice([-1,1])
      thetaAOD[i] = X*thetaAOD[i]+np.random.normal(0,1)*self.__sigma_AOD_ELS/7+ dAngle.get_inclination_degrees()+self.scenario.offsetZOD

    # Equations 7.5.12 and 7.5.17
    if self.__los:
      for i in range(self.reduced_n_scatters):
        phiAOA[i] = phiAOA[i] -  phiAOA[0] + aAngle.get_azimuth_degrees()
        phiAOD[i] = phiAOD[i] -  phiAOD[0] + dAngle.get_azimuth_degrees()
        thetaAOA[i] = thetaAOA[i] -  thetaAOA[0] + aAngle.get_inclination_degrees()
        thetaAOD[i] = thetaAOD[i] -  thetaAOD[0] + dAngle.get_inclination_degrees()
 
    for i in range(self.reduced_n_scatters):
      while (phiAOA[i] > 360):
        phiAOA[i] = phiAOA[i] - 360
      while (phiAOD[i] > 360):
        phiAOD[i] = phiAOD[i] - 360
      while (thetaAOA[i] > 360):
        thetaAOA[i] = thetaAOA[i] - 360
      while (thetaAOD[i] > 360):
        thetaAOD[i] = thetaAOD[i] - 360
      while (phiAOA[i] < 0):
        phiAOA[i] = phiAOA[i] + 360
      while (phiAOD[i] < 0):
        phiAOD[i] = phiAOD[i] + 360
      while (thetaAOA[i] < 0):
        thetaAOA[i] = thetaAOA[i] + 360
      while (thetaAOD[i] < 0):
        thetaAOD[i] = thetaAOD[i] + 360
      if (thetaAOD[i] > 180):
        thetaAOD[i] = 360 -thetaAOD[i] 
      if (thetaAOA[i] > 180):
        thetaAOA[i] = 360 -thetaAOA[i] 
    return phiAOA,phiAOD,thetaAOA,thetaAOD

  def __shuffle (self,vect):
    """ This is an auxiliary method that shuffles randomly the elements of an array.

    @type vect: An array or list.
    @param vect: The vector to shuffle.
    @return: the shuffled vector.
    """
    s = len(vect) - 1
    for  i in range(s,0, -1):
      aux = np.random.randint(0, i)
      (vect[i], vect[aux]) = (vect[aux],vect[i])
    return vect

  def __generate_rays_angles(self,shuffle = True):
    """This method generates and returns the arrival and departure angles of each ray inside each cluster acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 7, equations 7.5.13, 7.5.18 and 7.5.20.

    @return: phiAOA_m_rad,phiAOD_m_rad,thetaAOA_m_rad,thetaAOD_m_rad (number of clusters x number of rays)
    """
    # Table 7.5-3: Ray offset angles within a cluster, given for rms angle spread normalized to 1
    RayOffset_TABLE = [0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715, 0.5129, -0.5129,0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481, 1.5195, -1.5195, 2.1551, -2.1551]
    n_clus = self.reduced_n_scatters
    m_rays = self.scenario.raysPerCluster
    phiAOA_m_rad = np.zeros((n_clus,m_rays))
    phiAOD_m_rad = np.zeros((n_clus,m_rays))
    thetaAOA_m_rad = np.zeros((n_clus,m_rays))
    thetaAOD_m_rad = np.zeros((n_clus,m_rays))
    tmpAngle = ang.Angles(0, 0)
    for j in range(n_clus):
      for i in range(m_rays):
        phiAOA_m_rad[j][i] = np.deg2rad(self.__phiAOA[j] + self.scenario.cASA*RayOffset_TABLE[i])
        phiAOD_m_rad[j][i] = np.deg2rad(self.__phiAOD[j] + self.scenario.cASD*RayOffset_TABLE[i])
        thetaAOA_m_rad[j][i] = np.deg2rad(self.__thetaAOA[j] + self.scenario.cZSA*RayOffset_TABLE[i])
        thetaAOD_m_rad[j][i] = np.deg2rad(self.__thetaAOD[j] + 3/8 * np.power(10,self.scenario.muZSDLg)*RayOffset_TABLE[i])
        phiAOA_m_rad[j][i],thetaAOA_m_rad[j][i] = tmpAngle.wrap_angles3gpp(phiAOA_m_rad[j][i],thetaAOA_m_rad[j][i])
        phiAOD_m_rad[j][i],thetaAOD_m_rad[j][i] = tmpAngle.wrap_angles3gpp(phiAOD_m_rad[j][i],thetaAOD_m_rad[j][i])
    if shuffle:
        for j in range(n_clus):
          phiAOA_m_rad[j] = self.__shuffle(phiAOA_m_rad[j])
          phiAOD_m_rad[j] = self.__shuffle(phiAOD_m_rad[j])
          thetaAOA_m_rad[j] = self.__shuffle(thetaAOA_m_rad[j])
          thetaAOD_m_rad[j] = self.__shuffle(thetaAOD_m_rad[j])

    return phiAOA_m_rad,phiAOD_m_rad,thetaAOA_m_rad,thetaAOD_m_rad

  def __generate_xpolarization(self):
    """This method generates the cross polarization power ratios acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 9.

    @return: crossPolarizationPowerRatios (number of clusters x number of rays)
    """
    #Step 9: 
    n_clus = self.reduced_n_scatters
    m_rays = self.scenario.raysPerCluster
    crossPolarizationPowerRatios = np.zeros((n_clus,m_rays)) 
    # Equation 7.5.21
    for i in range(n_clus):
      for j in range(m_rays):
        muXprLinear = np.power(10, self.scenario.muXpr / 10)
        sigmaXprLinear = np.power(10, self.scenario.sigmaXpr / 10)
        crossPolarizationPowerRatios[i][j] = np.power(10, (np.random.normal(0,1) * sigmaXprLinear + muXprLinear) / 10)
    return crossPolarizationPowerRatios

  def __generate_initial_phases(self):
    """This method generates the initial phases acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 10.

    @return: The initial phases
    """
    n_clus = self.reduced_n_scatters
    m_rays = self.scenario.raysPerCluster
    clusterPhase = np.zeros((n_clus,m_rays,4))
    for i in range(n_clus):
      for j in range(m_rays):
        for p in range(4):
          clusterPhase[i][j][p] = np.random.uniform(-np.pi,np.pi)
    return clusterPhase

  def __find_strongest_clusters(self):  
    """Auxiliary method to find the two strongest clusters.

    @return: cluster1, cluster2. The index of the strogest clsuters.
    """ 
    n_clus = self.reduced_n_scatters
    cluster1 = 0
    cluster2 = 0 # for the first and second strongest cluster
    maxPower = 0
    for i in range(n_clus):
      if maxPower < self.__P[i]:
        maxPower = self.__P[i]
        cluster1 = i
    maxPower = 0
    for i in range(n_clus):
      if maxPower < self.__P[i] and i != cluster1:
        maxPower = self.__P[i]
        cluster2 = i 
    return cluster1,cluster2

  def __generateChannelCoeff(self):
    """This method generates channel coefficients for each cluster n and each receiver and transmitter element pair u,s, acording to 3GPP TR 38.901 version 14.0.0 Release 14, section 7.5 Fast fading model, step 11.
    
    @return: H_usn. The channel matrix for each pair of Rx-Tx antenna elements and for each cluster.
    """ 
    n_clus = self.reduced_n_scatters
    m_rays = self.scenario.raysPerCluster
    usize = self.MS_antenna.get_number_of_elements()
    ssize = self.BS_antenna.get_number_of_elements()
    self.__attenuation_dB = np.zeros(n_clus)
    for n in range(n_clus):
      if (self.scenario.blockage):
        print("blockage is not implemented yet")
        self.__attenuation_dB[n] = 0
      else:
        self.__attenuation_dB[n] = 0
    #NOTE Since each of the strongest 2 clusters are divided into 3 sub-clusters,
    #the total cluster will be reduced_n_scatters + 4.
    H_usn = np.zeros((usize,ssize,n_clus+4), dtype=complex) 
    # channel coefficients H_NLOS [u][s][n],where u and s are receive and transmit antenna element, n is cluster index.
    for u in range(usize):
      u_pos = self.MS_antenna.get_element_location(u)
      for s in range(ssize):
        s_pos = self.BS_antenna.get_element_location(s)
        for n in range(n_clus):
          if(n != self.__cluster1 and n != self.__cluster2):
            rays = complex(0,0)
            for m in range(m_rays):
              initial_phase = self.__ini_phases[n][m]
              k = self.__xpol[n][m]
              rays +=self.__compute_Husnm(self.__thetaAOA_m_rad[n][m],self.__thetaAOD_m_rad[n][m],self.__phiAOA_m_rad[n][m],self.__phiAOD_m_rad[n][m],u_pos,s_pos,k,initial_phase)
            rays *= np.sqrt(self.__P[n] / m_rays)
            H_usn[u][s][n] = rays
          else:  
            raysSub1 = complex(0,0)
            raysSub2 = complex(0,0)
            raysSub3 = complex(0,0)
            last = n_clus
            if n == self.__cluster1:
              last = n_clus
            if n == self.__cluster2:
              last = n_clus+2
            for m in range(m_rays):
              initial_phase = self.__ini_phases[n][m]
              k = self.__xpol[n][m] 
              if m in [9,10,11,12,17,18]:  
                raysSub2 += self.__compute_Husnm(self.__thetaAOA_m_rad[n][m],self.__thetaAOD_m_rad[n][m],self.__phiAOA_m_rad[n][m],self.__phiAOD_m_rad[n][m],u_pos,s_pos,k,initial_phase)
              elif(m in [13,14,15,16]):
                raysSub3 += self.__compute_Husnm(self.__thetaAOA_m_rad[n][m],self.__thetaAOD_m_rad[n][m],self.__phiAOA_m_rad[n][m],self.__phiAOD_m_rad[n][m],u_pos,s_pos,k,initial_phase)
              else:
                raysSub1 += self.__compute_Husnm(self.__thetaAOA_m_rad[n][m],self.__thetaAOD_m_rad[n][m],self.__phiAOA_m_rad[n][m],self.__phiAOD_m_rad[n][m],u_pos,s_pos,k,initial_phase)            
            raysSub1 *= np.sqrt(self.__P[n] / m_rays)
            raysSub2 *= np.sqrt(self.__P[n] / m_rays)
            raysSub3 *= np.sqrt(self.__P[n] / m_rays)
            H_usn[u][s][n] = raysSub1
            H_usn[u][s][last] = raysSub2
            H_usn[u][s][last+1] = raysSub3
        if (self.__los): #(7.5-29) and (7.5-30)
          dAngle = ang.Angles(0,0)
          dAngle.get_angles_vectors(self.MS_pos,self.scenario.BS_pos)
          aAngle = ang.Angles(0,0)
          aAngle.get_angles_vectors(self.scenario.BS_pos,self.MS_pos)
          ray = complex(0,0)
          ray = self.__compute_Husnm(aAngle.get_inclination(),dAngle.get_inclination(),aAngle.get_azimuth(),dAngle.get_azimuth(),u_pos,s_pos,0,[0,0,0,np.pi])
          lambda0 = 3e8 / self.scenario.fcGHz/1e9 # the wavelength of the carrier frequency
          d3D= np.sqrt((self.MS_pos[0] - self.scenario.BS_pos[0] )**2+(self.MS_pos[1] - self.scenario.BS_pos[1] )**2+(self.MS_pos[2] - self.scenario.BS_pos[2] )**2)
          ray = ray * np.exp(complex(0, -2 * np.pi * d3D / lambda0))
          K_linear = np.power(10,self.__sigma_K / 10);
          #the LOS path should be attenuated if blockage is enabled.
          H_usn[u][s][0] = np.sqrt(1 / (K_linear + 1)) * H_usn[u][s][0] + np.sqrt(K_linear / (1 + K_linear)) * ray / np.power(10,self.__attenuation_dB[0] / 10)           
          #Equation (7.5-30) 
          for n in range(1,n_clus+4):
            H_usn[u][s][n] *= np.sqrt (1 / (K_linear + 1)) 
    if (self.__cluster1 < self.__cluster2):
      min = self.__cluster1
      max = self.__cluster2
    else:
      min = self.__cluster2
      max = self.__cluster1

    self.__tau = np.append(self.__tau,self.__tau[min] + 1.28 * self.scenario.cDS)  
    self.__tau = np.append(self.__tau,self.__tau[min] + 2.56 * self.scenario.cDS)  
    self.__tau = np.append(self.__tau,self.__tau[max] + 1.28 * self.scenario.cDS)  
    self.__tau = np.append(self.__tau,self.__tau[max] + 2.56 * self.scenario.cDS) 

    self.__phiAOA = np.append(self.__phiAOA,self.__phiAOA[min])
    self.__phiAOA = np.append(self.__phiAOA,self.__phiAOA[min])  
    self.__phiAOA = np.append(self.__phiAOA,self.__phiAOA[max])  
    self.__phiAOA = np.append(self.__phiAOA,self.__phiAOA[max])  

    self.__thetaAOA = np.append(self.__thetaAOA,self.__thetaAOA[min])
    self.__thetaAOA = np.append(self.__thetaAOA,self.__thetaAOA[min])  
    self.__thetaAOA = np.append(self.__thetaAOA,self.__thetaAOA[max])  
    self.__thetaAOA = np.append(self.__thetaAOA,self.__thetaAOA[max])  
    
    self.__phiAOD = np.append(self.__phiAOD,self.__phiAOD[min])
    self.__phiAOD = np.append(self.__phiAOD,self.__phiAOD[min])  
    self.__phiAOD = np.append(self.__phiAOD,self.__phiAOD[max])  
    self.__phiAOD = np.append(self.__phiAOD,self.__phiAOD[max])  

    self.__thetaAOD = np.append(self.__thetaAOD,self.__thetaAOD[min])
    self.__thetaAOD = np.append(self.__thetaAOD,self.__thetaAOD[min])  
    self.__thetaAOD = np.append(self.__thetaAOD,self.__thetaAOD[max])  
    self.__thetaAOD = np.append(self.__thetaAOD,self.__thetaAOD[max])  
    return H_usn

  def __doppler(self,ZOA_rad,AOA_rad):
    """ This method computes doppler for time t assuming the BS is fixed and 
    only the MS is moving.
    
    @type ZOA_rad: float.
    @param ZOA_rad: the inclination angle of arrival to the MS.
    @type AOA_rad: float.
    @param AOA_rad: the azimuth angle of arrival to the MS. 
    @return: doppler: exp(j*2*pi*rx_n_m.MS_vel*t)
    """
    aux  = 2 * np.pi * (np.sin(ZOA_rad) * np.cos(AOA_rad) * self.MS_vel[0]+ 
                         np.sin(ZOA_rad) * np.sin(AOA_rad) * self.MS_vel[1]+ 
                         np.cos(ZOA_rad) * self.MS_vel[2])      
    doppler = np.exp(complex(0,aux* self.t * self.scenario.fcGHz*1e9/3e8))
    return doppler  

  def __compute_Husnm(self,ZOA,ZOD,AOA,AOD,u_pos,s_pos,k,initialPhase):
    """ This method computes for each rx antena u, tx antena s, cluster n and ray m the channel matrix acording to 7.5.22, 7.5.28, 7.5.29

    @type ZOA: float.
    @param ZOA: the inclination angle of arrival to the MS from the cluster m and ray n.
    @type ZOD: float.
    @param ZOD: the inclination angle of departure from the BS to the cluster m and ray n.
    @type AOA: float.
    @param AOA: the azimuth angle of arrival to the MS from the cluster m and ray n. 
    @type AOD: float.
    @param AOD: the azimuth angle of departure from the BS to the cluster m and ray n. 
    @type u_pos: 3d array.
    @param u_pos: the psoition of the rx device (MS). 
    @type s_pos: 3d array.
    @param s_pos: the psoition of the tx device (BS). 
    @type k: float.
    @param k: Riccean k. 
    @type initialPhase: 4d array.
    @param initialPhase: Initial phases for four different polarisation (theta theta ,theta phi,phi theta ,phi phi). 
    @return: the channel matrix for a antnna pair, cluster and ray. 
  
    """


    rx_ph = 2 * np.pi * (np.sin (ZOA) * np.cos(AOA) * u_pos[0]
                               + np.sin(ZOA) * np.sin(AOA) * u_pos[1] 
                               + np.cos(ZOA) * u_pos[2])
    tx_ph = 2 * np.pi * (np.sin(ZOD) * np.cos(AOD) * s_pos[0]
                               + np.sin(ZOD) * np.sin(AOD) * s_pos[1] 
                               + np.cos(ZOD) * s_pos[2])

    rx_field_phi, rx_field_theta, = self.MS_antenna.get_element_field_pattern(ang.Angles(AOA, ZOA),0)
    tx_field_phi, tx_field_theta = self.BS_antenna.get_element_field_pattern(ang.Angles(AOD,ZOD),0)
    H_usnm = np.exp(complex(0, initialPhase[0])) * rx_field_theta * tx_field_theta 
    if k!=0:
        H_usnm += np.exp(complex(0, initialPhase[1])) * np.sqrt (1 / k) * rx_field_theta * tx_field_phi 
        H_usnm += np.exp (complex(0, initialPhase[2])) * np.sqrt (1 / k) * rx_field_phi * tx_field_theta 
    H_usnm += np.exp (complex(0, initialPhase[3])) * rx_field_phi * tx_field_phi
    H_usnm = H_usnm * np.exp(complex(0, rx_ph))* np.exp(complex(0, tx_ph))*self.__doppler(ZOA,AOA)

    return H_usnm

  def save(self,path,point_number,MS_num):
    """This method save all ssp parameters and the channel matrix of one point in the path of one MS.

    @type path: string.
    @param path: The directory to save the ssp parameters.
    @type point_number: int.
    @param point_number: The number of this point in the path. 
    @type MS_num: int.
    @param MS_num: The number of the MS. 

    """
    try:
        lsp = np.array([self.__los,self.__sigma_shadow,self.__sigma_K,self.__sigma_tau,self.__sigma_AOD_AZS,self.__sigma_AOA_AZS,self.__sigma_AOD_ELS,self.__sigma_AOA_ELS ])  
        np.savetxt(path+'/lsp_'+str(MS_num)+ '_' +str(point_number)+'.csv', lsp, delimiter=',')
        np.savetxt(path+'/tau_'+str(MS_num)+ '_' +str(point_number)+'.csv', self.__tau, delimiter=',')
        np.savetxt(path+'/tauLOS_'+ str(MS_num)+ '_' +str(point_number)+'.csv', self.__tau_LOS, delimiter=',')
        np.savetxt(path+'/PDP_'+str(MS_num)+ '_' +str(point_number)+'.csv', self.__P, delimiter=',')
        np.savetxt(path+'/PDP_LOS_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__P_LOS, delimiter=',')
        np.savetxt(path+'/PHI_AOA_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__phiAOA, delimiter=',')
        np.savetxt(path+'/PHI_AOD_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__phiAOD, delimiter=',')
        np.savetxt(path+'/THETA_AOA_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__thetaAOA, delimiter=',')
        np.savetxt(path+'/THETA_AOD_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__thetaAOD, delimiter=',')
        np.savetxt(path+'/PHI_AOA_rays_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__phiAOA_m_rad, delimiter=',')
        np.savetxt(path+'/PHI_AOD_rays_'+ str(MS_num)+ '_' +str(point_number)+ '.csv', self.__phiAOD_m_rad, delimiter=',')
        np.savetxt(path+'/THETA_AOA_rays_'+str(MS_num)+ '_' +str(point_number)+ '.csv', self.__thetaAOA_m_rad, delimiter=',')
        np.savetxt(path+'/THETA_AOD_rays_' +str(MS_num)+ '_' +str(point_number)+ '.csv', self.__thetaAOD_m_rad, delimiter=',')
        np.savetxt(path+'/Xpol_'+str(MS_num)+ '_' +str(point_number)+'.csv', self.__xpol, delimiter=',')
        #np.savetxt('./data/ini_phases_'+str(i)+'.csv', fading.ini_phases, delimiter=',')
        with open(path+'/H_usn_'+str(MS_num)+ '_' +str(point_number)+'.npy', 'wb') as f:
            np.save(f, self.H_usn)
    except OSError as error: 
        print(error)  
 
class SSPs:
  """This class implments the Short Scale Parameterss object.
  
  This class enables the access to all ssps in a single object. The ssps are
  stored in an array, and can be acceded in this way but also they can be access by setter and getter methos 
  like properties.
  """  
  def __init__(self,number_sps):
    """
    The constructor method of the  SSPs Class.
    
    @type n_scatters: int.
    @param n_scatters: The number of the scatters in this channel model.
    """ 
    self.number_sps = number_sps
    """ The number of the short scale parameters. """ 
    self.ssp_array = np.empty(shape=(self.number_sps),dtype=object)
    """ An array of ssps. Each element of the array is an array of each ssp.""" 
class SSPsRayleigh(SSPs):
  """This class implments the Short Scale Parameterss object for 3Rayleigh fading model.
  
  This class enables the access to all ssps in a single object. The ssps are
  stored in an array, and can be acceded in this way but also they can be access by setter and getter methos 
  like properties.
  """  
  def __init__(self,number_sps,number_sin):
    """
    The constructor method of the  SSPs Class.
    
    @type n_scatters: int.
    @param n_scatters: The number of the scatters in this channel model.
    """ 
    super().__init__(number_sps)
    self.ssp_array[0]= number_sin
    self.set_angles()
   
  def set_angles(self):
    alpha = np.zeros(self.ssp_array[0])
    phi = np.zeros(self.ssp_array[0])
    for i in range(self.ssp_array[0]):
        alpha[i] = (np.random.rand() - 0.5) * 2 * np.pi
        phi[i] = (np.random.rand() - 0.5) * 2 * np.pi
    self.ssp_array[1] = alpha
    self.ssp_array[2] = phi
   
  @property  
  def n_sin(self):
    """ Gets the number of simusoids """ 
    return int(self.ssp_array[0])

  @n_sin.setter
  def n_sin(self,value):
    """ Sets the number of simusoids""" 
    self.ssp_array[0] = int(value)
  
  @property  
  def alpha(self):
    """ Gets the alpha angles """ 
    return self.ssp_array[1]

  @alpha.setter
  def alpha(self,value):
    """ Sets the alpha angles""" 
    self.ssp_array[1] = value
    
  @property  
  def phi(self):
    """ Gets the alpha angles """ 
    return self.ssp_array[2]

  @phi.setter
  def phi(self,value):
    """ Sets the alpha angles""" 
    self.ssp_array[2] = value
 
      


class SSPs3GPP(SSPs):
  """This class implments the Short Scale Parameterss object for 3GPP.
  
  This class enables the access to all ssps in a single object. The ssps are
  stored in an array, and can be acceded in this way but also they can be access by setter and getter methos 
  like properties.
  """  
  def __init__(self,number_sps):
    """
    The constructor method of the  SSPs Class.
    
    @type n_scatters: int.
    @param n_scatters: The number of the scatters in this channel model.
    """ 
    super().__init__(number_sps)

 
  @property  
  def n_scatters(self):
    """ Gets the array of relative delays of each cluster """ 
    return int(self.ssp_array[15])

  @property  
  def reduced_n_scatters(self):
    """ Gets the array of relative delays of each cluster """ 
    return int(self.ssp_array[16])


  @property  
  def tau(self):
    """ Gets the array of relative delays of each cluster """ 
    return self.ssp_array[0]

  @property  
  def tau_LOS(self):
    """ Gets the array of relative delays (in LOS condition) of each cluster """ 
    return self.ssp_array[1]

  @property  
  def P(self):
    """ Gets the array of powers recieved of each cluster . Power delay profile (PDP) """ 
    return self.ssp_array[2]

  @property  
  def P_LOS(self):
      """ Gets the array of powers recieved of each cluster in the LOS condition . Power delay profile (PDP) """ 
      return self.ssp_array[3]

  @property  
  def phiAOA(self):
    """Gets the array with the azimuth angle of arrival for each cluster.In degrees.""" 
    return self.ssp_array[4]

  @property  
  def phiAOD(self):
    """Gets the array with the azimuth angle of departure for each cluster.In degrees. """ 
    return self.ssp_array[5]

  @property  
  def thetaAOA(self):
    """Gets the array with the inclination angle of arrival for each cluster.In degrees. """ 
    return self.ssp_array[6]

  @property  
  def thetaAOD(self):
    """Gets the array with the inclination angle of departure for each cluster.In degrees. """ 
    return self.ssp_array[7]

  @property  
  def phiAOA_m_rad(self):
    """ Gets the matrix (cluster number x ray number) with the azimuth angle of arrival for each ray in each cluster. In radians. """ 
    return self.ssp_array[8]

  @property  
  def phiAOD_m_rad(self):
    """ Gets the matrix (cluster number x ray number) with the azimuth angle of departure for each ray in each cluster. In radians. """ 
    return self.ssp_array[9]

  @property  
  def thetaAOA_m_rad(self):
    """ Gets the matrix (cluster number x ray number) with the inclination angle of arrival for each ray in each cluster. In radians. """ 
    return self.ssp_array[10]

  @property  
  def thetaAOD_m_rad(self):
    """ Gets the matrix (cluster number x ray number) with the inclination angle of departure for each ray in each cluster. In radians. """     
    return self.ssp_array[11]

  @property  
  def xpol(self):
    """ Gets the cross polarization array """ 
    return self.ssp_array[12]
  
  @property  
  def iniphases(self):
    """ Gets the intial phases """ 
    return self.ssp_array[13]

  @property  
  def tau_min(self):
    """ Gets the minimum delay of all clusters """ 
    return self.ssp_array[14]

  @tau.setter
  def tau(self,value):
    """ Sets the array of relative delays of each cluster """ 
    self.ssp_array[0] = value

  @tau_LOS.setter
  def tau_LOS(self,value):
    """ Sets the array of relative delays (in the LOS condition) of each cluster """ 
    self.ssp_array[1] = value

  @P.setter
  def P(self,value):
    """ Sets the array of powers recieved of each cluster . Power delay profile (PDP) """ 
    self.ssp_array[2] = value

  @P_LOS.setter
  def P_LOS(self,value):
    """ Sets the array of powers recieved of each cluster in the LOS condition . Power delay profile (PDP) """ 
    self.ssp_array[3] = value
    
  @phiAOA.setter
  def phiAOA(self,value):
    """Sets the array with the azimuth angle of arrival for each cluster.In degrees.""" 
    self.ssp_array[4] = value

  @phiAOD.setter
  def phiAOD(self,value):
    """Sets the array with the azimuth angle of departure for each cluster.In degrees. """ 
    self.ssp_array[5] = value

  @thetaAOA.setter
  def thetaAOA(self,value):
    """Sets the array with the inclination angle of arrival for each cluster.In degrees. """ 
    self.ssp_array[6] = value

  @thetaAOD.setter
  def thetaAOD(self,value):
    """Sets the array with the inclination angle of departure for each cluster.In degrees. """ 
    self.ssp_array[7] = value

  @phiAOA_m_rad.setter
  def phiAOA_m_rad(self,value):
    """ Sets the matrix (cluster number x ray number) with the azimuth angle of arrival for each ray in each cluster. In radians. """ 
    self.ssp_array[8] = value

  @phiAOD_m_rad.setter
  def phiAOD_m_rad(self,value):
    """ Sets the matrix (cluster number x ray number) with the azimuth angle of departure for each ray in each cluster. In radians. """ 
    self.ssp_array[9] = value

  @thetaAOA_m_rad.setter
  def thetaAOA_m_rad(self,value):
    """Sets the array with the inclination angle of arrival for each cluster.In degrees. """ 
    self.ssp_array[10] = value
  
  @thetaAOD_m_rad.setter
  def thetaAOD_m_rad(self,value):
    """ Sets the matrix (cluster number x ray number) with the inclination angle of departure for each ray in each cluster. In radians. """     
    self.ssp_array[11] = value

  @xpol.setter
  def xpol(self,value):
    """ Sets the cross polarization array """ 
    self.ssp_array[12] = value

  @iniphases.setter
  def iniphases(self,value):
    """ Sets the initial phases array """ 
    self.ssp_array[13] = value
    
  @tau_min.setter
  def tau_min(self,value):
    """ Sets the minimum delay of all clusters """ 
    self.ssp_array[14] = value

  @n_scatters.setter
  def n_scatters(self,value):
    """ Sets the minimum delay of all clusters """ 
    self.ssp_array[15] = int(value)
    
  @reduced_n_scatters.setter
  def reduced_n_scatters(self,value):
    """ Sets the minimum delay of all clusters """ 
    self.ssp_array[16] = int(value)
