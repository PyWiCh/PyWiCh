#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements different simulation scenarios. The scenarios specifies different
parameters and the calculus of pathloss and shadowing.

@author: pablo belzarena
"""



import numpy as np
from scipy.linalg import cholesky
import scipy.spatial
import scipy.stats
import scipy.signal as signal

class Scenario:
  """ This class is the abstract class Scenario from where inherit all scenarios.   
  
      It implement only the save method an a shadowing model.
  """
  def __init__(self,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,sigma_shadow=2,shadow_corr_distance=10):  
    """
    The constructor of the abstract class Scenario.
    
    @type fcGHz: float .
    @param fcGHz: Frequency in GHz of the carrier frequency of the scenario.
    @type posx_min: float .
    @param posx_min: The minimum limit of the x coordinate in the scenario. 
    @type posx_max: float .
    @param posx_max: The maximum limit of the x coordinate in the scenario. 
    @type posy_min: float .
    @param posy_min: The minimum limit of the y coordinate in the scenario. 
    @type posy_max: float .
    @param posy_max: The maximum limit of the y coordinate in the scenario. 
    @type grid_number: int .
    @param grid_number: For calculating the spacial distribution of the parameters of the scenario, 
    the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate. 
    @type bspos: 3d array or list .
    @param bspos: The position of the Base Satation in the scenario in the coordinates system [x,y,z].
    @type Ptx_db: float.
    @param Ptx_db: The power transmited by the base station in dbm. 
    @type sigma_shadow: float.
    @param sigma_shadow: The variance of the shadow gaussian model.
    @type shadow_corr_distance: float.
    @param shadow_corr_distance: The shadow correlation distance.
    
   
    """ 
    self.fcGHz = fcGHz
    """ Frequency in GHz of the carrier frequency of the scenario. """
    self._LOS = True
    """ Line of sight in the scenario. Boolean. """    
    self.posx_max = posx_max
    """ The maximum limit of the x coordinate in the scenario. """ 
    self.posx_min = posx_min
    """ The minimum limit of the x coordinate in the scenario. """ 
    self.posy_min = posy_min
    """ The minimum limit of the y coordinate in the scenario. """ 
    self.posy_max = posy_max
    """ The maximum limit of the y coordinate in the scenario. """ 
    self.grid_number = int(grid_number)
    """ For calculating the spacial distribution of the parameters of the scenario, 
    the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate.""" 
    self.BS_pos =bspos
    """ The position of the Base Satation in the scenario in the coordinates system [x,y,z]. """ 
    self.Ptx_db = Ptx_db
    """ The power transmited by the base station in dbm. """ 
    self.sigma_shadow = sigma_shadow
    """ The variance of the shadow gaussian model """ 
    self.shadow_pre = 0
    """ The previous value of the shadow. Used to filter and impose correlation """ 
    self.shadow_corr_distance = shadow_corr_distance
    """ The shadow correlation distance """ 
    self.pos = [100,100,0]
    """ The previous position of the mobile""" 
    self.shadow_enabled = True
    """ If shadow is enabled or not """
    self.X = np.array([])
    """ The x grid "of the scenario """ 
    self.Y = np.array([])
    """ The y grid "of the scenario """ 
    self.XY = np.array([])
    """ np.array([self.X, self.Y]))  """ 
    stepx = (self.posx_max-self.posx_min)/self.grid_number
    stepy = (self.posy_max-self.posy_min)/self.grid_number

    x = np.linspace(self.posx_min,self.posx_max+stepx,self.grid_number+1) # 2*self.grid_number)*(self.posx_max-self.posx_min)/(2*self.grid_number-1)+self.posx_min
    y = np.linspace(self.posy_min,self.posy_max+stepy,self.grid_number+1) #np.arange(0, 2*self.grid_number)*(self.posy_max-self.posy_min)/(2*self.grid_number-1)+self.posy_min
    self.X, self.Y = np.meshgrid(x, y) 
    self._paranum = 1
    """ Number of LSP parameters """ 
    self.gridLSP_LOS,self.XY = self.__generate_correlated_LSP()


  def __generate_correlated_LSP(self):
    """This method first generates for each LSP parameter an independent gaussian N(0,1) random variable for each point in the scenario grid. Later, 
    using cholesky method and the correlation matrix between the LSP parameters, generates a set of correlated LSP params.
    At last, the method applies to each parameter its expected value and its variance.

    """ 
    gridLSP = np.zeros((1,self.grid_number+1,self.grid_number+1))   
    gridLSP[0],XY = self.generate_LSPgrid(self.shadow_corr_distance)    
    return gridLSP, XY


 
    
  def generate_LSPgrid(self,corr_distance):
    """This method generates a spatial correlated gaussian random variables using the correlation distance.

    The covariance matrix is defined by cov = exp(-distance/correlation_distance)
    @type corr_distance: float.
    @param corr_distance: The correlation distance for the spatial correlated gaussian random variable.
    @return: A 2D matrix where the values of the matrix are spatially correlated gaussian random variables.
    """ 
    # Create a vector of cells
    XY = np.column_stack((np.ndarray.flatten(self.X),np.ndarray.flatten(self.Y)))
    # Calculate a matrix of distances between the cells
    dist = scipy.spatial.distance.pdist(XY)
    dist = scipy.spatial.distance.squareform(dist)    
    # Convert the distance matrix into a covariance matrix
    cov = np.exp(-dist/(corr_distance)) 
    noise = scipy.stats.multivariate_normal.rvs(mean = np.zeros((self.grid_number+1)**2),cov = cov)
    return(noise.reshape((self.grid_number+1,self.grid_number+1)),np.array([self.X, self.Y]))
 

  def save(self,path):
    """This method save to disk the configuration of the scenario. 
    
    @type path: string.
    @param path: The directory path where the data is saved.
    """ 
    np.savetxt(path+'/BS_position.csv', self.BS_pos, delimiter=',')      
    params = np.array([self.fcGHz,self.posx_min,self.posx_max,self.posy_min,self.posy_max,self.grid_number,self.Ptx_db])  
    np.savetxt(path+'/scenario.csv', params, delimiter=',')
      
 
  def get_loss_los (self,distance,h_MS):   
    """ The default method the get the path loss in the LOS condition of the scenario. By default 0. 
    
    @type distance: float.
    @param distance: The distance between the BS and MS positions.
    @type h_MS: float.
    @param h_MS: The MS antenna height.
    @return: 0
    """  
    return 0

  def get_loss_nlos (self,distance,h_MS):
    """ The default method the get the path loss in the Non LOS condition of the scenario. By default 0. 
    
    @type distance: float.
    @param distance: The distance between the BS and MS positions.
    @type h_MS: float.
    @param h_MS: The MS antenna height.
    @return: 0
    """ 
    return 0

  def is_los_cond(self, MS_pos): 
    """The default method to calculate if the scenario is in LOS condition or not. Default True. 
    
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the movil device in the scenario.
    @return: True
    """   
    return True
        
  def _prob_los(self, distance,h_MS):
    """The default method to calculate the probability function that defines if the scenario is in LOS condition or not. Default 1. 
    
    @type distance: float.
    @param distance: The distance between the BS and MS positions.
    @type h_MS: float.
    @param h_MS: The MS antenna height.
    @return: 1
    """   
    return 1
  def generate_correlated_LSP_vector(self,MS_pos,type_approx):
    """ This method given the LSP parameters grid and a MS position, estimates the LSP parameters for this point.

    The method can use two approximation methods. The first one use the parameter of the closest point in the grid. 
    The second on interpolates between the two closests point in the grid.
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the mobile device in the scenario.
    @type type_approx: int.
    @param type_approx: The type of approximation. 0 for the closest point. 1 for interpolation.
    @return: the LSP parameters for the MS_pos point in the scenario.
    """ 

    gridLSP = np.copy(self.gridLSP_LOS)
    gridLSP[0] = gridLSP[0]* self.sigma_shadow
    return self.LSP_vector_position(gridLSP,MS_pos,type_approx)

  def LSP_vector_position(self,gridLSP,MS_pos,type_approx):
    LSP_xy = np.zeros((self._paranum))
    absolute_val_array = np.abs(self.XY[0][0]- MS_pos[0])
    smallest_difference_index_x = absolute_val_array.argmin()
    closest_element_x = self.XY[0][0][smallest_difference_index_x]
    
    absolute_val_array = np.abs(self.XY[1][:,0]- MS_pos[1])
    smallest_difference_index_y = absolute_val_array.argmin()
    closest_element_y = self.XY[1][smallest_difference_index_y,0]
    if type_approx == 0:
        LSP_xy[:] = gridLSP[:,smallest_difference_index_y,smallest_difference_index_x] 
    else:
        if smallest_difference_index_x < self.XY[0][0].size-1 and smallest_difference_index_y < self.XY[1][:,0].size-1:    
            distance = np.sqrt((closest_element_x - MS_pos[0])**2 + (closest_element_y - MS_pos[1])**2 )
            d_step = np.sqrt((closest_element_x -self.XY[0][0][smallest_difference_index_x]+1)**2 + (closest_element_y - self.XY[1][smallest_difference_index_y+1,0])**2) 
            LSP_xy[:] = (d_step -distance)/d_step*gridLSP[:,smallest_difference_index_y,smallest_difference_index_x] +  distance/d_step*gridLSP[:,smallest_difference_index_y+1,smallest_difference_index_x+1]
        else:
            LSP_xy[:] = gridLSP[:,smallest_difference_index_y,smallest_difference_index_x] 
    return LSP_xy

  def get_shadowing_db(self,MS_pos,type_approx):
    """ This method computes the shadowing value for the MS position, sets its values and return it.
    
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the movil device in the scenario.
    @type type_approx: int.
    @param type_approx: The type of approximation used. 0 for the closest point in the grid. 1 for interpolating between closets points in the grid.
    @return: The shadowing valu for the MS position.
    """ 
    
    LSP = self.generate_correlated_LSP_vector(MS_pos,type_approx)
    self.shadow = LSP[0]
    return self.shadow

    d= np.sqrt((self.pos[0] - MS_pos[0] )**2+(self.pos[1] - MS_pos[1])**2)
    self.pos = [MS_pos[0],MS_pos[1],MS_pos[2]]
    a = np.exp(-d/self.shadow_corr_distance )
    b = self.sigma_shadow*np.sqrt(1-a**2)
    sample = np.random.normal(0,1)
    shadow = sample + self.shadow_pre * a
    self.shadow_pre = shadow
    if (b != 0):
        shadow = shadow * b
    return shadow



   

class ScenarioSimpleLossModel(Scenario):
  """This class implements an extended Friis loss model scenario
  where the loss is not cuadratic with the distance but depends as distance**order  
  """  
  def __init__(self,fcGHz,posx_max,posx_min,posy_max,posy_min,grid_number,bspos,Ptx_db,order,sigma_shadow=2,shadow_corr_distance=10): 
    """ The constructor of the extended Friis loss model scenario. Calls the parent class constructor.  
    
    @type fcGHz: float .
    @param fcGHz: Frequency in GHz of the carrier frequency of the scenario.
    @type posx_min: float .
    @param posx_min: The minimum limit of the x coordinate in the scenario. 
    @type posx_max: float .
    @param posx_max: The maximum limit of the x coordinate in the scenario. 
    @type posy_min: float .
    @param posy_min: The minimum limit of the y coordinate in the scenario. 
    @type posy_max: float .
    @param posy_max: The maximum limit of the y coordinate in the scenario. 
    @type grid_number: int .
    @param grid_number: For calculating the spacial distribution of the parameters of the scenario, 
    the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate. 
    @type bspos: 3d array or list .
    @param bspos: The position of the Base Satation in the scenario in the coordinates system [x,y,z].
    @type Ptx_db: float.
    @param Ptx_db: The power transmited by the base station in db. 
    @type order: float.
    @param order: The order of the exponent of the distance in the loss model.
    """ 
    super().__init__(fcGHz,posx_max,posx_min,posy_max,posy_min,grid_number,bspos,Ptx_db,sigma_shadow,shadow_corr_distance)
    self._order = order
    """ The order of the exponent of the distance in the loss model. """ 
    self.loss = 0
    """ The path loss """ 
   
  def __set_params(self,d2d=1,h_MS=1):
    """ This  method sets the params of the scenario. Not used in this scenario
    
    @type d2d: float.
    @param d2d: The distance between the BS and MS positions. Default 1.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 1.
    """
    pass

  def get_loss_los (self,distance,h_MS=1): 
    """ This method computes the path loss of the scenario using the Friis equation but with
    the distance**order 

    @type distance: float.
    @param distance: The distance between the BS and MS positions.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default value 1.
    @return: -20*np.log10(3e8/4/np.pi/self.fcGHz/1e9) +10*np.log10( (distance)**self._brder)
    """ 
    self.loss = 0
    if distance > 0:
        self.loss = -20*np.log10(3e8/4/np.pi/self.fcGHz/1e9) +10*np.log10( (distance)**self._order)
    return self.loss
  
  def get_loss_nlos (self,distance,h_MS=1):
    """ This method computes the path loss of the scenario using the Friis equation. The Friis
    model asumes LOS condition so the loss y the NLOS condition calls the  get_loss_los method.

    @type distance: float.
    @param distance: The distance between the BS and MS positions.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default value 1.
    @return: get_loss_los(distance,h_MS)
    """    
    return self.get_loss_los(distance,h_MS)



#3GPP TR 38.901 version 14.0.0 Release 14
#https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf

class Scenario3GPP(Scenario):
  """This class is the parent class for the different scenarios 
    defined in the standard 3GPP TR 38.901 version 14.0.0 Release 14
    https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf
  """ 
  def __init__(self,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled=True,force_los=2):  
    """ Constructor of the 3gpp scenarios parent class. Calls the parent class constructor and
    defines the parameters of the 3gpp model (see 3GPP TR 38.901 version 14.0.0 Release 14)
    """ 
    super().__init__(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
    self._corr_LOS = 0
    """ The correlation distance of LOS parameter for the spatial consistency procedure. Table 7.6.3.1-2. """ 
    self._corr_ssp_LOS = 0
    """ The correlation distance for the short short scale parameters (ssp) in the LOS condition for the spatial consistency procedure. Table 7.6.3.1-2. """ 
    self._corr_ssp_NLOS = 0
    """ The correlation distance for the short short scale parameters (ssp) in the NLOS condition for the spatial consistency procedure. Table 7.6.3.1-2. """ 
    self.shadow_enabled = shadowing_enabled
    """ Boolean variable, it indicates if the shadowing is enabled."""  
    self.gridLSP_LOS = np.array([])
    """ An Array where each element has for one point in the scenario grid  the value of each Large scale parameter (LSP) of the 3gpp model in LOS condition. """
    self.gridLSP_NLOS = np.array([])
    """ An Array where each element has for one point in the scenario grid  the value of each Large scale parameter (LSP) of the 3gpp model in NLOS condition. """
    
    self.shadow = 0
    """ The shadow fading value for a given position of the mobile in the scenario """
    self._O2I = False 
    """ Outdoor to Indoor bolean value. Not implemented yet. """ 
    self._blockage = False 
    """Blockage model. Not implemented yet. """
    self._DScorr_distance = 0
    """Delay spread correlation distance. """ 
    self._AOD_AZScorr_distance = 0
    """Azimuth angle of departure spread correlation distance. """ 
    self._AOA_AZScorr_distance = 0
    """Azimuth angle of arraival spread correlation distance. """ 
    self._shadowScorr_distance = 0
    """Shadow fading correlatio  distance """ 
    self._AOD_ELScorr_distance = 0
    """Inclination angle of departure spread correlation distance. """ 
    self._AOA_ELScorr_distance = 0
    """Inclination angle of arraival spread correlation distance. """ 
    self._Kcorr_distance = 0
    """Ricean K factor correlation distance """ 
    self.n_scatters = 0
    """Number of the scatters of the model. """ 
    self._raysPerCluster = 0
    """Number of rays of each scatter. """ 
    self._muDSLg = 0
    """ The mean value of the delay spred in logarihtmic scale. """ 
    self._sigmaDSLg = 0
    """ The standar deviation value of the delay spred in logarihtmic scale. """
    self._muASDLg = 0
    """ The mean value of the azimuth departure angle spread in logarihtmic scale. """ 
    self._sigmaASDLg = 0
    """ The standar deviation value of the azimuth departure angle spread in logarihtmic scale. """ 
    self._muASALg = 0
    """ The mean value of the azimuth arrival angle spread in logarihtmic scale. """ 
    self._sigmaASALg = 0
    """ The standar deviation value of the azimuth arrival angle spread in logarihtmic scale. """ 
    self._muZSALg = 0
    """ The mean value of the inclination arrival angle spread in logarihtmic scale. """ 
    self._sigmaZSALg = 0
    """ The standard deviation value of the inclination arrival angle spread in logarihtmic scale. """ 
    self._sigmaShadow = 0
    """ The standard deviation of the shadow fading """ 
    self._cDS =0
    """ Is the intracluster delay spread specified in Table 7.5-6. """ 
    self._cASD = 0
    """ cASD is the cluster-wise rms spread of ASD  in Table 7.5-6. """ 
    self._cASA = 0
    """ cASA is the cluster-wise rms spread of ASA  in Table 7.5-6. """ 
    self._cZSA = 0
    """ cZSA is the cluster-wise rms spread of ZOA  in Table 7.5-6. """ 
    self._offsetZOD = 0
    """the mean value of the offset of ZOD in Tables 7.5.6./7/8""" 
    self._muZSDLg = 0
    """ The mean value of the inclination departure angle spread in logarihtmic scale. """ 
    self._sigmaZSDLg = 0
    """ The standard deviation value of the inclination departure angle spread in logarihtmic scale. """ 
    self._muKLg = 0
    """ The Ricean K factor mean value in logarithmc scale."""
    self._sigmaKLg = 0
    """ The Ricean K factor standard deviation value in logarithmc scale."""
    self._rTau = 0
    """ Is the delay distribution proportionality factor.""" 
    self._muXpr = 0
    """The cross polarization spread mean value. """ 
    self._sigmaXpr = 0
    """The cross polarization spread standard deviation value. """ 
    self._perClusterShadowingStd = 0
    """ Is the per cluster shadowing term in [dB]. """ 
    self._paranum = 0
    """ The number of LSP parameters of the model """ 
    self._corrLSPparams = [[]]    
    """The cross correlation matrix between the LSP parameters.""" 
    self.name = " "
    """The name of the scenario""" 
    self.force_los = force_los
    """ It takes the value 0 if the LOS condition is forced to False, it takes the value 1 if the LOS condition is forced to True and it takes the value 2 if the LOS condition is computed from the probability model. """ 
    
  @property  
  def corr_ssp_LOS(self):
    """ The correlation distance for the short scale parameters (ssp) in the LOS condition for the spatial consistency procedure. Table 7.6.3.1-2. """ 
    return self._corr_ssp_LOS

  @property  
  def corr_ssp_NLOS(self):
    """ The correlation distance for the short short scale parameters (ssp) in the NLOS condition for the spatial consistency procedure. Table 7.6.3.1-2. """ 
    return self._corr_ssp_NLOS

  @property  
  def rTau(self):
    """ Is the delay distribution proportionality factor.""" 
    return self._rTau

  @property  
  def perClusterShadowingStd(self):
    """ Is the per cluster shadowing term in [dB]. """ 
    return self._perClusterShadowingStd

  @property  
  def O2I(self):
    """ Outdoor to Indoor bolean value. Not implemented yet. """ 
    return self._O2I

  @property  
  def offsetZOD(self):
    """The mean value of the offset of ZOD in Tables 7.5.6./7/8""" 
    return self._offsetZOD

  @property  
  def raysPerCluster(self):
    """Number of rays of each scatter. """ 
    return self._raysPerCluster

  @raysPerCluster.setter
  def raysPerCluster(self,value):
    """ Sets the number of rays of each scatter. """ 
    self._raysPerCluster = value

  @property  
  def cASA(self):
    """ cASA is the cluster-wise rms spread of ASA  in Table 7.5-6. """ 
    return self._cASA

  @property  
  def cASD(self):
    """ cASD is the cluster-wise rms spread of ASD  in Table 7.5-6. """ 
    return self._cASD

  @property  
  def cZSA(self):
    """ cZSA is the cluster-wise rms spread of ZOA  in Table 7.5-6. """ 
    return self._cZSA

  @property  
  def muZSDLg(self):
    """ The mean value of the delay spred in logarihtmic scale. """ 
    return self._muZSDLg
 
  @property  
  def muXpr(self):
    """The cross polarization spread mean value. """ 
    return self._muXpr

  @property  
  def sigmaXpr(self):
    """The cross polarization spread standard deviation value. """ 
    return self._sigmaXpr

  @property  
  def blockage(self):
    """Blockage model. Not implemented yet. """
    return self._blockage
 
  @property  
  def cDS(self):
    """ Is the intracluster delay spread specified in Table 7.5-6. """ 
    return self._cDS


  def save(self,path):
    """ This method saves the scenario configuration to disk.
    
    @type path: string
    @param path: The directory path to save the data.
    """     
    gridLSP = self.__generate_grid_lsp_sigma()
    gridLOS = self.__generate_grid_los()
    super().save(path)
    with open(path+'/scenario_gridLSP.npy', 'wb') as f:
        np.save(f, gridLSP)
    # with open(path+'/scenario_gridLSP_NLOS.npy', 'wb') as f:
    #     np.save(f, self.gridLSP_NLOS)
    with open(path+'/scenario_gridLOS.npy', 'wb') as f:
        np.save(f, gridLOS)
    text_file = open(path+'/scenario_name.csv', "w")
    text_file.write(self.name) 
    text_file.close()


  def _set_scenario_LSP_correlated_params(self):
    """ This method computes and sets the random values of the correlated LSP parameters. This method
    also computes the grid of correlate LOS conditions in the scenario.

    """ 

    self.gridLSP_LOS,self.gridLSP_NLOS,self.XY = self.__generate_correlated_LSP()
    self.LOS_rv = self.__generate_LOSgrid(self._corr_LOS)
    """ A grid with the LOS rv with spatial correlation in each point of the grid.""" 
    #self.set_los_positions(MS_pos)
    #self.gridLSP,self.XY = self.generate_correlated_LSP()

  
  def get_shadowing_db(self,MS_pos,type_approx):
    """ This method computes the shadowing value for the MS position, sets its values and return it.
    
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the movil device in the scenario.
    @type type_approx: int.
    @param type_approx: The type of approximation used. 0 for the closest point in the grid. 1 for interpolating between closets points in the grid.
    @return: The shadowing valu for the MS position.
    """ 
    LSP = self.generate_correlated_LSP_vector(MS_pos,type_approx)
    self.shadow = LSP[0]
    return self.shadow
      
  def __set_los_positions(self,MS_pos):
    """ This method computes the computes the LOS condition given the distance between two devices and set the LOS parameter acording to it.

    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the mobile device in the scenario.
    @return: The LOS parameter (True or False).
    """   
    if self.force_los == 2:
        self._LOS = self.is_los_cond(MS_pos)
    else:
        if self.force_los == 0:
            self._set_los(False)
        else:
            self._set_los(True)
    #d2D= np.sqrt((self.BS_pos[0] - MS_pos[0] )**2+(self.BS_pos[1] - MS_pos[1])**2)
    #self.set_params(d2D,MS_pos[2])


    return self._LOS

  def __set_los(self,value):
    """ This method sets the LOS to the value (True or False)

    @type value: Boolean.
    @param value: True if LOS, False if NLOS.
    @return: The LOS parameter (True or False).
    """    
    self._LOS = value
    return self._LOS
  
  def is_los_cond(self,MS_pos):
    """ Computes the LOS condition acording to the Line-Of-Sight (LOS) probabilities are given in Table 7.4.2-1 and
    the distance between the devices.It uses also the force_los variable. If the force_los is 0 or 1 return False or
    True only if force_los =2 the los probability is taking into account.
  
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the mobile device in the scenario.
    @return: True or False acording to the LOS condition.
    """ 
    if self.force_los == 2:
        d2D= np.sqrt((self.BS_pos[0] - MS_pos[0] )**2+(self.BS_pos[1] - MS_pos[1])**2)
        self._r = self._inverse_distance_interpol([MS_pos[0],MS_pos[1]], self.X, self.Y, self.LOS_rv)
        return self._r < self._prob_los(d2D,MS_pos[2])
    else:
        if self.force_los == 0:
            return False
        else:
            return True
    
  def _inverse_distance_interpol(self,point, X, Y, values, p = 2):
      """
      This method interpolates one point in the grid with inverse distance interpolation with module p.
      
      @type point: 2D or 3D array.
      @param point: The point in the scenario to interpolate.
      @type X: array
      @param X: the result of X,Y = meshgrid(x,y).
      @type Y: array
      @param Y: the result of X,Y = meshgrid(x,y).
      @type values: 2D array
      @param values: the values to interpolate one value for each point of the grid.
      @type p: int.
      @param p: the order of the interpolation. Defaullt 2.
      @return: the value interpolated in the point. np.sum (w * values) / np.sum (w)
      
      """

      d2D = np.sqrt ((point[0] - X) ** 2 +(point[1] - Y) ** 2) ** p
      if d2D.min () == 0:
        ret = values[np.unravel_index(d2D.argmin(), d2D.shape)]
      else:
        w = 1.0 / d2D
        ret = np.sum (w * values) / np.sum (w)
    
      return ret

      

  def __generate_correlated_LSP(self):
    """This method first generates for each LSP parameter an independent gaussian N(0,1) random variable for each point in the scenario grid. Later, 
    using cholesky method and the correlation matrix between the LSP parameters, generates a set of correlated LSP params.
    At last, the method applies to each parameter its expected value and its variance.

    """ 
    gridLSP_LOS = np.zeros((7,self.grid_number+1,self.grid_number+1))
    gridLSP_NLOS = np.zeros((6,self.grid_number+1,self.grid_number+1))    
    
    gridLSP_LOS[0],XY = self.generate_LSPgrid(self._shadowScorr_distance_LOS) 
    gridLSP_LOS[1],XY = self.generate_LSPgrid(self._Kcorr_distance_LOS)
    gridLSP_LOS[2],XY = self.generate_LSPgrid(self._DScorr_distance_LOS)
    gridLSP_LOS[3],XY = self.generate_LSPgrid(self._AOD_AZScorr_distance_LOS)
    gridLSP_LOS[4],XY = self.generate_LSPgrid(self._AOA_AZScorr_distance_LOS)
    gridLSP_LOS[5],XY = self.generate_LSPgrid(self._AOD_ELScorr_distance_LOS)
    gridLSP_LOS[6],XY = self.generate_LSPgrid(self._AOA_ELScorr_distance_LOS)

    gridLSP_NLOS[0],XY = self.generate_LSPgrid(self._shadowScorr_distance_NLOS) 
    gridLSP_NLOS[1],XY = self.generate_LSPgrid(self._DScorr_distance_NLOS)
    gridLSP_NLOS[2],XY = self.generate_LSPgrid(self._AOD_AZScorr_distance_NLOS)
    gridLSP_NLOS[3],XY = self.generate_LSPgrid(self._AOA_AZScorr_distance_NLOS)
    gridLSP_NLOS[4],XY = self.generate_LSPgrid(self._AOD_ELScorr_distance_NLOS)
    gridLSP_NLOS[5],XY = self.generate_LSPgrid(self._AOA_ELScorr_distance_NLOS)
    
    c_LOS = cholesky(self._corrLSPparams_LOS, lower=True)
    c_NLOS = cholesky(self._corrLSPparams_NLOS, lower=True)
    for i in range(self.grid_number+1):
        for j in range(self.grid_number+1):
            gridLSP_LOS[:,i,j] = np.dot(c_LOS,gridLSP_LOS[:,i,j])
            gridLSP_NLOS[:,i,j] = np.dot(c_NLOS,gridLSP_NLOS[:,i,j])

    
    return gridLSP_LOS,gridLSP_NLOS, XY

  def __grid_lsp_mu_sigma(self,MS_pos):
    """ Adds mu and sigma to each lsp (the mu and sigma depend on the MS postion)

    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the mobile device in the scenario.
    @return: the grid lsp with the corresponding mu and sigma.    
    """
    d2D= np.sqrt((self.BS_pos[0] - MS_pos[0] )**2+(self.BS_pos[1] - MS_pos[1])**2)
    los = self.is_los_cond(MS_pos)
    self._set_params(d2D,MS_pos[2],los)

    if los:
      gridLSP = np.copy(self.gridLSP_LOS)
      gridLSP[0] = gridLSP[0]* self._sigmaShadow
      gridLSP[1] = gridLSP[1]* self._sigmaKLg+self._muKLg
      gridLSP[2] = gridLSP[2]* self._sigmaDSLg+self._muDSLg
      gridLSP[3] = gridLSP[3]* self._sigmaASDLg+self._muASDLg
      gridLSP[4] = gridLSP[4]* self._sigmaASALg+self._muASALg
      gridLSP[5] = gridLSP[5]* self._sigmaZSDLg+self._muZSDLg
      gridLSP[6] = gridLSP[6]* self._sigmaZSALg+self._muZSALg 

    else:
      gridLSP = np.copy(self.gridLSP_NLOS)
      gridLSP[0] = gridLSP[0]* self._sigmaShadow
      gridLSP[1] = gridLSP[1]* self._sigmaDSLg+self._muDSLg
      gridLSP[2] = gridLSP[2]* self._sigmaASDLg+self._muASDLg
      gridLSP[3] = gridLSP[3]* self._sigmaASALg+self._muASALg
      gridLSP[4] = gridLSP[4]* self._sigmaZSDLg+self._muZSDLg
      gridLSP[5] = gridLSP[5]* self._sigmaZSALg+self._muZSALg 
      
    return gridLSP,los

  def __generate_grid_lsp_sigma(self):
    """ Adds mu and sigma to each lsp (the mu and sigma depend on the MS postion)

    @return: the grid lsp with the corresponding mu and sigma.    
    """
    gridLSP = np.zeros((7,self.grid_number+1,self.grid_number+1))

    for i in range(self.grid_number+1):
        for j in range(self.grid_number+1):
            MS_pos = [self.X[0][i],self.Y[j][0],2]
            g,los = self.__grid_lsp_mu_sigma(MS_pos)
            if los:
                gridLSP[:,i,j] = g[:,i,j]
            else:
                gridLSP[0,i,j] = g[0,i,j]
                for k in range(2,7):
                    gridLSP[k,i,j] = g[k-1,i,j]
                
    return gridLSP

  def __generate_grid_los(self):
    """ This method generates a grid los with True or False in each point depending on the LOS condition.

    @return: the grid los with True or False in each point depending on the LOS condition.    
    """
    gridLOS = np.zeros((self.grid_number+1,self.grid_number+1))

    for i in range(self.grid_number+1):
        for j in range(self.grid_number+1):
            MS_pos = [self.X[0][i],self.Y[j][0],2]
            los = self.is_los_cond(MS_pos)
            gridLOS[i][j] = los
                
    return gridLOS


  def generate_correlated_LSP_vector(self,MS_pos,type_approx):
    """ This method given the LSP parameters grid and a MS position, estimates the LSP parameters for this point.

    The method can use two approximation methods. The first one use the parameter of the closest point in the grid. 
    The second on interpolates between the two closests point in the grid.
    @type MS_pos: 3D array or list.
    @param MS_pos: the position of the mobile device in the scenario.
    @type type_approx: int.
    @param type_approx: The type of approximation. 0 for the closest point. 1 for interpolation.
    @return: the LSP parameters for the MS_pos point in the scenario.
    """ 
    d2D= np.sqrt((self.BS_pos[0] - MS_pos[0] )**2+(self.BS_pos[1] - MS_pos[1])**2)
    los = self.is_los_cond(MS_pos)
    self._set_params(d2D,MS_pos[2],los)

    if los:
      gridLSP = np.copy(self.gridLSP_LOS)
      gridLSP[0] = gridLSP[0]* self._sigmaShadow
      gridLSP[1] = gridLSP[1]* self._sigmaKLg+self._muKLg
      gridLSP[2] = gridLSP[2]* self._sigmaDSLg+self._muDSLg
      gridLSP[3] = gridLSP[3]* self._sigmaASDLg+self._muASDLg
      gridLSP[4] = gridLSP[4]* self._sigmaASALg+self._muASALg
      gridLSP[5] = gridLSP[5]* self._sigmaZSDLg+self._muZSDLg
      gridLSP[6] = gridLSP[6]* self._sigmaZSALg+self._muZSALg 

    else:
      gridLSP = np.copy(self.gridLSP_NLOS)
      gridLSP[0] = gridLSP[0]* self._sigmaShadow
      gridLSP[1] = gridLSP[1]* self._sigmaDSLg+self._muDSLg
      gridLSP[2] = gridLSP[2]* self._sigmaASDLg+self._muASDLg
      gridLSP[3] = gridLSP[3]* self._sigmaASALg+self._muASALg
      gridLSP[4] = gridLSP[4]* self._sigmaZSDLg+self._muZSDLg
      gridLSP[5] = gridLSP[5]* self._sigmaZSALg+self._muZSALg 
      
    return self.LSP_vector_position(gridLSP,MS_pos,type_approx)
    

 
  def __generate_LOSgrid(self,corr_distance):
    """This method generates a spatial correlated uniform rvs for LOS.

    The covariance matrix is defined by cov = exp(-distance/correlation_distance). Copula method is used:
    first it generates gaussian rvs with the desired correlation; second, using gaussian cdf it converts
    the rvs to uniform[0,1] rvs (this is the last step in our case). If it is necessary rvs with other distribution,
    using the inverse distribution can be generated the rvs with the desired distribution from the uniform rvs.        
    @type corr_distance: float.
    @param corr_distance: The correlation distance for the spatial correlated gaussian random variable.
    @return: A 2D matrix where the values of the matrix are spatially correlated uniform random variables.
    """ 
    LOS_normal,XY = self.generate_LSPgrid(corr_distance)
    LOS_rv= scipy.stats.norm.cdf(LOS_normal)
    return LOS_rv

class Scenario3GPPInDoor(Scenario3GPP):
  """This class implements the indoor scenario 
    defined in the standard 3GPP TR 38.901 version 14.0.0 Release 14. See Table 7.2-2 and Figure 7.2-1
    https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf

    """ 

  def __init__(self,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled=True,force_loss=2): 
    """ Constructor of the indoor 3gpp scenario. Calls the parent class constructor.

    """ 
    super().__init__(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled,force_loss)    
    self.name = "3gpp Indoor "
    self._corr_LOS = 10
    self._corr_ssp_LOS = 10
    self._corr_ssp_NLOS = 10
    self._O2I = False # output to indoor not implemented yet
    self._blockage = False # sec 7.4.6 for the moment blockage is not implemented
    self._DScorr_distance_LOS = 8
    self._AOD_AZScorr_distance_LOS = 7
    self._AOA_AZScorr_distance_LOS = 5
    self._shadowScorr_distance_LOS = 10
    self._AOD_ELScorr_distance_LOS = 4
    self._AOA_ELScorr_distance_LOS = 4
    self._Kcorr_distance_LOS = 4
    self._DScorr_distance_NLOS = 5
    self._AOD_AZScorr_distance_NLOS = 3
    self._AOA_AZScorr_distance_NLOS = 3
    self._shadowScorr_distance_NLOS = 6
    self._AOD_ELScorr_distance_NLOS = 4
    self._AOA_ELScorr_distance_NLOS = 4
    # order SH K DS AOD AOA  ZOD ZOA  
    self._corrLSPparams_LOS = [ [1,0.5,-0.8, -0.4, -0.5, 0.2, 0.3],
                            [0.5,1,-0.5,0,0,0,0.1],
                            [-0.8,-0.5,1, 0.6, 0.8, 0.1, 0.2],
                            [-0.4,0,0.6, 1,0.4, 0.5, 0 ],
                            [ -0.5,0,0.8, 0.4, 1, 0, 0.5],
                            [0.2,0,0.1,0.5, 0, 1, 0 ],
                            [0.3,0.1,0.2, 0, 0.5, 0, 1 ]]       
  
    # order SH DS AOD AOA  ZOD ZOA 
    self._corrLSPparams_NLOS = [ [1,-0.5, 0, -0.4,  0, 0],
                            [-0.5, 1, 0.4, 0,  -0.27, -0.06],
                            [0, 0.4, 1, 0,  0.35, 0.23],
                            [-0.4,0, 0, 1,  -0.08, 0.43],
                            [0,-0.27,0.35, -0.08, 1, 0.42],
                            [ 0,-0.06, 0.23, 0.43, 0.42, 1]]   

    self._set_scenario_LSP_correlated_params()

  def _set_params(self,d2d,h_MS,los):
    """ This method sets the scenario parameters.
    
    It sets for LOS and NLOS condition, the correlation distance of the LSP parameters, 
    the mean and variance of the LSP parameters and the crosscorrelation matrix between LSP parameters.
    The LSP parameters are: The Delay Spread (DS), Azimuth angle of arrival spread (AOA_AZS or ASA azimuth spread arrival), Azimuth angle of departure spread (AOD_AZS or ASD),
    Shadowing (Shadow), elevation angle of departure spread(AOD_ELS or ZSD zenith spread departure), elevation angle of arrival spread (AOA_ELS or ZSD), Ricean K factor(K).
    See Table 7.5-6 Part-2: Channel model parameters for RMa (up to 7GHz) and Indoor-Office
    @type d2d: float.
    @param d2d: The distance between the BS and MS positions. Default 10.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.    
    """ 
    if los:
      self.n_scatters = 15
      self._raysPerCluster = 20
      self._muDSLg = -0.01 * np.log10 (1 + self.fcGHz) - 7.692
      self._sigmaDSLg = 0.18
      self._muASDLg = 1.60
      self._sigmaASDLg = 0.18
      self._muASALg = -0.19 * np.log10 (1 + self.fcGHz) + 1.781;
      self._sigmaASALg = 0.12 * np.log10 (1 + self.fcGHz) + 0.119;
      self._muZSALg = -0.26 * np.log10 (1 + self.fcGHz) + 1.44;
      self._sigmaShadow = 3
      self._sigmaZSALg = -0.04 * np.log10 (1 + self.fcGHz) + 0.264;
      self._cDS = 3.91e-9;
      self._cASD = 5;
      self._cASA = 8;
      self._cZSA = 9;
      self._offsetZOD = 0
      ######Table 7.5.10
      self._muZSDLg = -1.43 * np.log10 (1 + self.fcGHz) + 2.228;
      self._sigmaZSDLg = 0.13 * np.log10 (1 + self.fcGHz) + 0.30;
      #######
      self._muKLg = 7
      self._sigmaKLg = 4
      self._rTau = 3.6
      self._muXpr = 11
      self._sigmaXpr = 4
      self._perClusterShadowingStd = 6
      self._paranum = 7
    else:
      self.n_scatters = 19
      self._raysPerCluster = 20
      self._muDSLg = -0.28 * np.log10 (1 + self.fcGHz) - 7.173
      self._sigmaDSLg = 0.10* np.log10(1+self.fcGHz) + 0.055
      self._muASDLg = 1.62
      self._sigmaASDLg = 0.25
      self._muASALg = -0.11 * np.log10(1+self.fcGHz) + 1.863
      self._sigmaASALg = 0.12 * np.log10(1+self.fcGHz) + 0.059
      self._sigmaShadow = 8.03
      self._muZSALg = -0.15 * np.log10(1+self.fcGHz) + 1.387
      self._sigmaZSALg = -0.09 * np.log10(1+self.fcGHz) + 0.746
      self._cDS = 3.91e-9 #where cDS is cluster delay spread specified in Table 7.5-6. When intra-cluster delay spread is unspecified (i.e., N/A) 3.91 is used.
      self._cASD = 5;
      self._cASA = 11;
      self._cZSA = 9;
      self._offsetZOD = 0
      ######Table 7.5.10
      self._muZSDLg = 1.08
      self._sigmaZSDLg = 0.36 
      #######

      self._rTau = 3
      self._muXpr = 10
      self._sigmaXpr = 4
      self._perClusterShadowingStd = 3
      self._paranum = 6
      
  def get_loss_los (self,distance3D,h_MS=2):  
    """ This method computes in the LOS case the pathloss of the 3gpp indoor scenario (see 3GPP TR 38.901, Table 7.4.1-1) 
    
    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.    
    @return: 32.4 + 17.3 * np.log10 (distance3D) + 20.0 * np.log10(self.fcGHz)
    """ 
    #check if the distance is outside the validity range
    if (distance3D < 1.0 or distance3D > 150.0):
      print("The 3D distance is outside the validity range for indoor office",distance3D)
    loss = 32.4 + 17.3 * np.log10 (distance3D) + 20.0 * np.log10(self.fcGHz)
    return loss

  def get_loss_nlos (self,distance3D,h_MS=2):  
    """ This method computes in the NLOS case the pathloss of the 3gpp indoor scenario (see 3GPP TR 38.901, Table 7.4.1-1) 
   
    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.     
    @return: max(self.get_loss_los(distance3D,h_MS), 17.3 + 38.3 * np.log10 (distance3D) + 24.9 * np.log10(self.fcGHz))
    """ 
    # check if the distance is outside the validity range
    if (distance3D < 1.0 or distance3D > 150.0):
      print("The 3D distance is outside the validity range for indoor office",distance3D)
    plNlos = 17.3 + 38.3 * np.log10 (distance3D) + 24.9 * np.log10(self.fcGHz)
    loss = max(self.get_loss_los(distance3D,h_MS), plNlos)
    return loss

        
  def _prob_los(self, d2D,h_MS=2):
    """ Computes the LOS probability acording to the Line-Of-Sight (LOS) probabilities are given in Table 7.4.2-1 and
    the distance between the devices.
    
    @type d2D: float.
    @param d2D: The 2D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.     
    @return: LOS probability
    """ 
    if (d2D <= 5.0):       
      pLos = 1.0
    elif (d2D > 5.0 and d2D <= 49.0):
      pLos = np.exp (-(d2D - 5.0) / 70.8)
    else:
      pLos = np.exp (-(d2D - 49.0) / 211.7) * 0.54
    return pLos;

class Scenario3GPPUma(Scenario3GPP):
  """This class implements the Uma 3gpp scenario 
    defined in the standard 3GPP TR 38.901 version 14.0.0 Release 14. See Table 7.2-1.
    https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf

  """ 
  def __init__(self,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled=True,force_loss=2): 
    """ The constructor of the Uma 3gpp scenario. Calls the parent class constructor.
    """ 
    super().__init__(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled,force_loss)    
    self.name = "3gpp Uma "
    self._corr_LOS = 50
    self._corr_ssp_LOS = 40
    self._corr_ssp_NLOS = 50
    self._O2I = False # output to indoor not implemented yet
    self._blockage = False # sec 7.4.6 for the moment blockage is not implemented
    # Table 7.5.6 part 1
    self._DScorr_distance_LOS = 30
    self._AOD_AZScorr_distance_LOS = 18
    self._AOA_AZScorr_distance_LOS = 15
    self._shadowScorr_distance_LOS = 37
    self._AOD_ELScorr_distance_LOS = 15
    self._AOA_ELScorr_distance_LOS = 15
    self._Kcorr_distance_LOS = 12
    self._DScorr_distance_NLOS = 40
    self._AOD_AZScorr_distance_NLOS = 50
    self._AOA_AZScorr_distance_NLOS = 50
    self._shadowScorr_distance_NLOS = 50
    self._AOD_ELScorr_distance_NLOS = 50
    self._AOA_ELScorr_distance_NLOS = 50
    # order SH K DS AOD AOA  ZOD ZOA  
    self._corrLSPparams_LOS = [ [1,0,-0.4, -0.5, -0.5, 0, -0.8],
                            [0 ,1, -0.4,0 ,-0.2,0 ,0 ],
                            [-0.4,-0.4,1, 0.4, 0.8, -0.2, 0],
                            [-0.5,0,0.4, 1,0 , 0.5, 0 ],
                            [ -0.5,-0.2,0.8, 0, 1, -0.3, 0.4],
                            [0,0,-0.2,0.5, -0.3, 1, 0 ],
                            [-0.8, 0 , 0 , 0, 0.4, 0, 1 ]]       
    # order SH DS AOD AOA  ZOD ZOA 
    self._corrLSPparams_NLOS = [ [1,-0.4, -0.6, 0 ,  0, -0.4],
                            [-0.4, 1, 0.4, 0.6,  -0.5, 0],
                            [-0.6, 0.4, 1, 0.4,  0.5, -0.1],
                            [0,0.6, 0.4, 1, 0, 0],
                            [0,-0.5,0.5, 0, 1, 0],
                            [-0.4,0, -0.1, 0, 0, 1]]   
    self._set_scenario_LSP_correlated_params()
  
 
  def _set_params(self,d2d,h_MS,los):
    """ This method sets the scenario parameters.
    
    It sets for LOS and NLOS condition, the correlation distance of the LSP parameters, 
    the mean and variance of the LSP parameters and the crosscorrelation matrix between LSP parameters.
    The LSP parameters are: The Delay Spread (DS), Azimuth angle of arrival spread (AOA_AZS or ASA azimuth spread arrival), Azimuth angle of departure spread (AOD_AZS or ASD),
    Shadowing (Shadow), elevation angle of departure spread(AOD_ELS or ZSD zenith spread departure), elevation angle of arrival spread (AOA_ELS or ZSD), Ricean K factor(K).
    See Table 7.5-6 Part-1.
    @type d2d: float.
    @param d2d: The distance between the BS and MS positions. Default 10.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.        
    """ 
 
    if los:
      self.n_scatters = 12
      self._raysPerCluster = 20
      self._muDSLg = -0.0963 * np.log10 (self.fcGHz) - 6.955
      self._sigmaDSLg = 0.66
      self._muASDLg = 1.06 +0.1114* np.log10 (self.fcGHz)
      self._sigmaASDLg = 0.28
      self._muASALg = 1.81
      self._sigmaASALg = 0.2
      self._muZSALg = 0.95
      self._sigmaShadow = 4 # 7.4.1.1
      self._sigmaZSALg = 0.16 #
      self._cDS = 1e-9*max(0.25, 6.5622 -3.4084 * np.log10(self.fcGHz))
      self._cASD = 5 #
      self._cASA = 11#
      self._cZSA = 7 #
      self._offsetZOD = 0
      ######Table 7.5.10
      self._muZSDLg = max(-0.5, -2.1*(d2d/1000) -0.01 *(h_MS - 1.5)+0.75)
      self._sigmaZSDLg = 0.4
      #######
      self._muKLg = 9 #
      self._sigmaKLg = 3.5 #
      self._rTau = 2.5 #
      self._muXpr = 8 #
      self._sigmaXpr = 4 #
      self._perClusterShadowingStd = 3 #
      self._paranum = 7
    else:
      self.n_scatters = 20
      self._raysPerCluster = 20 #
      self._muDSLg = -6.28 - 0.204*np.log10(self.fcGHz) #
      self._sigmaDSLg = 0.39
      self._muASDLg = 1.5 - 0.1144*np.log10(self.fcGHz)
      self._sigmaASDLg = 0.28#
      self._muASALg = 2.08 - 0.27*np.log10(self.fcGHz)
      self._sigmaASALg = 0.11#
      self._sigmaShadow = 6 # 7.4.1.1
      self._muZSALg =-0.3236*np.log10(self.fcGHz) + 1.512
      self._sigmaZSALg = 0.16#
      self._cDS = 1e-9*max(0.25, 6.5622 -3.4084*np.log10(self.fcGHz))
      self._cASD = 2
      self._cASA = 15
      self._cZSA = 7#
      self._offsetZOD = 7.66*np.log10(self.fcGHz)-5.96-10**((0.208*np.log10(self.fcGHz)- 0.782)* np.log10(max(25, d2d)) -0.13*np.log10(self.fcGHz)+2.03 -0.07*(h_MS-1.5))
      ######Table 7.5.10
      self._muZSDLg = max(-0.5, -2.1*(d2d/1000) -0.01*(h_MS - 1.5)+0.9)
      self._sigmaZSDLg = 0.49
      #######

      self._rTau = 2.3
      self._muXpr = 7
      self._sigmaXpr = 3#
      self._perClusterShadowingStd = 3#
      self._paranum = 6
     
  def get_loss_los (self,distance3D,h_MS):  
    """ This method computes in the LOS case the pathloss of the 3gpp Uma scenario (see 3GPP TR 38.901, Table 7.4.1-1) 

    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: the path loss in los condition for 3gpp Uma model. See3GPP TR 38.901 version 14.0.0 Release 14,
    """ 
    dBP = np.array(4*self.BS_pos[2]*h_MS*self.fcGHz*1e9/3e8)
    #check if the distance is outside the validity range
    if (distance3D < 10.0 or distance3D > 10000):
      print("Warning: The 3D distance is outside the validity range for UMA model",distance3D)
    if(h_MS < 1.5 or h_MS >22.5 ):
      print("Warning: The MS Height is outside the validity range for UMA model",h_MS)

    if (distance3D < dBP):
        loss =28.0+22*np.log10(distance3D)+20*np.log10(self.fcGHz)
    else:
        loss = 28.0+40*np.log10(distance3D)+20*np.log10(self.fcGHz)-9*np.log10((dBP)**2+(self.BS_pos[2]-h_MS)**2)
    return loss

  def get_loss_nlos (self,distance3D,h_MS):  
    """ This method computes in the NLOS case the pathloss of the 3gpp Uma scenario (see 3GPP TR 38.901, Table 7.4.1-1) 

    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: the path loss in nlos condition for 3gpp Uma model. See3GPP TR 38.901 version 14.0.0 Release 14,
    """ 
    plNlos = 13.54+39.08*np.log10(distance3D)+20*np.log10(self.fcGHz)-0.6*(h_MS-1.5)    
    loss = max(self.get_loss_los(distance3D,h_MS), plNlos)
    return loss



  def _prob_los(self, d2d,h_MS):
    """ Computes the LOS probability acording to the Line-Of-Sight (LOS) probabilities are given in Table 7.4.2-1 and
    the distance between the devices.

    @type d2d: float.
    @param d2d: The 2D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: LOS probability
    """ 
    if (d2d <= 18.0):       
      pLos = 1.0
    else:
      C = 0
      if h_MS> 13:
         C=((h_MS-13)/10)**(1.5)
      pLos = (18/d2d+np.exp(-d2d/63)*(1-18/d2d))*(1+C*5/4**(d2d/100)**3) *np.exp(-d2d/150)
    return pLos

class Scenario3GPPUmi(Scenario3GPP):
  """This class implements the 3gpp Umi scenario 
    defined in the standard 3GPP TR 38.901 version 14.0.0 Release 14. See Table 7.2-1.
    https://www.etsi.org/deliver/etsi_tr/138900_138999/138901/14.00.00_60/tr_138901v140000p.pdf

  """ 
  def __init__(self,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled=True,force_loss=2): 
    """ The constructor of the Umi 3gpp scenario. Calls the parent class constructor.

    """ 
    super().__init__(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,shadowing_enabled,force_loss)    
    self.name = "3gpp Umi "
    self._corr_LOS = 50
    self._corr_ssp_LOS = 12
    self._corr_ssp_NLOS = 15
    self._O2I = False # output to indoor not implemented yet
    self._blockage = False # sec 7.4.6 for the moment blockage is not implemented
    # Table 7.5.6 part 1
    self._DScorr_distance_LOS = 7
    self._AOD_AZScorr_distance_LOS = 8
    self._AOA_AZScorr_distance_LOS = 8
    self._shadowScorr_distance_LOS = 10
    self._AOD_ELScorr_distance_LOS = 12
    self._AOA_ELScorr_distance_LOS = 12
    self._Kcorr_distance_LOS = 15
    self._DScorr_distance_NLOS = 10
    self._AOD_AZScorr_distance_NLOS = 10
    self._AOA_AZScorr_distance_NLOS = 9
    self._shadowScorr_distance_NLOS = 13
    self._AOD_ELScorr_distance_NLOS = 10
    self._AOA_ELScorr_distance_NLOS = 10
    self._corrLSPparams_LOS = [ [1,0.5,-0.4, -0.5, -0.4, 0, 0],
                            [0.5 ,1, -0.7,-0.2 ,-0.3,0 ,0 ],
                            [-0.4,-0.7,1, 0.5, 0.8, 0, 0.2],
                            [-0.5,-0.2,0.5, 1,0.4 , 0.5, 0.3 ],
                            [ -0.4,-0.3,0.8, 0.4, 1, 0, 0],
                            [0,0,0,0.5, 0, 1, 0 ],
                            [0, 0 , 0.2 , 0.3, 0, 0, 1 ]]       

    # order SH DS AOD AOA  ZOD ZOA 
    self._corrLSPparams_NLOS = [ [1,-0.7, 0 , -0.4 , 0 , 0 ],
                            [-0.7, 1, 0 , 0.4,  -0.5, 0],
                            [0, 0 , 1, 0 ,  0.5, 0.5],
                            [-0.4,0.4, 0 , 1, 0, 0.2],
                            [0 ,-0.5,0.5, 0, 1, 0],
                            [0 ,0, 0.5, 0.2, 0, 1]]   
    
    self._set_scenario_LSP_correlated_params()

 
  def _set_params(self,d2d,h_MS,los):
    """ This method sets the scenario parameters.

    It sets for LOS and NLOS condition the correlation distance of the LSP parameters, 
    the mean and variance of the LSP parameters and the crosscorrelation matrix between LSP parameters.
    The LSP parameters are: The Delay Spread (DS), Azimuth angle of arrival spread (AOA_AZS or ASA azimuth spread arrival), Azimuth angle of departure spread (AOD_AZS or ASD),
    Shadowing (Shadow), elevation angle of departure spread(AOD_ELS or ZSD zenith spread departure), elevation angle of arrival spread (AOA_ELS or ZSD), Ricean K factor(K).
    See Table 7.5-6 Part-1.
    @type d2d: float.
    @param d2d: The distance between the BS and MS positions. Default 10.
    @type h_MS: float.
    @param h_MS: The MS antenna height. Default 2.            
    """ 

    if los:
      self.n_scatters = 12
      self._raysPerCluster = 20
      self._muDSLg = -0.24 * np.log10 (1+self.fcGHz) - 7.14
      self._sigmaDSLg = 0.38
      self._muASDLg = 1.21 -0.05* np.log10 (1+self.fcGHz)
      self._sigmaASDLg = 0.41
      self._muASALg = 1.73 -0.08* np.log10 (1+self.fcGHz)
      self._sigmaASALg = 0.28 -0.014* np.log10 (1+self.fcGHz)
      self._muZSALg = 0.73 -0.1* np.log10 (1+self.fcGHz)
      self._sigmaShadow = 4 # 7.4.1.1
      self._sigmaZSALg =0.34 -0.04* np.log10 (1+self.fcGHz)
      self._cDS = 5*1e-9 #
      self._cASD = 3 #
      self._cASA = 17 #
      self._cZSA = 7 #
      self._offsetZOD = 0
      ######Table 7.5.10
      self._muZSDLg = max(-0.21, -14.8*(d2d/1000) +0.11 *abs(h_MS - self.BS_pos[2])+0.83)
      self._sigmaZSDLg = 0.35
      #######
      self._muKLg = 9 #
      self._sigmaKLg = 5 #
      self._rTau = 3#
      self._muXpr = 9 #
      self._sigmaXpr = 3# 
      self._perClusterShadowingStd = 3 #
      self._paranum = 7
      # order SH K DS AOD AOA  ZOD ZOA  
    else:
      self.n_scatters = 19
      self._raysPerCluster = 20 #
      self._muDSLg = -6.83 - 0.24*np.log10(1+self.fcGHz) #
      self._sigmaDSLg = 0.28 + 0.16*np.log10(1+self.fcGHz)
      self._muASDLg = 1.53 - 0.23*np.log10(1+self.fcGHz)
      self._sigmaASDLg = -6.83 - 0.24*np.log10(1+self.fcGHz)
      self._muASALg = 1.81 -0.08*np.log10(1+self.fcGHz)
      self._sigmaASALg = 0.3 + 0.05*np.log10(1+self.fcGHz)
      self._sigmaShadow = 7.82 # 7.4.1.1
      self._muZSALg =-0.04*np.log10(self.fcGHz) + 0.92
      self._sigmaZSALg = 1.41 - 0.07*np.log10(1+self.fcGHz)
      self._cDS = 11e-9#
      self._cASD = 10 #
      self._cASA = 22 #
      self._cZSA = 7 #
      self._offsetZOD = 10**(-1.5*np.log10(max(10,d2d))+3.3)
      ######Table 7.5.10
      self._muZSDLg = max(-0.5, -3.1*(d2d/1000) +0.01*max(h_MS - self.BS_pos[2],0)+0.2)
      self._sigmaZSDLg = 0.35
      #######

      self._rTau = 2.1#
      self._muXpr = 8#
      self._sigmaXpr = 3#
      self._perClusterShadowingStd = 2.1#
      self._paranum = 6
       
  def get_loss_los (self,distance3D,h_MS):  
    """ This method computes in the LOS case the pathloss of the 3gpp Umi scenario (see 3GPP TR 38.901, Table 7.4.1-1) 
    @type distance3D: float.

    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: the path loss in los condition for 3gpp Umi model. See3GPP TR 38.901 version 14.0.0 Release 14,
    """ 
    dBP = np.array(4*self.BS_pos[2]*h_MS*self.fcGHz*1e9/3e8)
    #check if the distance is outside the validity range
    if (distance3D < 10.0 or distance3D > 5000):
      print("Warning: The 3D distance is outside the validity range for UMI model",distance3D)
    if(h_MS < 1.5 or h_MS >22.5 ):
      print("Warning: The MS Height is outside the validity range for UMI model",h_MS)

    if (distance3D < dBP):
        loss =32.4+21*np.log10(distance3D)+20*np.log10(self.fcGHz)
    else:
        loss = 32.4+40*np.log10(distance3D)+20*np.log10(self.fcGHz)-9.5*np.log10((dBP)**2+(self.BS_pos[2]-h_MS)**2)
    return loss

  def get_loss_nlos (self,distance3D,h_MS):  
    """ This method computes in the NLOS case the pathloss of the 3gpp Umi scenario (see 3GPP TR 38.901, Table 7.4.1-1) 

    @type distance3D: float.
    @param distance3D: The 3D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: the path loss in nlos condition for 3gpp Umi model.
    """ 
    plNlos = 22.4+35.3*np.log10(distance3D)+21.3*np.log10(self.fcGHz)-0.3*(h_MS-1.5)    
    loss = max(self.get_loss_los(distance3D,h_MS), plNlos)
    return loss


         
  def _prob_los(self, d2d,h_MS):
    """ Computes the LOS probability acording to the Line-Of-Sight (LOS) probabilities are given in Table 7.4.2-1 and
    the distance between the devices.
 
    @type d2d: float.
    @param d2d: The 2D distance in meters between the Tx and Rx devices. 
    @type h_MS: float.
    @param h_MS: The MS antenna height.     
    @return: LOS probability
    """ 
    if (d2d <= 18.0):       
      pLos = 1.0
    else:
      pLos = (18/d2d+np.exp(-d2d/63)*(1-18/d2d))
    return pLos




