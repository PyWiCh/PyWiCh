#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module implements different antennas models and calculates the field and
power patterns

@author: pablo belzarena
"""
import numpy as np
import angles as an


class Antenna:
  """This class is the parent class for all antenna models
  """  
  def __init__(self):
    """
    Default cosntructor that only set the gain of the antenna in db.
    
    """
    self.gaindb = 1
    """Maximum directional gain of the antenna. Default value 1 db """

  def get_gaindb(self,angle):
    """
    This is the default method.This method computes the power radiation of the antenna in the specified angle.
    
    @type angle: Class Angle object.
    @param angle: The angle to  calculate the power gain in db.
    """
    return self.gaindb

      
class AntennaIsotropic(Antenna):
  """
  This class implements an isotropic antenna model
  """  
  def __init__(self,maxgaindb=1):
    """This is the constructor of the isotropic antenna model.

    @type maxgaindb: float.
    @param maxgaindb: Maximum directional gain of an isotropic antenna element in db.
    """  
    self.gaindb = maxgaindb
    """Maximum directional gain of the antenna in db """
 
  def save(self,path,name):
    """This is method saves to disk the configuration of the isotropic antenna model.

    @type path: string.
    @param path: the path to the directory where the antenna configuration will be saved.
    @type name: string.
    @param name: A name that identifies the antenna. For example indicates if the antenna is Tx or Rx.
    """ 
    antype = type(self).__name__
    text_file = open(path+'/antenna_element'+name+'_type.csv', "w")
    text_file.write(antype) 
    text_file.close()
    np.savetxt(path+'/antenna_element'+name+'.csv', [self.gaindb], delimiter=',')      
 

class Antenna3gpp3D(Antenna):
  """
  This class implements the 3gpp antenna model defined in table 7.3-1 in 3GPP TR 38.901. 
  """  
  def __init__(self,maxgaindb=1,A_max=30,SLA_v=30,beamwidth=65):
    """
    This is the constructor of Antenna3gpp3D Class.

    @type maxgaindb: float .
    @param maxgaindb: Maximum directional gain of the antenna. Default value 0 db.In 3GPP TR 38.901 value 8 db.
    @type A_max: float .
    @param A_max: Front-back ratio in db. Defualt value 30 db.
    @type SLA_v: float .
    @param SLA_v: Slidelobe level limit in db. Default 30 db.
    @type beamwidth: float .
    @param beamwidth: Beamwidth of the antenna in degrees. Default 65 degrees.   
    """ 
    self.A_max= A_max 
    """ Front-back ratio in db."""
    self.SLA_v = SLA_v 
    """ Slidelobe level limit in db."""
    self.maxgaindb = maxgaindb
    """The Maximum directional gain of the antenna. """
    self.beamwidth = beamwidth
    """ Beamwidth of the antenna in degrees."""

  def save(self,path,name):
    """This is method saves to disk the configuration of the isotropic antenna model.
 
    @type path: string.
    @param path: the path to the directory where the antenna configuration will be saved.
    @type name: string.
    @param name: A name that identifies the antenna. For example indicates if the antenna is Tx or Rx.
    """ 
    antype = type(self).__name__
    text_file = open(path+'/antenna_element'+name+'_type.csv', "w")
    text_file.write(antype) 
    text_file.close()
    np.savetxt(path+'/antenna_element'+name+'.csv', [self.maxgaindb,self.A_max,self.SLA_v,self.beamwidth], delimiter=',')      
    

  def get_gaindb(self,angle_prime):
    """
    This method computes the power radiation of the antenna in the specified angle.

    The azimuth is forced in the method to[-pi,pi].table 7.3-1 in 3GPP TR 38.901
    The inclination is forced in the method to[0,pi]. The angles are in the local
    coordinate system.
    @type angle_prime: Class Angles .
    @param angle_prime: The azimuth and inclination angle in the local reference system of the antenna.
    @return: maxgaindb-min(A_max,min(12*(azimuth/beamwidth)**2,A_max)+min(12*((inclination-90)/beamwidth)**2,SLA_v)).
    """
    phi = angle_prime.phi 
    while (phi < -np.pi):
      phi = phi + np.pi
    while (phi > np.pi):
      phi = phi - np.pi  
    theta = angle_prime.theta
    while (theta > np.pi*2):
      theta = theta - 2*np.pi
    while (theta < 0):
      theta = theta + 2*np.pi
    if (theta > np.pi):
      theta = theta -np.pi 
    phiDeg = phi*180/np.pi
    thetaDeg = theta*180/np.pi
    A_v = -1 * min (self.SLA_v,12 * np.power((thetaDeg - 90)/self.beamwidth,2)) # vertical cut of the radiation power pattern (dB)
    A_h = -1 * min (self.A_max,12 * np.power(phiDeg/self.beamwidth,2)) # horizontal cut of the radiation power pattern (dB)
    A = self.maxgaindb - 1 * min (self.A_max,- A_v - A_h) #power radiation pattern
    return A 
    
 
class AntennaArray3gpp(Antenna):
  """
  This class implements one panel of the antenna array model specified in 3GPP TR 38.901.
  
  On each antenna panel, antenna elements are placed in the vertical and horizontal direction, where n_cols is the number of columns, n_rows is the number of antenna elements in each column. 
  Antenna numbering on the panel illustrated in Figure 7.3-1 assumes observation of the antenna array from the front (with x-axis pointing towards broad-side and increasing y-coordinate for increasing column number). 
  The antenna elements are uniformly spaced in the horizontal direction with a spacing of d_h and in the vertical direction with a spacing of d_v. 
  The antenna panel is either single polarized (P =1) or dual polarized (P =2). In this version only P=1 is implemented.
  """ 
  def __init__(self,d_h,d_v,n_rows,n_cols,bearing_angle,downtilt_angle,slant_angle,antenna_element,polarization,name=" "):
    """
    This method is the constructor method  of AntennaArray3gpp.

    @type d_h: float.
    @param d_h: The antenna elements are uniformly spaced in the horizontal direction with a spacing of d_h. In number of wavelength. 
    @type d_v: float.
    @param d_v: The antenna elements are uniformly spaced in the vertical direction with a spacing of d_v. In number of wavelength. 
    @type n_cols: float .
    @param n_cols: The number of columns of the panel.
    @type n_rows: float .
    @param n_rows: The number of rows of the panel.
    @type bearing_angle: float .
    @param bearing_angle: Bearing angle of the transformation of the local coordinate system (LCS) to a global coordinate system(GCS). See section 7.1.3 of 3GPP TR 38.901
    @type downtilt_angle: float .
    @param downtilt_angle: Downtilt angle of the transformation of the local coordinate system to a global coordinate system. See section 7.1.3 of 3GPP TR 38.901
    @type slant_angle: float .
    @param slant_angle: Slant angle of the transformation of the local coordinate system to a global coordinate system. See section 7.1.3 of 3GPP TR 38.901
    @type antenna_element: Class Antena.
    @param antenna_element: The type of the antenna elements of the panel. 
    @type polarization: integer.
    @param polarization: Equal 1 for single polarization, equal 2 for dual polarization. 
    @type name: string.
    @param name: A name that identifies the antenna array. For example indicates if the antenna is Tx or Rx.
    """ 
    self.d_h = d_h
    """ The antenna elements are uniformly spaced in the horizontal direction with a spacing of d_h. In number of wavelength. """ 
    self.d_v = d_v
    """ The antenna elements are uniformly spaced in the vertical direction with a spacing of d_v. In number of wavelength.""" 
    self.n_cols = n_cols
    """  The number of columns of the panel."""
    self.n_rows = n_rows
    """ The number of rows of the panel."""
    self.alpha = bearing_angle
    """ Bearing angle of the transformation of LCS to GCS. See section 7.1.3 of 3GPP TR 38.901. """
    self.beta = downtilt_angle
    """" Downtilt angle of the transformation of LCS to GCS. See section 7.1.3 of 3GPP TR 38.901. """
    self.gamma = slant_angle
    """ Slant angle of the transformation of LCS to GCS. See section 7.1.3 of 3GPP TR 38.901. """
    self.antenna_element = antenna_element
    """ The type of the antenna elements of the panel.  """
    if polarization != 1 and polarization != 2:
        print("Error polarization must be 1 or 2" )
    self.polarization = polarization
    """ Equal 1 for single polarization, equal 2 for dual polarization. """ 
    self.antenna_name = name 
    """ A name that identifies the antenna array. For example indicates if the antenna is Tx or Rx.""" 
    self.beamforming_vector = None
    """ The beamforming vector """

  def save(self,path):
    """ This method save the antenna array configuration and also call the save method of the antenna element.
 
    @type path: string.
    @param path: The directory path to save the data.    
    """ 
    name = type(self).__name__
    text_file = open(path+'/' +self.antenna_name+'_type.csv', "w")
    text_file.write(name) 
    text_file.close()
    np.savetxt(path+'/' +self.antenna_name+'.csv', [self.get_number_of_elements(),self.d_h,self.d_v,self.n_cols,self.n_rows,self.alpha,self.beta,self.gamma,self.polarization], delimiter=',')      
    self.antenna_element.save(path,self.antenna_name)

  def GCS_to_LCS(self,angle):
    """
    This method calculates the angle in LCS given an angle in GCS.
    
    @type angle: Class Angles .
    @param angle: The azimuth and inclination angle in the GCS.
    @return: the angle in the LCS
    """
    a = angle
    #convert from GCS to LCS. Eq. 7.1-7 and 7.1-8 in 3GPP TR 38.901
    thetaPrime = np.arccos(np.cos(self.beta)*np.cos(self.gamma)*np.cos(a.theta) + (np.sin(self.beta)*np.cos(self.gamma)*np.cos(a.phi-self.alpha)-np.sin(self.gamma)*np.sin(a.phi-self.alpha))*np.sin(a.theta));
    phiPrime = np.angle(complex(np.cos(self.beta)*np.sin(a.theta)*np.cos(a.phi-self.alpha) - np.sin(self.beta)*np.cos(a.theta), 
                                np.cos(self.beta)*np.sin(self.gamma)*np.cos(a.theta) +
                                (np.sin(self.beta)*np.sin(self.gamma)*np.cos(a.phi-self.alpha)+
                                np.cos(self.gamma)*np.sin(a.phi-self.alpha))*np.sin(a.theta)))
    ang = an.Angles(phiPrime,thetaPrime)
    return ang
      
    
  def get_element_field_pattern(self,angle,index):
    """
    This method calculates the field radiation of the antenna in the specified angle.
    
    The index is number of the antenna element where the antenna element in the left bottom corner has coordinates (0,0,0). 
    In case of dual polarization the first n_row*n_cols elements have psi = 45 the las n_rows*n_cols elements have psi = -45.
    @type angle: Class Angles .
    @param angle: The azimuth and inclination angle in the GCS.
    @type angle: Integer.
    @param index: The index number of the antenna element in the array panel. For single polarization (vertical polarization) 
    the index does not matter in this method. For dual polarization, all indexes from 0 to (n_row * n_cols-1) have polarization angle 45 and all indexes
    form n_rows*n_cols to (2*n_rows*n_cols -1) have polarization angle -45 degrees.
    @return: The azimuth, inclination angles of the field pattern in GCSand in radians.
    """  
    a = angle
    if self.polarization == 1:
        polarization_angle = 0
    elif index < self.n_rows*self.n_cols:
        polarization_angle = np.pi/4
    else:
        polarization_angle = -np.pi/4
    ang = self.GCS_to_LCS(a)
    aPrimeDb = self.antenna_element.get_gaindb(ang)
    #from power in db to linear
    aPrime = np.power(10, aPrimeDb / 10) 
    # Eq. 7.1-15 in 3GPP TR 38.901,
    zeda = np.angle(complex(np.sin(self.gamma)* np.cos(a.theta)*np.sin(a.phi-self.alpha)+
                           np.cos(self.gamma)*(np.cos(self.beta) * np.sin(a.theta) - 
                                               np.sin(self.beta) * np.cos(a.theta)* np.cos(a.phi - self.alpha)), 
                           np.sin(self.gamma)*np.cos(a.phi-self.alpha)+
                           np.sin(self.beta)* np.cos(self.gamma)*np.sin(a.phi-self.alpha)))
    # compute the antenna element field pattern eq. 7.3-4 and 7.3.5 in 3GPP TR 38.901
    field_theta_prime = np.sqrt(aPrime)*np.cos(polarization_angle)
    field_phi_prime = np.sqrt(aPrime)*np.sin(polarization_angle)
    # convert the antenna element field pattern to GCS using eq. 7.1-11
    field_theta = np.cos(zeda) * field_theta_prime - np.sin(zeda)*field_phi_prime
    field_phi = np.sin(zeda) * field_theta_prime + np.cos(zeda)*field_phi_prime
    return field_phi, field_theta

  def get_element_location(self,index):
    """
    This method calculates the element coordinates in the GCS, where the antenna element in the left bottom corner has coordinates (0,0,0), and the rectangular antenna array is on the y-z plane.
 
    In the case of dual polarization all elements form 0 to (n_rows*n_cols-1) has polarization 45 degrees and the corresponding elements with polarization angle
    -45 degrees have the same location but the index are in n_rows*n_cols to (2*n_rows*n_cols-1) 
    @type index: integer .
    @param index: the index number in the panel of the antenna element. For dual polarization, all indexes from 0 to (n_row * n_cols-1) have polarization angle 45 and all indexes
    form n_rows*n_cols to (2*n_rows*n_cols -1) have polarization angle -45 degrees. The indexes j and j+n_rows*n_cols has the same location but different polarization angles.
    @return: The cordinates in GCS of the element with the given index.
    """    
    if self.polarization == 2:
        index = index% self.n_cols*self.n_rows
    if index >= self.n_cols*self.n_rows :
      print("Error: index of antena array is grather than the number of elements")
      return [0,0,0]
    xPrime = 0
    yPrime = self.d_h * (index % self.n_cols)
    zPrime = self.d_v * np.floor(index / self.n_cols)
    #convert the coordinates to the GCS using the rotation matrix 7.1-4 in 3GPP TR 38.901
    loc = [0,0,0]
    loc[0] = np.cos(self.alpha) * np.cos(self.beta) * xPrime - np.sin(self.alpha) * yPrime + np.cos(self.alpha) * np.sin(self.beta) * zPrime
    loc[1] = np.sin(self.alpha) * np.cos(self.beta) * xPrime + np.cos(self.alpha) * yPrime + np.sin(self.alpha) * np.sin(self.beta) * zPrime
    loc[2] = -np.sin(self.beta) * xPrime + np.cos(self.beta) * zPrime
    return loc

  def get_number_of_elements(self):
    """
    This method computes the number of elements in the array. In case of dual polarization in each location ther are two elments.
 
    @return: n_rows*n_cols*polarization
    """    
    return self.n_rows * self.n_cols*self.polarization
  
  def set_beamforming_vector(self,w):
    """
    This method sets the beamforming vector of the array. 

    @type w:  1D numpy array  .
    @param w: beamforming vector of dimension n_rows*n_cols*polarization
    """
    self.beamforming_vector = w

  def get_beamforming_vector(self):
    """
    This method gets the beamforming vector of the array. 

    """
    return self.beamforming_vector


  # def set_coding_matrix(self, v):
  #   """
  #   This method set the channel pre or poscoding matrix. 
  #   @type w:  2D numpy array  .
  #   @param w: Pre or post coding matrix
  #   """
  #   self.coding_matrix = v

  # def get_coding_matrix(self):
  #   """
  #   This method get the pre or post coding matrix. 
  #   """
  #   return self.coding_matrix


  def compute_phase_steering (self,phi_AOA,theta_AOA,phi_add = 0,theta_add =0):
    """
    This method computes, sets and return the steering vector.

    The steering vctor is computed using the parameters azimuth angle of arrival (or depature),
    the inclination angle of arrival (or departure). An  azimuth angle , and an inclination can be 
    added to the AOA or AOD.
    @type phi_AOA:  float.
    @param phi_AOA: The azimuth angle of arrival for Rx (or departure for Tx). The angle must be between 
    -pi and pi.
    @type theta_AOA:  float.
    @param theta_AOA: The inclination angle of arrival for Rx (or departure for Tx). The angle must be between 
    0 and pi.
    @type phi_add  float.
    @param phi_add The azimuth additional rotation angle for beamforming.Default 0 radians.
    @type theta_add:  float.
    @param theta_add: The inclination aditional rotation angle for beamforming. Default 0 radians.
    """    
    n_elements = self.get_number_of_elements()
    power = 1 / np.sqrt(n_elements)
    w = np.zeros(n_elements, dtype=complex)
    for i in range(n_elements):
      pos = self.get_element_location(i)
      ####Assuming planar array and in y,z plane
      phase = -2 * np.pi * (np.sin(theta_AOA+theta_add) * np.cos(phi_AOA+phi_add) * pos[0]+np.sin(theta_AOA+theta_add) * np.sin(phi_AOA+phi_add) * pos[1]+ np.cos(theta_AOA+theta_add) * pos[2])
      w[i] = np.exp(complex(0, phase))*power
    self.set_beamforming_vector(w)
    return w




