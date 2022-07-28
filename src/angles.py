#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module gives support to work with 3D angles in antenna and 
wireles channel systems.

@author: pablo belzarena
"""

import numpy as np

class Angles:
  """This class gives variables and methods to work with 3D angles"""
  def __init__(self,phi,theta):
    """Angles class cosntructor.
    
    This method constructs an Angles object given the azimuth and inclination 
    angles. The azimuth (phi) is measure in the horizontal plane xy from the 
    axe x positive to axe y positive and the inclination (theta) is measure in 
    the vertical plane from axe z positive to the horizontal plane.
    @type phi: float in radians.
    @param phi: azimuth angle.
    @type theta: float in radians.
    @param theta: inclination angle.
    """     
    self.phi = phi
    "Azimuth angle ."
    self.theta = theta
    "AInclination angle ."
    
  def get_angles_vectors(self,a,b):
    """This method computes the angles between two vectors a and b.
    
    This method calculates the angles phi and theta of the vector a-b.
    The method set the class variables phi and theta and return phi and 
    theta 
    @type a:  1D numpy array, size 3 .
    @param a: first vector.
    @type b: 1D numpy array, size 3 .
    @param b: second vector.
    @return: phi and theta in radians.
    """
    self.phi = np.arctan2(a[1]-b[1],a[0]-b[0])
    if self.get_distance3D(a, b) != 0:
        self.theta = np.arccos((a[2] - b[2]) / self.get_distance3D(a, b))
    else:
        self.theta=0
    return self.phi,self.theta

  def get_distance3D(self,a,b):
    """ This method computes the distance between two 3D vectors.        
    
    @type a:  1D numpy array, size 3 .
    @param a: first vector.
    @type b: 1D numpy array, size 3 .
    @param b: second vector.
    @return: the distance between both vectors.
    """
    return np.sqrt((b[0] - a[0])**2+(b[1] - a[1])**2+(b[2] - a[2])**2)

  def get_azimuth_degrees(self):
    """ This method gets the azimuth angle in degrees.
    @return: the azimuth angle in degrees.
    """
    return self.phi*180/np.pi

  def get_inclination_degrees(self):
    """ This method gets the inclination angle in degrees.      
    @return: the inclination angle in degrees.
    """
    return self.theta*180/np.pi
  
  def get_azimuth(self):
    """ This method gets the azimuth angle in radians.
    
    @return: the azimuth angle in radians.
    """
    return(self.phi)
  
  def get_inclination(self):
    """ This method gets the inclination angle in radians.
    
    @return: the inclination angle in radians.
    """
    return(self.theta)

  def wrap_angles3gpp (self,azimuthRad, inclinationRad):
    """This method wraps the azimuth to [0,2pi] and the inclination to [0,pi] 

    See 3gpp eq (7.5-18).
    Assuming that inclination  angle of arrival (theta AOA) is wrapped within [0, 360°],
    if theta AOA is within [180, 360°], then theta AOA is set to 360°-thetaAOA .    
    @type azimuthRad:  float.
    @param azimuthRad: the azimuth angle in radians.
    @type inclinationRad: float.
    @param inclinationRad: inclination angle in radians.
    @return: the azzimuth,inclination angles in radians.
    """
    inclinationRad = self.wrap_to_2pi(inclinationRad)
    if (inclinationRad > np.pi): 
      inclinationRad = 2*np.pi - inclinationRad
    azimuthRad = self.wrap_to_2pi(azimuthRad)
    return azimuthRad, inclinationRad

  def wrap_to_2pi(self,a):
    """This method wraps an angle to [0,2pi].
    
    @type a:  float.   
    @param a: the angle to be wrapped in radians.
    @return: the wrapped angle in radians.
    """
    a = np.fmod (a, 2 * np.pi)
    if (a < 0):
      a += 2 * np.pi
    return a


