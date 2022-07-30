#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the angles module.

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
from  angles import Angles
import numpy as np
class AnglesTest(unittest.TestCase):
  """Unitest class for testing angles."""   
  def test_constructor(self):
    """ This method tests the constructor of Angles class and the get functions.
    
    """ 
    a  = Angles(np.pi/2,np.pi/2)
    self.assertEqual(a.phi,np.pi/2 )
    self.assertEqual(a.theta, np.pi/2)
    self.assertEqual(a.get_azimuth(), np.pi/2)
    self.assertEqual(a.get_inclination(), np.pi/2)
    self.assertEqual(a.get_azimuth_degrees(), 90)
    self.assertEqual(a.get_inclination_degrees(), 90)
  def test_get_angles_vector(self):
    """ This method tests the creation of an Angle given two 3D vectors. 
    
    """ 
    a  = Angles(np.pi/2,np.pi/2)
    ar1=np.array([0,0,0])
    ar2= np.array([1/np.sqrt(2),1/np.sqrt(2),1])
    aux1,aux2 = a.get_angles_vectors(ar1, ar2)
    self.assertEqual(aux1,-np.pi+np.pi/4,"azimuth angle error" )
    self.assertEqual(aux2,np.pi-np.pi/4,"inclination angle error" )
    aux1,aux2 = a.get_angles_vectors(ar2, ar1)
    self.assertEqual(aux1,np.pi/4,"azimuth angle error" )
    self.assertEqual(aux2,np.pi/4,"inclination angle error" )
    ar1=np.array([0,0,0])
    ar2= np.array([0,0,0])
    aux1,aux2 = a.get_angles_vectors(ar1, ar2)
    self.assertEqual(aux1,0,"azimuth angle error")
    self.assertEqual(aux2,0,"inclination angle error")
  def test_wrap_angles(self):
    """ This method tests the functions to wrap the angles.
    
    """ 
    a  = Angles(np.pi/2,np.pi/2)
    self.assertEqual(a.wrap_to_2pi(np.pi/2),np.pi/2,"pi/2 wrap to 2 pi angle error" )
    self.assertEqual(a.wrap_to_2pi(np.pi),np.pi,"pi wrap to 2 pi angle error" )
    self.assertEqual(a.wrap_to_2pi(-np.pi),np.pi,"-pi wrap to 2 pi angle error" )
    self.assertEqual(a.wrap_to_2pi(-np.pi/2),np.pi+np.pi/2,"-pi/2 wrap to 2 pi angle error" )
    self.assertEqual(a.wrap_to_2pi(0),0,"0 wrap to 2 pi angle error" )
    
    self.assertEqual(a.wrap_angles3gpp(np.pi/2, np.pi/2),(np.pi/2,np.pi/2),"pi/2,pi/2 wrap angles 3gpp error" )
    self.assertEqual(a.wrap_angles3gpp(np.pi, np.pi),(np.pi,np.pi),"pi,pi wrap angles 3gpp error" )
    self.assertEqual(a.wrap_angles3gpp(-np.pi, -np.pi),(np.pi,np.pi),"-pi,-pi wrap angles 3gpp error" )
    self.assertEqual(a.wrap_angles3gpp(-np.pi/2, -np.pi/2),(np.pi+np.pi/2,np.pi/2),"-pi/2,-pi/2 wrap angles 3gpp error" )
  

if __name__ == "__main__":
    
    test = AnglesTest()
    test.test_constructor()
    test.test_get_angles_vector()
    test.test_wrap_angles()