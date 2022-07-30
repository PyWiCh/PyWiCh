#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the antennas module.

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

import unittest
import antennas as antennas
import angles as angles
import numpy as np

class AntennasTest(unittest.TestCase):
  """Unitest class for testing antennas."""  
  
  def test_antenna_constructor(self):
    """ This method tests the constructor of Antenna class and the get_gaindb method.

    """ 
    a  = antennas.Antenna()
    self.assertEqual(a.gaindb,1 )
    aux = angles.Angles(0, 0)
    self.assertEqual(a.get_gaindb(aux),1 )
    
  def test_antenna_isotropic(self):
    """ This method tests the AntennaIsotropic class.
    
    """
    a  = antennas.AntennaIsotropic()
    self.assertEqual(a.gaindb,1 )
    a.gaindb = 8
    aux = angles.Angles(0, 0)
    self.assertEqual(a.get_gaindb(aux),8,"antenna isotropic gaindb" )
    
  def test_antenna3gpp3D(self):
    """ This method tests the Antenna3gpp3D class.
    
    """

    a  = antennas.Antenna3gpp3D(8)
    self.assertEqual(a.maxgaindb,8,"max gaindb antenna 3gpp 3D" )
    self.assertEqual(a.A_max,30,"A_max antenna 3gpp 3D" )
    self.assertEqual(a.SLA_v,30,"SLA_v antenna 3gpp 3D" )
    self.assertEqual(a.beamwidth,65,"beamwidth antenna 3gpp 3D" )
    aux = angles.Angles(0, np.pi/2)
    self.assertEqual(a.get_gaindb(aux),8)
    aux = angles.Angles(65/180*np.pi, 155/180*np.pi)
    self.assertEqual(a.get_gaindb(aux),-16)
    aux = angles.Angles(np.pi, np.pi)
    self.assertEqual(a.get_gaindb(aux),-22)
    aux = angles.Angles(-np.pi, np.pi)
    self.assertEqual(int(a.get_gaindb(aux)),-22)  
    aux = angles.Angles(np.pi/4, np.pi/2)
    self.assertAlmostEqual(a.get_gaindb(aux), 2.2485,places=4)

    
  def test_antenna_array_3gpp(self):
    """ This method tests the AntennaArray3gpp class.
    
    """
    antenna_element = antennas.Antenna3gpp3D(8)
    a  = antennas.AntennaArray3gpp(0.5, 0.5, 2, 2, 0, 0, 0, antenna_element, 2)
    self.assertEqual(a.antenna_element.maxgaindb,8,"antenna array max gaindb of 3gpp antenna element" )
    self.assertEqual(a.get_number_of_elements(),8,"" )
    antenna_element = antennas.AntennaIsotropic(8)
    a  = antennas.AntennaArray3gpp(0.5, 0.5, 2, 2, 0, 0, 0, antenna_element, 1)
    self.assertEqual(a.antenna_element.gaindb,8,"antenna array gaindb of isotropic antenna element" )
    self.assertEqual(a.get_number_of_elements(),4,"" )
    aux = angles.Angles(0, np.pi/2)
    self.assertEqual(a.get_element_field_pattern(aux,0),(0,np.sqrt(np.power(10, 8 / 10) )))
    antenna_element = antennas.Antenna3gpp3D(8)
    a  = antennas.AntennaArray3gpp(0.5, 0.5, 2, 2, 0, 0, 0, antenna_element, 1)
    aux = angles.Angles(0, np.pi/2)
    self.assertEqual(a.get_element_field_pattern(aux,0),(0,np.sqrt(np.power(10, 8 / 10) )))
    aux = angles.Angles(np.pi, np.pi)
    self.assertEqual(a.get_element_field_pattern(aux,0),(0,np.sqrt(np.power(10, -22 / 10) )))   
    a  = antennas.AntennaArray3gpp(0.5, 0.5, 2, 2, np.pi/4, 0, 0, antenna_element, 1)
    aux = angles.Angles(np.pi/4, np.pi/2)
    self.assertEqual(a.get_element_field_pattern(aux,0),(0,np.sqrt(np.power(10, 8 / 10) )))
    aux = angles.Angles(np.pi+np.pi/4, np.pi)
    self.assertEqual(a.get_element_field_pattern(aux,0),(0,np.sqrt(np.power(10, -22 / 10) )))
    
if __name__ == "__main__":
    
    test = AntennasTest()
    test.test_antenna_constructor()
    test.test_antenna_isotropic()
    test.test_antenna3gpp3D()
    test.test_antenna_array_3gpp()
