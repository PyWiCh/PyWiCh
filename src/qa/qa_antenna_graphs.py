#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the graph_antennas module.

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

import antennas as antennas
import graph_antennas as gra
class AntennasGraphaTest():
  """This class tests graph_antennas."""  

  def test_all_graph(self):
    """ This method tests the main functions of the graph_antennas module
    
    It is not a Unitest module, it only plots the main diagrams.
    """ 
    antenna_element = antennas.Antenna3gpp3D(8)
    ant  = antennas.AntennaArray3gpp(0.5, 0.5, 1, 16, 0, 0, 0, antenna_element, 1)
    #antenna_element = antennas.AntennaIsotropic(8)
    gra.plot_3d_pattern(antenna_element)
    gra.plot_radiation_pattterns(antenna_element)
    gra.plot_array_factor_phi(ant,0,0)
    gra.plot_pattern_array_factor_product(ant,0,0)
    gra.plot_3d_pattern_array_factor_product(ant,0,0)

clgrtest = AntennasGraphaTest()
clgrtest.test_all_graph()