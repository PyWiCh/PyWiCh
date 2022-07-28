#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module tests the scenarios module.

@author: pablobelzarena
"""

# qa_scenarios.py : unittest for scenarios.py
import unittest
import src.scenarios as sc
import numpy as np
class ScenariosTest(unittest.TestCase):
  """Unitest class for testing scenarios."""  
  def test_Friis(self):
    """ This method tests the Friis loss model.

    """ 
    fcGHz = 1
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 10
    BS_pos = [0,0,0]
    Ptx_db = 10
    scf=  sc.ScenarioFriis(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db)
    self.assertAlmostEqual(scf.get_loss_los(1), 32.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_los(1000), 92.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1), 32.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1000), 92.44177,places = 5)

    fcGHz = 10
    scf=  sc.ScenarioFriis(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db)
    self.assertAlmostEqual(scf.get_loss_los(1), 52.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_los(1000), 112.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1), 52.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1000), 112.44177,places = 5)

  def test_extendedFriis(self):
    """ This method tests the ScenarioSimpleLOS loss model.

    """ 
    fcGHz = 1
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 10
    BS_pos = [0,0,0]
    Ptx_db = 10
    order = 2
    scf=  sc.ScenarioSimpleLOS(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db, order) 
    self.assertAlmostEqual(scf.get_loss_los(1), 32.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_los(1000), 92.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1), 32.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1000), 92.44177,places = 5)

    fcGHz = 10
    scf=  sc.ScenarioFriis(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,BS_pos, Ptx_db)
    self.assertAlmostEqual(scf.get_loss_los(1), 52.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_los(1000), 112.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1), 52.44177,places = 5)
    self.assertAlmostEqual(scf.get_loss_nlos(1000), 112.44177,places = 5)

  def test_3gpp_indoor(self):
    """ This method tests the 3gpp Indoor model.
    
    The method run 2000 iterations. In each iteration cmputes the LSP of
    the 3gpp indoor model for LOS condition for one point of the grid. Assert that the mean and standard
    deviation of the LSP parmeters are all close to the average and std of the
    iterations results. The result take a few minutes. As the results is in average some times the assert can failure.
    TODO: rewrite in general for any 3gpp scenario and LOS condition.

    """ 
    fcGHz = 1
    posx_min = -100
    posx_max = 100
    posy_min = -100
    posy_max = 100
    grid_number = 50
    BS_pos = [0,0,2]
    Ptx_db = 10
    MS_pos = [1,1,2]
    LOS = 1
    scf=  sc.Scenario3GPPInDoor(fcGHz,posx_min,posx_max, posy_min, posy_max, grid_number, BS_pos, Ptx_db,True,LOS) 
    iterations = 2000
    lsp_acum = np.zeros((iterations,7))
   
    for i in range(iterations):
      pos = np.array([np.random.uniform(posx_min,posx_max),np.random.uniform(posy_min,posy_max),2])
      lsp_acum[i] = scf.generate_correlated_LSP_vector(pos,0)
    msg = "Mean of LSP params"
    msg = msg + str(np.mean(lsp_acum,axis=0))
    msg = msg + " compare to: "+ str([0,7,-0.01 * np.log10 (1 +fcGHz) - 7.692,1.6,-0.19 * np.log10 (1 +fcGHz) + 1.781, -1.43 * np.log10 (1 + fcGHz) + 2.228,-0.26 * np.log10 (1 + fcGHz) + 1.44])
    aer = np.abs( np.array([0.0,7,-0.01 * np.log10 (1 +fcGHz) - 7.692,1.6,-0.19 * np.log10 (1 +fcGHz) + 1.781, -1.43 * np.log10 (1 + fcGHz) + 2.228,-0.26 * np.log10 (1 + fcGHz) + 1.44]) - np.mean(lsp_acum,axis=0 ))
    msg  = msg + " absolute error: " + str(aer)
    self.assertTrue(np.allclose([0.0,7,-0.01 * np.log10 (1 +fcGHz) - 7.692,1.6,-0.19 * np.log10 (1 +fcGHz) + 1.781, -1.43 * np.log10 (1 + fcGHz) + 2.228,-0.26 * np.log10 (1 + fcGHz) + 1.44],np.mean(lsp_acum,axis=0) ,rtol=2e-01, atol=2e-01),msg)

    msg = " Standard(desviation of LSP params "
    msg = msg + str(np.std(lsp_acum,axis=0))
    msg = msg + " compare to "+ str([3,4,0.18,0.18,0.12 * np.log10 (1 + fcGHz) + 0.119,0.13 * np.log10 (1 + fcGHz) + 0.30,-0.04 * np.log10 (1 + fcGHz) + 0.264])
    self.assertTrue(np.allclose(np.std(lsp_acum,axis=0), [3,4,0.18,0.18,0.12 * np.log10 (1 + fcGHz) + 0.119,0.13 * np.log10 (1 + fcGHz) + 0.30,-0.04 * np.log10 (1 + fcGHz) + 0.264],rtol=1e-01, atol=1e-01),msg)


if __name__ == "__main__":
    
    test = ScenariosTest()
    test.test_Friis()
    test.test_extendedFriis()
    test.test_3gpp_indoor()
 