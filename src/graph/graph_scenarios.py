#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module plots different scenario properties.

@author: pablo belzarena
"""


import numpy as np
import matplotlib.pyplot as plt
import scenarios as sc
import angles as angles
from pylab import rcParams
#import seaborn as sns
from matplotlib import cm  




rcParams['figure.figsize'] = 11, 6
#sns.set_theme(style = "whitegrid")

def graph_params_scenario(X,Y,lsp,title,fc,sc_name):  
  """ This method plots the values of one LSP parameter in the scenario grid .
  
  @type X: 2D array with the shape of lsp.
  @param X: 2D array created via numpy.meshgrid.
  @type Y: 2D array with the shape of lsp.
  @param Y: 2D array created via numpy.meshgrid.
  @type lsp: 2D srray.
  @param lsp: A 2D array with the values of the lsp parameter in each point of the grid.
  @type fc: float.
  @param fc: The scenario frequency in GHz.
  @type sc_name: string
  @param sc_name: The name of the scenario.
  @type title: string.
  @param title: The title of the plot.

  """
  lsp_1D = lsp.reshape(lsp.size)
  X2 = np.sort(lsp_1D)
  avg = np.average(lsp_1D)
  std = np.std(lsp_1D)
  subtitle = " Scenario: " + sc_name+ ", fc (GHz): " + str(fc) +" \n avg: " +  "{:.2f}".format(avg) +", std: "+ "{:.2f}".format(std)

  F2 = np.array(range(lsp.size))/float(lsp.size)
  plt.plot(X2, F2)
  plt.xlabel(title)
  plt.ylabel(" CDF " )
  title_ini = "CDF of "
  plt.title(title_ini+title+subtitle)
  plt.show()
  #lsp = 10**lsp
  #lsp[lsp > 104] = 104
  plot2 = plt.figure()
  CS = plt.contourf(X,Y,lsp)
  cbar = plot2.colorbar(CS)
  plt.xlabel("x in m" )
  plt.ylabel("y in m" )
  title_ini = "Spatial Variation of "
  plt.title(title_ini+title+subtitle)
  plt.show(block=False)



def graph_multiple_pathlos(x,cname,pls,tit):
  """ This method plots the pathlosses of a set of scenarios as function of the sitance between two devices.

  @type x: array.
  @param x: An array with the distances of the Tx where the pathloss function is evaluated.
  @type cname: list of ints.
  @param cname: A list with the indices of the scenarios to be plotted.
  @type pls: 2D array.
  @param pls: An array with the pathlosses of each scenario.
  @type tit: array.
  @param tit: An array with the titles of the plot of each scenario.
  """ 
  plt.plot(x,pls[0],label = tit[0])
  plt.plot(x,pls[1],label = tit[1])
  for i in cname:
      plt.plot(x,pls[2*i+2],label = tit[2*i+2])
      plt.plot(x,pls[2*i+3],label = tit[2*i+3])
  plt.xlabel("distance m")
  plt.ylabel("Pathloss db")
  plt.title("Pathloss ")
  plt.legend()
  plt.show(block=False)

def graph_pathloss(cname,fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db):
  """ This method computes and plots the pathloss of a set of scenarios as function of the distance between two devices. It assumes
  that the MS height is 2m.

  @type cname: list of ints.
  @param cname: A list with the indices of the scenarios to be plotted.
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
  """    
  tit =["Friis", "LOS grade 4", "3gpp Indoor LOS", "3gpp Indoor NLOS", "3gpp UMA LOS", "3gpp UMA NLOS","3gpp UMI LOS", "3gpp UMI NLOS"]
  max_d = int(posx_max - bspos[0])
  n_points = 200
  step= max_d/n_points
  pl_3gppIndoorNlos = np.zeros(n_points) 
  pl_3gppIndoorLos = np.zeros(n_points)         
  pl_friis = np.zeros(n_points)
  pl_scLOS = np.zeros(n_points) 
  pl_3gppUmaLos = np.zeros(n_points)
  pl_3gppUmaNlos = np.zeros(n_points) 
  pl_3gppUmiLos = np.zeros(n_points) 
  pl_3gppUmiNlos = np.zeros(n_points) 
  
  path_losses = np.zeros((len(tit),n_points))
  order = 2
  scFriis = sc.ScenarioSimpleLossModel(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,order)
  order = 4
  scLOS = sc.ScenarioSimpleLossModel(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db,order)
  sc3gppIndoorLos =  sc.Scenario3GPPInDoor(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  sc3gppIndoorNLos =  sc.Scenario3GPPInDoor(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  
  sc3gppUmaLos =  sc.Scenario3GPPUma(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  sc3gppUmaNLos =  sc.Scenario3GPPUma(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  
  sc3gppUmiLos =  sc.Scenario3GPPUmi(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  sc3gppUmiNLos =  sc.Scenario3GPPUmi(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number,bspos,Ptx_db)
  
  
  x = np.arange(0,max_d,step)
  i = 0
  for d in x:
      pl_friis[i] = scFriis.get_loss_los(d)
      pl_scLOS[i] = scLOS.get_loss_los(d)
      pl_3gppIndoorLos[i] = sc3gppIndoorLos.get_loss_los(d) 
      pl_3gppUmaLos[i] = sc3gppUmaLos.get_loss_los(d,2) 
      pl_3gppUmiLos[i] = sc3gppUmiLos.get_loss_los(d,2) 

      pl_3gppIndoorNlos[i] = sc3gppIndoorNLos.get_loss_nlos(d) 
      pl_3gppUmaNlos[i] = sc3gppUmaNLos.get_loss_nlos(d,2)           
      pl_3gppUmiNlos[i] = sc3gppUmiNLos.get_loss_nlos(d,2)           
      i=i+1
  path_losses[0] = pl_friis
  path_losses[1] = pl_scLOS
  for i in cname:
     if i == 0:
         path_losses[2] = pl_3gppIndoorLos
         path_losses[3] = pl_3gppIndoorNlos
     if i == 1:
         path_losses[4] = pl_3gppUmaLos
         path_losses[5] = pl_3gppUmaNlos
     if i == 2:
         path_losses[6] = pl_3gppUmiLos
         path_losses[7] = pl_3gppUmiNlos
  graph_multiple_pathlos(x,cname,path_losses,tit)
   
def graph_Txpower_map(antenna_BS,scenario,shadow,positions,phi_add=0,theta_add=0):
  """ This method plots the power recieved by a device placed in each point of the scenario.It also plots the MS path in the scenario.
  
  @type antenna_BS: Class Antenna.
  @param antenna_BS: The antenna of the Base Station. 
  @type scenario: Class Scenario.
  @param scenario: The scenario used to compute the pathloss and shadowing. 
  @type shadow: Boolean.
  @param shadow: If shadowing is enabled or not. 
  @type positions: Array. 
  @param positions: An array with the positions of each MS. Rows MS, columns points in the path. 
  @type phi_add: float
  @param phi_add: The aditional azimuth rotation angle for  BS beamforming . 
  @type theta_add: float.
  @param theta_add: The aditional inclination rotation angle for BS beamforming . 
  """
  antennaTx_type = type(antenna_BS.antenna_element).__name__
  antennaTx_elements = antenna_BS.get_number_of_elements()
  Bs_pos = scenario.BS_pos
  Ptx = scenario.Ptx_db #db
  fc = scenario.fcGHz # 1 GHz
  sc_name = scenario.name
  subtitle = "\n Scenario: "+ sc_name+ ", frequency(GHz): "+ str(fc)+ ", PTx (dbm): " +str(Ptx) + " \n Antenna Tx: "+ antennaTx_type + ", number of elements: "+ str(antennaTx_elements)
  dx_min = scenario.posx_min
  dy_min = scenario.posy_min
  dx_max = scenario.posx_max
  dy_max = scenario.posy_max
  grid = scenario.grid_number
  x_vec = np.zeros(grid+1)
  y_vec = np.zeros(grid+1)
  power_radiated = np.zeros((grid+1,grid+1))
  ptot = np.zeros((grid+1,grid+1))
  aux = angles.Angles(0,0)
  i = 0
  for y in np.linspace(dy_min,dy_max,grid+1):
    y_vec[i] = y
    j = 0 
    for x in np.linspace(dx_min,dx_max,grid+1):
      x_vec[j] = x
      aux.get_angles_vectors([x,y,0],Bs_pos)
      p = aux.get_azimuth()
      t = np.pi/2
      angle = angles.Angles(p,np.pi/2)
      anglePrime = antenna_BS.GCS_to_LCS(angle)
      Gt = pow(10,antenna_BS.antenna_element.get_gaindb(anglePrime)/10)*np.abs(np.sum(antenna_BS.compute_phase_steering(p,t,phi_add,theta_add)))**2
      d = np.sqrt((x-Bs_pos[0])**2 + (y-Bs_pos[1])**2)
      if scenario.is_los_cond([x,y,0]):
          pl = scenario.get_loss_los(d,2)
      else:
          pl = scenario.get_loss_nlos(d,2)  
      if shadow:
          pl = pl+scenario.get_shadowing_db([x,y,0],0)
      power_radiated[i][j] =Ptx -pl+10*np.log10(Gt)
      if(power_radiated[i][j] < -200):
        power_radiated[i][j] = -200
      ptot[i][j] = power_radiated[i][j]
      j = j+1
    i = i + 1
  X, Y = np.meshgrid(x_vec, y_vec)
  fig1, ax = plt.subplots(constrained_layout=True)

  cs = ax.contourf (X,Y,ptot,levels = 50,cmap=cm.coolwarm)#,locator = ticker.LogLocator())#locator = ticker.LogLocator())#extent=[2*dx_min,2*dx_max,dy_min,dy_max],
  cbar = plt.colorbar(cs)
  ax.set_title('Tx Power Map in dbm' + subtitle)
  ax.set_xlabel('x in m')
  ax.set_ylabel('y in m')
  nMS  = positions.shape[0]
  for ms in range(nMS):
      ax.plot(positions[ms][:,0],positions[ms][:,1],"*")
  plt.show(block=False)

def graph_H_f(s,prb,point,subtitle):
    """ This method plots the eigenvalues and the condition number of Fourir Transform of the channel matrix in one prb and one point in the MS path.
 
    @type s: Array
    @param s: An array with the singular values of the channel matrix.
    @type prb: int.
    @param prb: the number of the prb.
    @type point: int.
    @param point: the number of the point in the MS path.
    @type subtitle: string
    @param subtitle: The subtitle of the plot.
    """ 
    plot2 = plt.figure()
    my_colors = list('rgbkymc')
    plt.bar(range(s.size),s**2,color = my_colors)
    plt.title('Eigenvalues and condition number for prb '+str(prb)+ ' and point number '+ str(point)+ subtitle  )
    plt.grid(color='r', axis = 'y',linestyle='-', linewidth=2,which = 'major')
    cond = 10 *np.log10(max(s)/min(s))
    plt.ylabel('power gain of \n each eigenvalue')
    plt.text(s.size*0.55, max(s**2)*3/4, 'Condition number \n of H: '+str(int(cond))+ 'db', fontsize = 14)
    plt.show(block=False)

def graph_ssp(title,ylabel,n,param):
    """ This method plots one of the short scale parameters (ssp) for each cluster.
    
    @type title: string
    @param title: The title of the plot.
    @type ylabel: string
    @param ylabel: The name of the ssp parameter to plot.
    @type n: int.
    @param n: number of clusters.
    @type param: array.
    @param param: An array with ssp value for each cluster.    
    """ 
    plot2 = plt.figure()
    plt.title(title)
    my_colors = list('rgbkymc')
    plt.bar(range(n),param,color = my_colors)
    plt.grid(color='r', axis = 'y',linestyle='-', linewidth=2,which = 'major')

    plt.xlabel(" cluster number" )
    plt.ylabel(ylabel )
    plt.show(block=False)
    
def graph_path_performance(los,positions,snr,H,n_MS,n_BS,prb,snr_pl,sp_eff,subtitle,linear_losses,snr_pl_shadow,times):
    """This method plots the average snr, the average spectral efficiency and the module of the channel matrix elements for one selected prb.

    @type los: 1D Array
    @param los: The LOS condition for each point of the path. 
    @type snr: 1D Array
    @param snr: The  average snr (average over the prbs) for each point of the path. 
    @type H: 4D Array
    @param H: The Fourier Transform of the channel matrix for each prb, each TX-RX antenna pair and for each point of the path. 
    @type linear_losses: 2D Array
    @param linear_losses: The path loss in linear scale for each point of the path. 
    @type positions: 1D Array.
    @param positions: The distance for each point in the path to the BS. 
    @type snr_pl: 1D Array.
    @param snr_pl: The average snr of each point of the path using only the path losses and not using shadowing and fading. 
    @type sp_eff: 1D Array.
    @param sp_eff: The average spectral efficiency (bits/s/Hz) for each point of the path. 
    @type snr_pl_shadow: 1D Array.
    @param snr_pl_shadow: The average snr of each point of the path using only the path losses and shadowing and not using the fast fading. 
    @type n_MS: int.
    @param n_MS: The number of elements of the MS antenna array.
    @type n_BS: int.
    @param n_BS: The number of elements of the BS antenna array.
    @type prb: int
    @param prb: The number of the prb.
    @type subtitle: string
    @param subtitle: The subtitle of the plot.
    """ 
    plot2 = plt.figure()
    plt.title(" Average snr along the path "  +subtitle )
    plt.plot(times,snr)
    plt.xlabel(" Simulation time in s" )
    plt.ylabel("Average SNR db" )    
    plt.show(block=False)                  

    plot2 = plt.figure()
    plt.title(" Distance from the BS "  +subtitle )
    plt.plot(times,positions)
    plt.xlabel(" Simulation time in s" )
    plt.ylabel("distance from BS in m" )    
    plt.show(block=False)                  



    plot2 = plt.figure()
    plt.title(" Components of Average snr along the path "  +subtitle )
    plt.plot(times,snr-snr_pl_shadow,label = " Fast fading" )
    plt.plot(times,snr_pl,label= "Path loss" )
    plt.plot(times,snr_pl_shadow-snr_pl,label =" Shadowing" )
    #plt.ylim([0,50])
    plt.xlabel(" Simulation time in s" )
    plt.ylabel("Average SNR db" )    
    plt.legend()
    plt.show(block=False)                  
 
    plot2 = plt.figure()
    i = 0
    snr_pl_los = []
    snr_pl_nlos = []
    sp_eff_los = []
    sp_eff_nlos = []
    for ilos in los:
        if ilos == 1:
            snr_pl_los.append(snr_pl[i])
            sp_eff_los.append(sp_eff[i])
        else: 
            snr_pl_nlos.append(snr_pl[i])
            sp_eff_nlos.append(sp_eff[i])
        i = i + 1
    if len(snr_pl_los) > 0: 
        sp_eff_avg = exp_mavg(0.05, np.flip(np.array(sp_eff_los)))
        plt.title(" Average spectral efficiency along the path \n as function of snr "+subtitle )
        plt.plot(np.flip(np.array(snr_pl_los)),np.flip(np.array(sp_eff_los)),'*',label=" LOS")
        plt.plot(np.flip(np.array(snr_pl_los)),sp_eff_avg,label=" moving average LOS")
    if len(snr_pl_nlos) > 0:    
        sp_eff_avg = exp_mavg(0.05, np.flip(np.array(sp_eff_nlos)))
        plt.title(" Average spectral efficiency along the path \n as function of snr "+subtitle )
        plt.plot(np.flip(np.array(snr_pl_nlos)),np.flip(np.array(sp_eff_nlos)),'*',label=" NLOS")
        plt.plot(np.flip(np.array(snr_pl_nlos)),sp_eff_avg,label=" moving average NLOS")

    plt.legend(loc="upper left")
    #plt.xticks(range(len(snr_pl)),snr_pl)
    plt.xlabel(" snr" )
    plt.ylabel("Spectral efficiency in bits/s/Hz" )    
    plt.show(block=False)      

   
    plot2 = plt.figure()
    plt.title(" Elements of H matrix for prb: "+str(prb)+subtitle  )
    for i in range(n_MS):
        for j in range(n_BS):                    
            plt.plot(times,np.abs(H[:,prb,i,j])**2)
    plt.xlabel(" Simulation time in s" )
    plt.ylabel("abs(H[i][j])**2" )    
    plt.show(block=False)   

def graph_performance(point,rxpsd,n_MS,n_BS,H,subtitle):
    """This method plots the recieved power spectral density and the module of the elemnts of the channel matrix for each prb in one point of the MS path. 

    @type rxpsd: 1D Array
    @param rxpsd: The  recieved power spectral density for each prb.
    @type H: 4D Array
    @param H: The Fourier Transform of the channel matrix for each prb, each TX-RX antenna pair and for each point of the path. 
    @type n_MS: int.
    @param n_MS: The number of elements of the MS antenna array.
    @type n_BS: int.
    @param n_BS: The number of elements of the BS antenna array.
    @type point: int
    @param point: The number of the point in the MS path.
    @type subtitle: string
    @param subtitle: The subtitle of the plot.
    """ 

    plot2 = plt.figure()
    plt.title(" Rx power per prb in the point of the path number "+str(point)+subtitle )
    #for i in range(self.points_in_paths):
    plt.plot(10*np.log10(rxpsd[point]*180000)+30)
    plt.xlabel(" prb number" )
    plt.ylabel("Rx power dbm " )    
    plt.show(block=False)
    
    plot2 = plt.figure()
    plt.title(" Elements of H matrix for each prb in the point of the path number "+str(point)+subtitle  )
    for i in range(n_MS):
        for j in range(n_BS):                    
            plt.plot(np.abs(H[point,:,i,j])**2)
            plt.xlabel(" prb " )
            plt.ylabel("abs(H[i][j])**2" )    
    plt.show(block=False)   


def exp_mavg(alpha,x):
    """ This auxiliary method computes the exponential moving average of the vector x with parameter alpha.
    
    @type alpha: float between 0 and 1.
    @param alpha: the parameter of the exponential moving average.
    @type x: array.
    @param x: the input array.
    @return: dato*alpha + prom*(1-alpha)
    """
    out = np.zeros(len(x))
    i = 0
    prom = x[0]
    for dato in x:
        prom = dato*alpha + prom*(1-alpha)
        out[i] = prom
        i = i+1
    return out
    # Pr = pow(10,30/10)
    # power_rec = np.zeros((int((dx_max -dx_min)/1),int((dy_max -dy_min)/1)))
    # Ms_pos =[30,30,0]
    # aux = angles.Angles(0,0)
    # i = 0
    # for y in range(dy_min,dy_max,1):
    #   y_vec[i] = y
    #   j = 0 
    #   for x in range(dx_min,dx_max,1):
    #     x_vec[j] = x
    #     aux.get_angles_vectors([x,y,0],Ms_pos)
    #     p = aux.GetAzimuth()
    #     t = np.pi/2
    #     phiPrime = np.angle(complex(np.cos(downtiltAngle)*np.sin(t)*np.cos(p-bearingAngle1) - np.sin(downtiltAngle)*np.cos(t), np.sin(p-bearingAngle1)*np.sin(t)))
    
    #     angle = angles.Angles(phiPrime,t)
    #     Gr = pow(10,antenna_MS.antenna_element.get_gaindb(angle)/10)*np.abs(np.sum(antenna_BS.compute_phase_steering(p,t,0,0)))**2
    #     d = (x-Ms_pos[0])**2 + (y-Ms_pos[1])**2
    #     if scenario.LOS:
    #         pl = scenario.get_loss_los(d)
    #     else:
    #         pl = scenario.get_loss_nlos(d)
    #      power_rec[i][j] = pow(10,-pl)*Pr*Gr
    #     if(power_rec[i][j] < pow(10,-20)):
    #         power_rec[i][j] = pow(10,-20)
    #     ptot[i][j] = power_radiated[i][j] + power_rec[i][j]
    
    #     j = j+1
    #   i = i + 1
    
    # x=61
    # y=0
    # aux.get_angles_vectors(Bs.pos,Ms.pos)
    # p = aux.GetAzimuth()
    # print("------------",Bs.pos,Ms.pos,p)
        
    # cs = plt.imshow(ptot)
    # #cs = plt.contourf(X,Y,Z1) #,locator = ticker.LogLocator(),cmap ="bone")  
    # cbar = plt.colorbar(cs)
    # plt.plot(-0.5, 0.3, 'ro')
    
    # plt.show() 
    

               
           # plot2 = plt.figure()
           # plt.suptitle('Cluster Azimuth angle of departure')
           # plt.bar(range(n),self.fading.phiAOD)
           # plt.xlabel(" cluster number" )
           # plt.ylabel(" Azimuth AOD in degrees" )
           # plt.show(block=False)
           
           # plot2 = plt.figure()
           # plt.suptitle('Cluster inclination angle of arrival')
           # plt.bar(range(n),self.fading.thetaAOA)
           # plt.xlabel(" cluster number" )
           # plt.ylabel(" Inclination AOA in degrees" )
           # plt.show(block=False)
           
           # plot2 = plt.figure()
           # plt.suptitle('Cluster inclination angle of departure')
           # plt.bar(range(n),self.fading.thetaAOD)
           # plt.xlabel(" cluster number" )
           # plt.ylabel(" Inclination AOD in degrees" )
           # plt.show(block=False)

           #phiAOA_m_rad = np.loadtxt('./data/PHI_AOA_rays_'+str(i)+'.csv', delimiter=',')
           # if self.fading.scenario.LOS:
           #      cluster = np.argmax(self.fading.P_LOS)
           # else:
           #     cluster = np.argmax(self.fading.P)
           # plot2 = plt.figure()
           # plt.suptitle('Rays azimuth angle of arrival for clsuter with max power')
           # plt.bar(range(phiAOA_m_rad[cluster].size),phiAOA_m_rad[cluster]*180/np.pi)
           # plt.xlabel(" ray number" )
           # plt.ylabel(" Azimuth AOA in degrees" )
           # plt.show(block=False)
           
           # plot2 = plt.figure()
           # plt.suptitle('Rays azimuth angle of departure for clsuter with max power')
           # plt.bar(range(self.fading.scenario.raysPerCluster),self.fading.phiAOD_m_rad[cluster]*180/np.pi)
           # plt.xlabel(" ray number" )
           # plt.ylabel(" Azimuth AOD in degrees" )
           # plt.show(block=False)
           
           # plot2 = plt.figure()
           # plt.suptitle('Rays inclination angle of arrival for clsuter with max power')
           # plt.bar(range(self.fading.scenario.raysPerCluster),self.fading.thetaAOA_m_rad[cluster]*180/np.pi)
           # plt.xlabel(" ray number" )
           # plt.ylabel(" Inclination AOA in degrees" )
           # plt.show(block=False)
           
           # plot2 = plt.figure()
           # plt.suptitle('Rays ainclination angle of departure for clsuter with max power')
           # plt.bar(range(self.fading.scenario.raysPerCluster),self.fading.thetaAOD_m_rad[cluster]*180/np.pi)
           # plt.xlabel(" ray number" )
           # plt.ylabel(" Inclination AOD in degrees" )
           # plt.show(block=False)
           #else:
           #self.msg_user("Error message", " Scenario or AntennaTx or AntennaRx are not configured.\n Please configure all of them. " )

    
# antenna_elementBS = ant.Antenna3gpp3D(8)
# ant_BS  = ant.AntennaArray3gpp(0.5, 0.5, 1, 8, np.pi/2, 0, 0, antenna_elementBS, 1)
# antenna_elementMS = ant.AntennaIsotropic(8)
# ant_MS  = ant.AntennaArray3gpp(0.5, 0.5, 1, 1, 0, 0, 0, antenna_elementMS, 1)
# fcGHz = 1
# posx_min = 1
# posx_max = 20
# posy_min = 1
# posy_max = 20
# grid_number = 20
# max_d = 50
# step= 1

# sc3gppLos =  sc.Scenario3GPPInDoor(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number)
# sc3gppLos.set_los(True)
# sc3gppLos.set_params()
# sc3gppNLos =  sc.Scenario3GPPInDoor(fcGHz,posx_min,posx_max,posy_min,posy_max,grid_number)
# sc3gppNLos.set_los(False)
# sc3gppNLos.set_params()

# scFriis = sc.ScenarioFriis(fcGHz)   
# graph_power_map(ant_BS,ant_MS,sc3gppLos)
