#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""This module has auxiliariy functions to plot different antenna characteristics.

@author: pablo belzarena
"""
import numpy as np
import matplotlib.pyplot as plt
import angles as angles
from pylab import rcParams
import matplotlib as mpl

#import seaborn as sns

rcParams['figure.figsize'] = 11, 6
#sns.set_theme(style = "whitegrid")


def plot_3d_pattern(antenna):
    """This method plots 3D antenna power radiation pattern.
    
    The pattern is obtained from get_gaindb Antenna method. The method plots in
    3D (pow(10,antenna.get_gaindb(angle)/10)). The angle is sample each 5 degrees. 
    The azimuth varies between -180 and 180 degrees and the inclination varies
    between 0 and 180 degrees.
    @type antenna: Class Antenna .
    @param antenna: The antenna whose radiation pattern is plotted.
    """
    antype = type(antenna).__name__
    if antype == "Antenna3gpp3D" :
        max_gain = antenna.maxgaindb
        SLA_v = antenna.SLA_v
        A_max = antenna.A_max
        beam = antenna.beamwidth
        subtitle = " \n Antenna element: "+antype+ ", max gain: "+str(+max_gain)+ ", max vertical attenuation: "+str(SLA_v)+"\n max horizontal attenuation: "+ str(A_max)+ ", beamwidth: "+ str(beam)
    if antype == "AntennaIsotropic" :
        max_gain = antenna.gaindb
        subtitle = " \n Antenna element: "+antype+ ", max gain: "+str(+max_gain)
    vals_theta = np.arange(0,181,5)
    vals_phi = np.arange(-180,181,5)
    THETA = np.deg2rad(vals_theta)
    PHI = np.deg2rad(vals_phi)
    R = np.zeros((PHI.size,THETA.size))
    X = np.zeros((PHI.size,THETA.size))
    Y = np.zeros((PHI.size,THETA.size))
    Z = np.zeros((PHI.size,THETA.size))
    i= 0
    for p in PHI:
        j =0
        for t in THETA:
            angle = angles.Angles(p,t)
            R[i][j] =  (pow(10,antenna.get_gaindb(angle)/10))
            X[i][j] = R[i][j] * np.sin(t) * np.cos(p)
            Y[i][j] = R[i][j] * np.sin(t) * np.sin(p)
            Z[i][j] = R[i][j] * np.cos(t)
            j=j+1
        i = i+1
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.grid(True)
    ax.axis('on')
    surf = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, linewidth=0.1, antialiased=True, shade=False ,cmap='viridis', edgecolor='none')#  alpha=0.5, zorder = 0.5)#facecolors=mycol,facecolors=mycol,
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(' 3D Power radiation pattern, linear scale'+subtitle)
    ax.view_init(azim=100, elev=30)
    plt.show(block = False)
  


def plot_radiation_pattterns(antenna):
    """This method plots 2D antenna power radiation patterns. 
    
    The pattern is obtained from get_gaindb Antenna method. The method plots in
    2D (pow(10,antenna.get_gaindb(angle)/10)). The angle is sample each 1 degrees. 
    The method generate two plots. One is an horizontal cut with the inclination 
    fixed at 90 degrees and azimuth varying between -180 and 180 degrees. The other 
    is a vertical cut with the azimuth fixed in cero degrees and the inclination 
    variying between 0 and 180 degrees.
    @type antenna: Class Antenna .
    @param antenna: The antenna whose radiation pattern is plotted.
    """
    antype = type(antenna).__name__
    if antype == "Antenna3gpp3D" :
        max_gain = antenna.maxgaindb
        SLA_v = antenna.SLA_v
        A_max = antenna.A_max
        beam = antenna.beamwidth
        subtitle = " \n Antenna element: "+antype+ ", max gain: "+str(+max_gain)+ ", max vertical attenuation: "+str(SLA_v)+"\n max horizontal attenuation: "+ str(A_max)+ ", beamwidth: "+ str(beam)
    if antype == "AntennaIsotropic" :
        max_gain = antenna.gaindb
        subtitle = " \n Antenna element: "+antype+ ", max gain: "+str(+max_gain)

    fig = plt.figure()
    plt.suptitle('Power radiation pattern, linear scale' + subtitle)
    vals_theta = np.arange(0,181,1)
    vals_phi = np.arange(-180,181,1)
    THETA = np.deg2rad(vals_theta)
    PHI = np.deg2rad(vals_phi)
    R = np.zeros((THETA.size))
    i= 0
    for t in THETA:
      angle = angles.Angles(0,t)
      R[i] = antenna.get_gaindb(angle)
      i = i+1   
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.set_xlabel(" 'Vertical cut, phi = 0, theta ")
    #ax1.set_rticks([-20, -15, -10, -5, 0])
    plt.polar(THETA,pow(10,R/10))
    R = np.zeros((PHI.size))
    i= 0
    for p in PHI:
      angle = angles.Angles(p,np.pi/2)
      R[i] = antenna.get_gaindb(angle)
      i = i+1
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.set_xlabel("Horizontal cut, theta = 90, phi ")
    plt.polar(PHI,  pow(10,R/10)) 
    #ax2.set_rticks([-20, -15, -10, -5, 0])
    #ax.set_rlabel_position(45)
    plt.show(block = False)



def plot_array_factor_theta(antenna,phi_add,theta_add):
    """This method plots the vertical array factor. The azimuth is fixed in 0 degrees and
    theta varies between 0 and 180 degrees sampled each one degree.
    
    The method plots 
    2np.abs(np.sum(antenna.phase_steering(0,inclination,phi_add,theta_add))). 
    @type antenna: Class Antenna .
    @param antenna: The antenna whose pattern id plotted.
    @type phi_add: float.
    @param phi_add: the azimuth additional rotation angle.
    @type theta_add: float.
    @param theta_add: the inclination additional rotation angle.
    """
    fig = plt.figure()
    plt.suptitle('Vertical Array Factor')
    vals_theta = np.arange(0,181,1)
    THETA = np.deg2rad(vals_theta)
    R = np.zeros((THETA.size))
    i= 0
    for t in THETA:
      R[i] = np.abs(np.sum(antenna.compute_phase_steering(0,t,phi_add,theta_add)))
      i= i+1
    plt.plot(vals_theta, R) 
    plt.show(block = False)


def plot_array_factor_phi(antenna,phi_add,theta_add):
    """This method plots the horizontal array factor. The inclination fixed in 90 degrees and the azimuth 
    variying between -180 and 180 degrees, sampled each one degree.
    
    The method plots 
    2np.abs(np.sum(antenna.phase_steering(azimuth,np.pi/2,phi_add,theta_add)).     
    @type antenna: Class Antenna .
    @param antenna: The antenna whose pattern id plotted.
    @type phi_add: float.
    @param phi_add: the azimuth additional rotation angle.
    @type theta_add: float.
    @param theta_add: the inclination additional rotation angle.   
    """
    fig = plt.figure()
    plt.suptitle('Horizontal Array Factor')
    vals_phi = np.arange(-180,181,1)
    PHI = np.deg2rad(vals_phi)
    R = np.zeros((PHI.size))
    i= 0
    for p in PHI:
      R[i]= np.abs(np.sum(antenna.compute_phase_steering(p,np.pi/2,phi_add,theta_add)))
      i= i+1
    plt.plot(vals_phi,R )
    plt.show(block = False)

    
def plot_pattern_array_factor_product(antenna,phi_add,theta_add):
    """This method plots the 2D antenna power radiation pattern multiplied by the antenna array factor.
    
    The pattern is obtained from get_gaindb Antenna method. The array factor is obtained
    from phase_steering Antenna method.
    The method plots in 2d polar coordinates:
    (pow(10,antenna.get_gaindb(angle)/10))*np.abs(np.sum(antenna.phase_steering(azimuth,inclination,phi_add,theta_add)))**2. The angle is sample each 1 degrees. 
    The method generate two plots. One is an horizontal cut with the inclination 
    fixed at 90 degrees and azimuth varying between -180 and 180 degrees. The other 
    is a vertical cut with the azimuth fixed in cero degrees and the inclination 
    variying between 0 and 180 degrees.
    
    @type antenna: Class Antenna .
    @param antenna: The antenna whose pattern id plotted.
    @type phi_add: float.
    @param phi_add: the azimuth additional rotation angle.
    @type theta_add: float.
    @param theta_add: the inclination additional rotation angle.
    """
    fig = plt.figure()
    plt.suptitle('Power radiation pattern multiplied by array factor')
    vals_phi = np.arange(-180,181,1)
    PHI = np.deg2rad(vals_phi)
    R = np.zeros((PHI.size))
    A = np.zeros((PHI.size))
    i= 0
    for p in PHI:
      A[i]= np.abs(np.sum(antenna.compute_phase_steering(p,np.pi/2,phi_add,theta_add)))**2
      angle = angles.Angles(p,np.pi/2)
      R[i] = pow(10,antenna.antenna_element.get_gaindb(angle)/10)*A[i]
      i= i+1
    ax1 = fig.add_subplot(121, projection='polar')
    ax1.title.set_text('Horizontal cut, theta = 90')
    ax1.set_xlabel(" Phi in degrees")
    #ax1.set_rticks([-20, -15, -10, -5, 0])
    plt.polar(PHI, R)
    vals_theta = np.arange(0,181,1)
    THETA = np.deg2rad(vals_theta)
    R = np.zeros((THETA.size))
    A = np.zeros((THETA.size))
    i= 0
    for t in THETA:
      A[i] = np.abs(np.sum(antenna.compute_phase_steering(0,t,phi_add,theta_add)))**2
      angle = angles.Angles(0,t)
      R[i] = pow(10,antenna.antenna_element.get_gaindb(angle)/10)*A[i]
      i= i+1
    ax2 = fig.add_subplot(122, projection='polar')
    ax2.title.set_text('Vertical cut, phi = 90')
    ax2.set_xlabel(" Theta in degrees")
    #ax1.set_rticks([-20, -15, -10, -5, 0]
    plt.polar(THETA, R)
    plt.show(block = False)


def plot_3d_pattern_array_factor_product(antenna,phi_add,theta_add):
    """This method plots the 3D antenna power radiation pattern multiplied by the antenna array factor
    
    The pattern is obtained from get_gaindb Antenna method. The array factor is obtained
    from phase_steering Antenna method.
    The method plots in 3D:
    (pow(10,antenna.get_gaindb(angle)/10))*np.abs(np.sum(antenna.phase_steering(azimuth,inclination,phi_add,theta_add)))**2. The angle is sample each 1 degrees. 
    The angle is sample each 5 degrees. 
    The azimuth varies between -180 and 180 degrees and the inclination varies
    between 0 and 180 degrees.
    
    @type antenna: Class Antenna .
    @param antenna: The antenna whose pattern id plotted.
    @type phi_add: float.
    @param phi_add: the azimuth additional rotation angle.
    @type theta_add: float.
    @param theta_add: the inclination additional rotation angle.
    """
    subtitle = "\n Antenna array : "+ str(antenna.n_rows) + "x"+ str(antenna.n_cols)+ " dh: " + str(antenna.d_h)+ " dv: " +str(antenna.d_v)
    vals_theta = np.arange(0,181,5)
    vals_phi = np.arange(-180,181,5)
    THETA = np.deg2rad(vals_theta)
    PHI = np.deg2rad(vals_phi)
    R = np.zeros((PHI.size,THETA.size))
    X = np.zeros((PHI.size,THETA.size))
    Y = np.zeros((PHI.size,THETA.size))
    Z = np.zeros((PHI.size,THETA.size))
    i= 0
    for p in PHI:
        j =0
        for t in THETA:
            angle = angles.Angles(p,t)
            #if (-0.1<p<0.1 or -np.pi+0.1<p<np.pi-0.1 ) and np.pi/2-0.01<t< np.pi/2+0.01:
            R[i][j] = pow(10,antenna.antenna_element.get_gaindb(angle)/10)* np.abs(np.sum(antenna.compute_phase_steering(p,t,phi_add,theta_add)))**2
            X[i][j] = R[i][j] * np.sin(t) * np.cos(p)
            Y[i][j] = R[i][j] * np.sin(t) * np.sin(p)
            Z[i][j] = R[i][j] * np.cos(t)
            j=j+1
        i = i+1
    fig = plt.figure()
    plt.suptitle(' 3D Power radiation pattern and power radiation pattern multiplied by array factor' + subtitle)
    ax = fig.add_subplot(1,1,1, projection='3d')
    ax.grid(True)
    ax.axis('on')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0.1, color='b',edgecolors='w', shade=False , antialiased=True,label = " Antenna array" )# antialiased=True, shade=False ,,cmap = 'gnuplot',  alpha=0.5, zorder = 0.5)#facecolors=mycol,facecolors=mycol,
    surf._edgecolors2d = surf._edgecolor3d
    surf._facecolors2d = surf._facecolor3d

    i= 0
    for p in PHI:
        j =0
        for t in THETA:
            angle = angles.Angles(p,t)
            R[i][j] = pow(10,antenna.antenna_element.get_gaindb(angle)/10)# (pow(10,antenna.get_gaindb(angle)/10))
            X[i][j] = R[i][j] * np.sin(t) * np.cos(p)
            Y[i][j] = R[i][j] * np.sin(t) * np.sin(p)
            Z[i][j] = R[i][j] * np.cos(t)
            j=j+1
        i = i+1
    surf1 = ax.plot_surface(X, Y, Z,rstride=1, cstride=1, linewidth=0.1,color='r', shade=False ,edgecolors='w',  antialiased=True,label = "Antenna element" ) #,rstride=1, cstride=1, linewidth=0.1, antialiased=True, shade=False , edgecolor='none',,cmap='viridis'  facecolors=mycol,facecolors=mycol,
    surf1._edgecolors2d = surf1._edgecolor3d
    surf1._facecolors2d = surf1._facecolor3d
    ax.legend()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(azim=60, elev=30)
    plt.show(block = False)


