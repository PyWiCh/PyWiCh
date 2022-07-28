#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the antenna.

@author: pablo belzarena
"""
import tkinter as tk
import graph.graph_antennas as ga
import gui.gui_user_message as gum
import antennas as antennas

class AppAntenna():
    """ This class is the form for the configuration of the antenna. """

    def __init__(self,window,function_cbk,title,name=" ",rows=1,cols=2,dh=0.5,dv=0.5,bearing=0,downtilt=0,slant=0,phi=0,theta=0,gain_iso=8,gain_3gpp=8,SLA_v=30,A_max=30,beam=65,ant_type=1):
        """The constructor of the AppAntenna Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type name: string
        @param name: The name of the antenna.
        @type dh: float.
        @param dh: The antenna elements are uniformly spaced in the horizontal direction with a spacing of dh. In number of wavelength. 
        @type dv: float.
        @param dv: The antenna elements are uniformly spaced in the vertical direction with a spacing of dv. In number of wavelength. 
        @type cols: float .
        @param cols: The number of columns of the panel.
        @type rows: float .
        @param rows: The number of rows of the panel.
        @type bearing: float .
        @param bearing: Bearing angle of the transformation of the local coordinate system (LCS) to a global coordinate system(GCS). See section 7.1.3 of 3GPP TR 38.901
        @type downtilt: float .
        @param downtilt: Downtilt angle of the transformation of the local coordinate system to a global coordinate system. See section 7.1.3 of 3GPP TR 38.901
        @type slant: float .
        @param slant: Slant angle of the transformation of the local coordinate system to a global coordinate system. See section 7.1.3 of 3GPP TR 38.901
        @type phi:  float.
        @param phi: The azimuth additional rotation angle for beamforming.Default 0 radians.
        @type theta:  float.
        @param theta: The inclination aditional rotation angle for beamforming. Default 0 radians.
        @type gain_3gpp: float .
        @param gain_3gpp: Maximum directional gain of the 3gpp antenna model. Default value in 3GPP TR 38.901 value 8 db.
        @type A_max: float .
        @param A_max: Front-back ratio in db. Defualt value 30 db.
        @type SLA_v: float .
        @param SLA_v: Slidelobe level limit in db. Default 30 db.
        @type beam: float .
        @param beam: Beamwidth of the antenna in degrees. Default 65 degrees.   
        @type gain_iso: float .
        @param gain_iso: Maximum directional gain of the isotropic antenna model. default value 8db.
        @type ant_type: int
        @param ant_type: Antenna type. Value 1 for Isotropic, value 2 for 3gpp model.
        """
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.antenna_name = name
        """  The name of the antenna. """ 
        self.rows = rows
        """ The number of rows of the panel.  """ 
        self.cols = cols
        """ The number of columns of the panel.  """ 
        self.dh = dh
        """ The antenna elements are uniformly spaced in the horizontal direction with a spacing of dh. In number of wavelength.   """ 
        self.dv = dv
        """ The antenna elements are uniformly spaced in the vertical direction with a spacing of dv. In number of wavelength.   """ 
        self.bearing = bearing
        """ Bearing angle of the transformation of the local coordinate system (LCS) to a global coordinate system(GCS). See section 7.1.3 of 3GPP TR 38.901  """ 
        self.downtilt = downtilt
        """ Downtilt angle of the transformation of the local coordinate system (LCS) to a global coordinate system(GCS). See section 7.1.3 of 3GPP TR 38.901  """ 
        self.slant = slant
        """ Slant angle of the transformation of the local coordinate system (LCS) to a global coordinate system(GCS). See section 7.1.3 of 3GPP TR 38.901  """ 
        self.phi = phi
        """ The azimuth additional rotation angle for beamforming.Default 0 radians.  """ 
        self.theta = theta
        """ The inclination additional rotation angle for beamforming.Default 0 radians.  """ 
        self.gain_iso = gain_iso
        """ Maximum directional gain of the isotropic antenna model. default value 8db.  """ 
        self.gain_3gpp = gain_3gpp
        """  Maximum directional gain of the 3gpp antenna model. default value 8db. """ 
        self.SLA_v = SLA_v
        """ Slidelobe level limit in db. Default 30 db.  """ 
        self.A_max = A_max
        """ Front-back ratio in db. Defualt value 30 db.  """ 
        self.beam = beam
        """ Beamwidth of the antenna in degrees. Default 65 degrees.     """ 
        self.ant_type = ant_type
        """ Antenna type. Value 1 for Isotropic, value 2 for 3gpp model.  """ 
        self.var = 1
        """This variable store the type of the antenna element. """ 

        self.gui_configuration()
         
    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """         
        self.window.rowconfigure(10, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3], minsize=50, weight=1)       
        self.gui_antenna_array() 
        self.gui_array_factor()
        self.gui_antenna_element()
    
    def gui_antenna_array(self):
        """This method configures the antenna array part of the form.
        
        """                 
        frm_antenna = tk.Frame(master=self.window)
        lbl_antenna = tk.Label(master=frm_antenna, text="Antenna array specification",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_antenna.grid(row=0, column=0, sticky="w")
        
        frm_rows = tk.Frame(master=self.window)
        self.ent_rows = tk.Entry(master=frm_rows, width=12)
        """The tkinter Entry object to enter the number of rows of the antenna array. """
        lbl_rows = tk.Label(master=frm_rows, text="number of rows")
        self.ent_rows.grid(row=1, column=0, sticky="e")
        lbl_rows.grid(row=0, column=0, sticky="w")
        self.ent_rows.insert(tk.END,str(self.rows))
        
        frm_cols = tk.Frame(master=self.window)
        self.ent_cols = tk.Entry(master=frm_cols, width=12)
        """The tkinter Entry object to enter the number of columns of the antenna array. """
        lbl_cols = tk.Label(master=frm_cols, text="number of columns")
        self.ent_cols.grid(row=1, column=0, sticky="e")
        lbl_cols.grid(row=0, column=0, sticky="w")
        self.ent_cols.insert(tk.END,str(self.cols))
        
        frm_dh = tk.Frame(master=self.window)
        self.ent_dh = tk.Entry(master=frm_dh, width=12)
        """The tkinter Entry object to enter the horizontal distance between elements of the antenna array. """
        lbl_dh = tk.Label(master=frm_dh, text="horizontal distance in wavelenghts")
        self.ent_dh.grid(row=1, column=0, sticky="e")
        lbl_dh.grid(row=0, column=0, sticky="w")
        self.ent_dh.insert(tk.END,str(self.dh))
        
        frm_dv = tk.Frame(master=self.window)
        self.ent_dv = tk.Entry(master=frm_dv, width=12)
        """The tkinter Entry object to enter the vertical distance between elements of the antenna array. """
        lbl_dv = tk.Label(master=frm_dv, text="vertical distance in wavelenghts")
        self.ent_dv.grid(row=1, column=0, sticky="e")
        lbl_dv.grid(row=0, column=0, sticky="w")
        self.ent_dv.insert(tk.END,str(self.dv))
               
        frm_bearing = tk.Frame(master=self.window)
        self.ent_bearing = tk.Entry(master=frm_bearing, width=12)
        """The tkinter Entry object to enter the bearing angle of the antenna array. """
        lbl_bearing = tk.Label(master=frm_bearing, text="Bearing angle (radians)")
        self.ent_bearing.grid(row=1, column=0, sticky="e")
        lbl_bearing.grid(row=0, column=0, sticky="w")
        self.ent_bearing.insert(tk.END,str(self.bearing))
            
        frm_downtilt = tk.Frame(master=self.window)
        self.ent_downtilt = tk.Entry(master=frm_downtilt, width=12)
        """The tkinter Entry object to enter the downtilt angle of the antenna array. """
        lbl_downtilt = tk.Label(master=frm_downtilt, text="Downtilt angle (radians)")
        self.ent_downtilt.grid(row=1, column=0, sticky="e")
        lbl_downtilt.grid(row=0, column=0, sticky="w")
        self.ent_downtilt.insert(tk.END,str(self.downtilt))
                
        frm_slant = tk.Frame(master=self.window)
        self.ent_slant = tk.Entry(master=frm_slant, width=12)
        """The tkinter Entry object to enter the slant angle of the antenna array. """
        lbl_slant = tk.Label(master=frm_slant, text="Slant angle (radians)")
        self.ent_slant.grid(row=1, column=0, sticky="e")
        lbl_slant.grid(row=0, column=0, sticky="w")
        self.ent_slant.insert(tk.END,str(self.slant))
        
        frm_antenna.grid(row=0, column=1, padx=10)
        frm_rows.grid(row=1, column=0, padx=10)
        frm_cols.grid(row=1, column=1, padx=10)        
        frm_dh.grid(row=1, column=2, padx=10)        
        frm_dv.grid(row=1, column=3, padx=10)        
        frm_bearing.grid(row=3, column=0, padx=10)
        frm_downtilt.grid(row=3, column=1, padx=10)
        frm_slant.grid(row=3, column=2, padx=10)        
        
    def gui_array_factor(self):
        """This method configures the additional rotation angles for the LOS beamforming.
        
        """                       
        frm_array = tk.Frame(master=self.window)
        lbl_array = tk.Label(master=frm_array, text="Additional rotation angles for beamforming" ,fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_array.grid(row=0, column=0, sticky="w")
        
        frm_phi = tk.Frame(master=self.window)
        self.ent_phi = tk.Entry(master=frm_phi, width=12)
        """The tkinter Entry object to enter the azimuth additional rotation angle for LOS beamforming. """
        lbl_phi = tk.Label(master=frm_phi, text="Azimuth rotation angle\n in radians")
        self.ent_phi.grid(row=1, column=0, sticky="e")
        lbl_phi.grid(row=0, column=0, sticky="w")
        self.ent_phi.insert(tk.END,str(self.phi))
        
        frm_theta = tk.Frame(master=self.window)
        self.ent_theta = tk.Entry(master=frm_theta, width=12)
        """The tkinter Entry object to enter the inclination additional rotation angle for LOS beamforming. """
        lbl_theta = tk.Label(master=frm_theta, text= " Inclination rotation angle\n in radians")
        self.ent_theta.grid(row=1, column=0, sticky="e")
        lbl_theta.grid(row=0, column=0, sticky="w")
        self.ent_theta.insert(tk.END,str(self.theta))
        
        frm_array.grid(row=4, column=1, padx=10)
        frm_phi.grid(row=5, column=0, sticky="w")
        frm_theta.grid(row=5, column=2, sticky="w")        
      
    def gui_antenna_element(self):
        """This method selcts and configures the antenna elements.
        
        """                       
        frm_element = tk.Frame(master=self.window)
        lbl_element = tk.Label(master=frm_element, text="Antenna Element",fg = "blue",bg = "white",font = "Verdana 14 bold")
        lbl_element.grid(row=0, column=0, sticky="w")
              
        btn_pattern = tk.Button(master=self.window,text="Antenna Element Pattern",command=self.cmd_element_pattern )
        btn_pattern_array = tk.Button(master=self.window,text="Antenna Element Pattern by array factor",command=self.cmd_pattern_array )
        btn_select = tk.Button(master=self.window,text="Select",command=self.cmd_select)

        self.v = tk.IntVar(self.window)
        """This variable store the type of the antenna element. """ 
        if self.ant_type == 1:
            self.v.set(1) 
        else:
            self.v.set(2)
        rb_iso = tk.Radiobutton(master=self.window, text='Isotropic',variable = self.v,value =1,command=self.cmd_rbiso)
        rb_3gpp = tk.Radiobutton(master=self.window, text='3gpp(Parabolic)',variable = self.v,value =2, command=self.cmd_rb3gpp)

        self.frm_isotropic = self.dFrame(master=self.window)
        """ The frame to group the isotropic antenna element configuration. """
        self.ent_gain_iso = tk.Entry(master=self.frm_isotropic, width=12)
        """The tkinter Entry object to enter the gain of the isotropic antenna element. """
        lbl_gain_iso = tk.Label(master=self.frm_isotropic, text="Max gain in db")
        self.ent_gain_iso.grid(row=1, column=0, sticky="e")
        lbl_gain_iso.grid(row=0, column=0, sticky="w")
        self.ent_gain_iso.insert(tk.END,str(self.gain_iso))
        self.frm_isotropic.disable()
        
        self.frm_3gpp = self.dFrame(master=self.window)
        """ The frame to group the 3gpp antenna element configuration. """
        self.ent_gain_3gpp = tk.Entry(master=self.frm_3gpp, width=12)
        """The tkinter Entry object to enter the gain of the 3gpp antenna element. """
        lbl_gain_3gpp = tk.Label(master=self.frm_3gpp, text="Max gain in db")
        self.ent_gain_3gpp.grid(row=1, column=0, sticky="e")
        lbl_gain_3gpp.grid(row=0, column=0, sticky="w")
        self.ent_gain_3gpp.insert(tk.END,str(self.gain_3gpp))
        
        self.ent_SLA_v_3gpp = tk.Entry(master=self.frm_3gpp, width=12)
        """The tkinter Entry object to enter the Slidelobe level limit in db."""  
        lbl_SLA_v_3gpp = tk.Label(master=self.frm_3gpp, text="SLA_v in db")
        self.ent_SLA_v_3gpp.grid(row=3, column=0, sticky="e")
        lbl_SLA_v_3gpp.grid(row=2, column=0, sticky="w")
        self.ent_SLA_v_3gpp.insert(tk.END,str(self.SLA_v))
        
        self.ent_A_max_3gpp = tk.Entry(master=self.frm_3gpp, width=12)
        """The tkinter Entry object to enter the Front-back ratio in db."""  
        lbl_A_max_3gpp = tk.Label(master=self.frm_3gpp, text="A max in db")
        self.ent_A_max_3gpp.grid(row=5, column=0, sticky="e")
        lbl_A_max_3gpp.grid(row=4, column=0, sticky="w")
        self.ent_A_max_3gpp.insert(tk.END,str(self.A_max))
        
        self.ent_beam_3gpp = tk.Entry(master=self.frm_3gpp, width=12)
        """The tkinter Entry object to enter the beamwidth in degrees of the 3gpp antenna element."""  
        lbl_beam_3gpp = tk.Label(master=self.frm_3gpp, text="Beamwidth in degrees")
        self.ent_beam_3gpp.grid(row=7, column=0, sticky="e")
        lbl_beam_3gpp.grid(row=6, column=0, sticky="w")
        self.ent_beam_3gpp.insert(tk.END,str(self.beam))
        
        self.frm_3gpp.disable()
        if self.ant_type == 1:
            self.var = 1
            self.frm_isotropic.enable()
            self.frm_3gpp.disable()
        else:
            self.var = 2
            self.frm_isotropic.disable()
            self.frm_3gpp.enable()
        
        frm_element.grid(row=6, column=1, padx=10)
        rb_iso.grid(row=7, column=0, sticky="w")
        rb_3gpp.grid(row=7, column=2, sticky="w")
        self.frm_isotropic.grid(row=8, column=0, padx=10)
        self.frm_3gpp.grid(row=8, column=2, padx=10)
        btn_pattern.grid(row=9, column=0, pady=10)
        btn_pattern_array.grid(row=9, column=1, pady=10)  
        btn_select.grid(row=9, column=2, pady=10)  
              
    def cmd_element_pattern(self):
      """ This method is called if the user press the button to plot the radiation pattern. 
      
      """  
      if(self.var==1):  
          antenna = antennas.AntennaIsotropic(float(self.ent_gain_iso.get()))
          ga.plot_3d_pattern(antenna)
          ga.plot_radiation_pattterns(antenna)
      if(self.var==2):
          antenna = antennas.Antenna3gpp3D(float(self.ent_gain_3gpp.get()),float(self.ent_A_max_3gpp.get()),float(self.ent_SLA_v_3gpp.get()),float(self.ent_beam_3gpp.get()))
          ga.plot_3d_pattern(antenna)
          ga.plot_radiation_pattterns(antenna)
    
    def cmd_pattern_array(self):
      """ This method is called if the user press the button to plot the array factor and radiation pattern. 
      
      """  
      if(self.var==1):  
          antenna = antennas.AntennaIsotropic(float(self.ent_gain_iso.get()))
          ant  = antennas.AntennaArray3gpp(float(self.ent_dh.get()), float(self.ent_dv.get()), int(self.ent_rows.get()),int(self.ent_cols.get()) , float(self.ent_bearing.get()), float(self.ent_downtilt.get()), float(self.ent_slant.get()), antenna, 1)
          ga.plot_3d_pattern_array_factor_product(ant,float(self.ent_phi.get()),float(self.ent_theta.get()))
       
      if(self.var==2):
          antenna = antennas.Antenna3gpp3D(float(self.ent_gain_3gpp.get()),float(self.ent_A_max_3gpp.get()),float(self.ent_SLA_v_3gpp.get()),float(self.ent_beam_3gpp.get()))
          ant  = antennas.AntennaArray3gpp(float(self.ent_dh.get()), float(self.ent_dv.get()), int(self.ent_rows.get()),int(self.ent_cols.get()) , float(self.ent_bearing.get()), float(self.ent_downtilt.get()), float(self.ent_slant.get()), antenna, 1)
          ga.plot_3d_pattern_array_factor_product(ant,float(self.ent_phi.get()),float(self.ent_theta.get()))
     
    def cmd_rbiso(self):
      """ This method is called if the user click the isotropic radio button. 
      
      """           
      self.var = 1
      self.frm_isotropic.enable()
      self.frm_3gpp.disable()
    
    def cmd_rb3gpp(self):
      """ This method is called if the user click the 3gpp radio button. 
      
      """           
      self.var = 2
      self.frm_3gpp.enable()
      self.frm_isotropic.disable()
      
    def cmd_select(self):
        """ This method is called if the user click the Select scenario button. """ 
        try:
            ant = None
            if(self.var==1):  
              antenna = antennas.AntennaIsotropic(float(self.ent_gain_iso.get()))
              ant  = antennas.AntennaArray3gpp(float(self.ent_dh.get()), float(self.ent_dv.get()), int(self.ent_rows.get()),int(self.ent_cols.get()) , float(self.ent_bearing.get()), float(self.ent_downtilt.get()), float(self.ent_slant.get()), antenna, 1,self.antenna_name)
              #ga. plot_3d_pattern_array_factor_product(ant,np.pi/3,0)
           
            if(self.var==2):
              antenna = antennas.Antenna3gpp3D(float(self.ent_gain_3gpp.get()),float(self.ent_A_max_3gpp.get()),float(self.ent_SLA_v_3gpp.get()),float(self.ent_beam_3gpp.get()))
              ant  = antennas.AntennaArray3gpp(float(self.ent_dh.get()), float(self.ent_dv.get()), int(self.ent_rows.get()),int(self.ent_cols.get()) , float(self.ent_bearing.get()), float(self.ent_downtilt.get()), float(self.ent_slant.get()), antenna, 1,self.antenna_name)
              #ga. plot_3d_pattern_array_factor_product(ant,np.pi/3,0)
            self.function_cbk(ant,self.var,float(self.ent_phi.get()),float(self.ent_theta.get()))
        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )


    class dFrame(tk.Frame):        
        """ Auxiliary class to enable and disable a tkinter frame. 
        
        """
        def enable(self, st='normal'):
            """ This method enable all components of a frame. 
            
            @type st: string
            @param st: the state to set the frame: 'normal' or 'disable'. Default: 'normal'.
            """
            for w in self.winfo_children():
                # change its state
                w.config(state = st)
                # and then recurse to process ITS children
                        #cstate(w)
        def disable(self):
            """ This method disable all components of a frame. 
            
            """
            self.enable('disabled')





