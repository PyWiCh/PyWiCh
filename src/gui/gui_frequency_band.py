#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the OFDM frequency band of the system.

@author: pablobelzarena
"""

import tkinter as tk
import gui.gui_user_message as gum


class AppFreqBand():
    """ This class is the form for the configuration of the OFDM frequency band. """

    def __init__(self,window,function_cbk,title,ok_title,nprb=100,bwprb=180000,noisefig=5,thnoise=-174):
        """The constructor of the AppSFreqBand Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type bwprb: float
        @param bwprb: The bandwidth in Hz of each prb.
        @type nprb: int
        @param nprb: The number of Physical Resource Blocks in the OFDM frequency band.
        @type noisefig: int 
        @param noisefig: The noise figure of the equipments.Default value 5db.
        @type thnoise: float
        @param thnoise: The thermal noise in this frequency band.Default value -174 dbm/Hz.
        """
        self.ok_title = ok_title
        """ The text to show in the OK button.""" 
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.bwprb = bwprb
        """ TThe bandwidth in Hz of each prb. """
        self.nprb = int(nprb)
        """ The number of Physical resource blocks in the OFDM frequency band. """        
        self.noisefig = noisefig
        """ The noise figure of the equipments.Default value 5db. """ 
        self.thnoise =thnoise
        """ The thermal noise in this frequency band.Default value -174 dbm/Hz. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()


    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """         
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        frm_nprb = tk.Frame(master=self.window)
        self.ent_nprb = tk.Entry(master=frm_nprb, width=12)
        """The tkinter Entry object to enter the number of prbs """
        lbl_nprb = tk.Label(master=frm_nprb, text="Number of resource blocks (prbs)")
        self.ent_nprb.grid(row=1, column=0, sticky="e")
        lbl_nprb.grid(row=0, column=0, sticky="w")
        self.ent_nprb.insert(tk.END,str(self.nprb))
        frm_nprb.grid(row=0, column=0, padx=10)
        
        frm_bwprb = tk.Frame(master=self.window)
        self.ent_bwprb = tk.Entry(master=frm_bwprb, width=12)
        """The tkinter Entry object to enter the bandwidth of the prbs """
        lbl_bwprb = tk.Label(master=frm_bwprb, text="prb bandwidth (Hz)")
        self.ent_bwprb.grid(row=1, column=0, sticky="e")
        lbl_bwprb.grid(row=0, column=0, sticky="w")
        self.ent_bwprb.insert(tk.END,str(self.bwprb))
        frm_bwprb.grid(row=0, column=1, padx=10)
        
        frm_noisefig = tk.Frame(master=self.window)
        self.ent_noisefig = tk.Entry(master=frm_noisefig, width=12)
        """The tkinter Entry object to enter the noise figure. """
        lbl_noisefig = tk.Label(master=frm_noisefig, text="Noise Figure (db) ")
        self.ent_noisefig.grid(row=1, column=0, sticky="e")
        lbl_noisefig.grid(row=0, column=0, sticky="w")
        self.ent_noisefig.insert(tk.END,str(self.noisefig))
        frm_noisefig.grid(row=0, column=2, padx=10)
        
        frm_thnoise = tk.Frame(master=self.window)
        self.ent_thnoise = tk.Entry(master=frm_thnoise, width=12)
        """The tkinter Entry object to enter the thermal noise. """
        lbl_thnoise = tk.Label(master=frm_thnoise, text="Thermal noise (dbm/Hz) ")
        self.ent_thnoise.grid(row=1, column=0, sticky="e")
        lbl_thnoise.grid(row=0, column=0, sticky="w")
        self.ent_thnoise.insert(tk.END,str(self.thnoise))
        frm_thnoise.grid(row=0, column=3, padx=10)
        
        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=3, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
        
        This method calls the callback function.
        """ 
        try:
            self.function_cbk(int(self.ent_nprb.get()),float(self.ent_bwprb.get()),float(self.ent_noisefig.get()),float(self.ent_thnoise.get()))
        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )
 