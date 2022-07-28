#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for asking the user to select the Small Scale parameter (ssp) to graph.

@author: pablobelzarena
"""

import tkinter as tk
import tkinter.font as tkfont
import gui.gui_user_message as gum


class AppSSP():
    """ This class is the form for select the SSP """

    def __init__(self,window,function_cbk,title,ok_title,n_points=0):
        """The constructor of the AppSSP Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of this form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type n_points: string
        @param n_points: The number of points in the MS route.
        """
        self.ok_title = ok_title
        """ The text to show in the OK button.""" 
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.ssp_param = 0
        """ The ssp parameter selected. Default 0. """ 
        self.n_points = int(n_points)
        """ The number of points in the MS route. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()
        """ The number of points in the MS route.""" 

    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """ 
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        self.v = tk.IntVar(self.window)
        """This variable store the SSP parameter selected """ 
        self.v.set(self.ssp_param)
        frm_ssp = tk.Frame(master=self.window)
        rb_tau = tk.Radiobutton(master=frm_ssp, text='Cluster delays',variable = self.v,value =0)
        rb_pdp = tk.Radiobutton(master=frm_ssp, text='Power Delay Profile',variable = self.v,value =1 )
        rb_AOA = tk.Radiobutton(master=frm_ssp, text='Cluster Azimuth arrival angle',variable = self.v,value = 2)
        rb_AOD = tk.Radiobutton(master=frm_ssp, text='Cluster Azimuth departure angle ',variable = self.v,value =3)
        rb_ZOA = tk.Radiobutton(master=frm_ssp, text='Cluster Inclination arrival angle ',variable = self.v,value = 4)
        rb_ZOD = tk.Radiobutton(master=frm_ssp, text='Cluster Inclination departure angle ',variable = self.v,value =5)
        rb_tau.grid(row=0, column=0, sticky="e")
        rb_pdp.grid(row=1, column=0, sticky="e")
        rb_AOA.grid(row=2, column=0, sticky="e")
        rb_AOD.grid(row=3, column=0, sticky="e")
        rb_ZOA.grid(row=4, column=0, sticky="e")
        rb_ZOD.grid(row=5, column=0, sticky="e")
        self.listbox_points = tk.Listbox(self.window,exportselection=False)
        """ The tkinter.Listbox to select the point in MS route. """ 
        self.listbox_points.grid(row=0, column=1, padx=10)
        for values in range(self.n_points):
            self.listbox_points.insert(tk.END, values)
        frm_ssp.grid(row=0, column=0, padx=10)        
        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=0, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
        
        This method calls the callback function.
        """ 
        if len(self.listbox_points.curselection()) > 0:
            aux = self.listbox_points.curselection()[0]
            self.function_cbk(int(self.v.get()),aux)#,int(self.v1.get()))
        else: 
            gum.AppUserMsg("Error message", " Please, select the point of the path " )
           
