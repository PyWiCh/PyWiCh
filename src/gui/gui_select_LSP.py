#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for asking the user to select the Large Scale parameter (LSP) to graph.

@author: pablobelzarena
"""

import tkinter as tk

class AppLSP():
    """ This class is the form for select the LSP """
    
    def __init__(self,window,function_cbk,title,ok_title,lsp_param=0):
        """The constructor of the AppLSP Class
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type lsp_param: string
        @param lsp_param: The lsp parameter selected. Default 0.
 
        """
        self.ok_title = ok_title
        """ The text to show in the OK button.""" 
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        self.lsp_param = int(lsp_param)
        """ The lsp parameter selected. Default 0. """ 
        # self.los = int(los)
        # """ Takes the value "0" for NLOS, and "1" for LOS. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """ 
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        self.v = tk.IntVar(self.window)
        """This variable store the LSP parameter selected """ 
        self.v.set(self.lsp_param)
        frm_lsp = tk.Frame(master=self.window)
        rb_Shadow = tk.Radiobutton(master=frm_lsp, text='Shadowing',variable = self.v,value =0)
        rb_K = tk.Radiobutton(master=frm_lsp, text='Ricean K',variable = self.v,value =1 )
        rb_DS = tk.Radiobutton(master=frm_lsp, text='Delay Spread',variable = self.v,value = 2)
        rb_AOD = tk.Radiobutton(master=frm_lsp, text='Azimuth departure angle spread',variable = self.v,value =3)
        rb_AOA = tk.Radiobutton(master=frm_lsp, text='Azimuth arrival angle spread',variable = self.v,value = 4)
        rb_ZOD = tk.Radiobutton(master=frm_lsp, text='Inclination departure angle spread',variable = self.v,value =5)
        rb_ZOA = tk.Radiobutton(master=frm_lsp, text='Inclination arrival angle spread',variable = self.v,value = 6)
        rb_LOS = tk.Radiobutton(master=frm_lsp, text='LOS condition',variable = self.v,value = 7)
  
        rb_DS.grid(row=2, column=0, sticky="e")
        rb_K.grid(row=1, column=0, sticky="e")
        rb_Shadow.grid(row=0, column=0, sticky="e")
        rb_AOA.grid(row=3, column=0, sticky="e")
        rb_AOD.grid(row=4, column=0, sticky="e")
        rb_ZOA.grid(row=5, column=0, sticky="e")
        rb_ZOD.grid(row=6, column=0, sticky="e")
        rb_LOS.grid(row=7, column=0, sticky="e")

        # self.v1 = tk.IntVar(self.window)
        # """ This variable store the LOS or NLOS condition selected. """ 
        # self.v1.set(self.los)
        # frm_los = tk.Frame(master=self.window)
        # rb_los = tk.Radiobutton(master=frm_los, text='LOS LSP params',variable = self.v1,value =1)
        # rb_nlos = tk.Radiobutton(master=frm_los, text='NLOS LSP params',variable = self.v1,value =0)
        # rb_los.grid(row=0, column=0, sticky="w")
        # rb_nlos.grid(row=1, column=0, sticky="w")

        # frm_los.grid(row=0, column=1, padx=10)        
        frm_lsp.grid(row=0, column=0, padx=10)
        
        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=0, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
        
        This method calls the callback function.
        """ 
        self.function_cbk(int(self.v.get()))
    
