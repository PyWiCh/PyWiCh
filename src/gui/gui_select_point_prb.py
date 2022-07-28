#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for asking the user to select one Physical Resource Block (prb) and one point in the MS route..


@author: pablobelzarena
"""

import tkinter as tk
import tkinter.font as tkfont
import gui.gui_user_message as gum


class AppSelectPointPrb():
    """ This class is the form for select the prb and the point in  MS route. """

    def __init__(self,window,function_cbk,title,ok_title,n_points,n_prb):
        """The constructor of the AppSelectPointPrb Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type n_points: string
        @param n_points: The number of points in the MS route.
        @type n_prb: string
        @param n_prb: The number of Physical Resource Blocks in the OFDM frequency band.

        """
        self.ok_title = ok_title
        """ The text to show in the OK button.""" 
        self.window  = window
        """ The main window of this form.""" 
        self.window.title(title)
        """ The title of this form. """ 
        self.n_points = int(n_points)
        """ The number of points in the MS route. """
        self.n_prb = int(n_prb)
        """ The number of Physical resource blocks in the OFDM frequency band. """        
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()
        """ The number of points in the MS route.""" 

    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """         
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
 
        lbl_points = tk.Label(master=self.window, text="Selct point numer \n in path" )
        lbl_points.grid(row=0, column=1, sticky="w")

        self.listbox_points = tk.Listbox(self.window,exportselection=False)
        """ The tkinter.Listbox to select the point in MS route. """ 
        self.listbox_points.grid(row=1, column=1, padx=10)
        for values in range(self.n_points):
            self.listbox_points.insert(tk.END, values)
 
        lbl_prb = tk.Label(master=self.window, text="Selct prb number " )
        lbl_prb.grid(row=0, column=2, sticky="w")

        self.listbox_prb = tk.Listbox(self.window,exportselection=False)
        """ The tkinter.Listbox to select the prb number. """ 
        self.listbox_prb.grid(row=1, column=2, padx=10)
        for values in range(self.n_prb):
            self.listbox_prb.insert(tk.END, values)

        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=3, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
        
        This method calls the callback function.
        """ 
        if len(self.listbox_points.curselection()) > 0 and len(self.listbox_prb.curselection())>0 :
            self.function_cbk(self.listbox_points.curselection()[0],self.listbox_prb.curselection()[0])
        else:
            gum.AppUserMsg("Error message", "The point and prb cannot be empty " )

