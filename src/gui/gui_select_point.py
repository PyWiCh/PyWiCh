#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for asking the user to select one point in the MS route.

@author: pablobelzarena
"""

import tkinter as tk
import tkinter.font as tkfont
import gui.gui_user_message as gum

class AppSelectPoint():
    """ This class is the form for select the point in MS route """

    def __init__(self,window,function_cbk,title,ok_title,n_points):
        """The constructor of the AppSelectPoint Class
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type n_points: int
        @param n_points: The number of points in the MS route.

        """

        self.ok_title = ok_title
        """ The text to show in the OK button. """ 
        self.n_points = n_points
        """ The number of points in the MS route. """ 
        self.window  = window
        """ he main window of this form. """
        self.window.title(title)
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the form to enter the data
        
        """ 
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        self.listbox_points = tk.Listbox(self.window,exportselection=False)
        """ The tkinter.Listbox to select the point in MS route. """ 
        self.listbox_points.grid(row=0, column=1, padx=10)
        for values in range(self.n_points):
            self.listbox_points.insert(tk.END, values)
        aux1 = tk.Button(self.window, text=self.ok_title, command=self.OK_button)
        aux1.grid(row=1, column=3, padx=10)

    def OK_button(self):
        """This method is called when user press the OK button
        
        """ 
        if len(self.listbox_points.curselection()) > 0  :
            self.function_cbk(self.listbox_points.curselection()[0])
        else:
            gum.AppUserMsg("Error message", "The point selection cannot be empty " )

