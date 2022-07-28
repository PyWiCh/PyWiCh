#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This module is a gui to display messages to the user.

@author: pablo belzarena
"""

import tkinter as tk
import tkinter.font as tkfont

class AppUserMsg():
    """ This class is the form to display messages 
    
    """ 
    def __init__(self,title,msg):
        """The constructor of the AppUserMsg Class.
        
        @type title: string.
        @param title: The title of the message window.
        @type msg: string.
        @param msg: the message to show in the message window.
        
        """
        self.window_msg = tk.Tk()
        """ The tkinter window of the message window. """
        self.window_msg.title(title)
        self.window_msg.rowconfigure(4, minsize=20, weight=1)
        self.window_msg.columnconfigure([0, 1, 2], minsize=20, weight=1) 

        frm = tk.Frame(master=self.window_msg)
        lbl = tk.Label(master=frm, text=msg)
        lbl.grid(row=0, column=0, sticky="w")
        frm.grid(row=1, column=0, padx=10)
        
        font = tkfont.Font(family="Helvetica", size=25, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/4)

        aux0 = tk.Button(self.window_msg, text="OK", font=font, compound=tk.CENTER,command=self.cmd_msg_OK)
        aux0.grid(row=3, column=0, padx=10)
        aux0.config(width=square_size, height=int(square_size/2))

        self.window_msg.mainloop()
        
    def cmd_msg_OK(self):
        """This method is called when the user press the OK button of the message window.
        
        """ 
        self.window_msg.destroy() 

 