#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This module is a gui for the configuration of the simulation scenario.

@author: pablo belzarena
"""
import tkinter as tk
import tkinter.font as tkfont
import graph.graph_scenarios as grs
import gui.gui_user_message as gum
import numpy as np

class AppScenarios():
    """ This class is the form for the configuration of the simulation scenario. """

    def __init__(self,window,function_cbk,title,ok_title,cname=[0],fcGHz=10,xmin=-100,xmax=100,ymin=-100,ymax=100,grid=25,bspos=[0,0,2],ptx_db=30,force_los=0,move_probability=0,v_min_scatters=0,v_max_scatters=0):
        """The constructor of the AppScenarios Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type cname: list.
        @param cname: A list with the slected scenarios.
        @type fcGHz: float .
        @param fcGHz: Frequency in GHz of the carrier frequency of the scenario.
        @type xmin: float .
        @param xmin: The minimum limit of the x coordinate in the scenario. 
        @type xmax: float .
        @param xmax: The maximum limit of the x coordinate in the scenario. 
        @type ymin: float .
        @param ymin: The minimum limit of the y coordinate in the scenario. 
        @type ymax: float .
        @param ymax: The maximum limit of the y coordinate in the scenario. 
        @type grid: int .
        @param grid: For calculating the spacial distribution of the parameters of the scenario, 
        the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate. 
        @type bspos: 3d array or list .
        @param bspos: The position of the Base Satation in the scenario in the coordinates system [x,y,z].
        @type ptx_db: float.
        @param ptx_db: The power transmited by the base station in dbm. 
        @type force_los: int
        @param force_los: This parameter can take value 0 if LOS condition is forced to NLOS, 1 if is forcd to LOS, and 2 if iti is calculated from the probability model.
        @type move_probability: float
        @param move_probability: The probability that each scatter is moving.
        @type v_min_scatters: float
        @param v_min_scatters: The minimum velocity of one scatter movement.
        @type v_max_scatters: float
        @param v_max_scatters: The maximum velocity of one scatter movement.

        """ 
        self.window = window
        """ The main window of this form. """ 
        self.window.title(title)
        self.cname = cname
        """  A list with the slected scenarios. """ 
        self.ok_title = ok_title
        """  The text to show in the OK button.""" 
        self.fcGHz = fcGHz
        """ Frequency in GHz of the carrier frequency of the scenario. """ 
        self.xmin = xmin
        """ The minimum limit of the x coordinate in the scenario.  """ 
        self.xmax = xmax
        """ The maximum limit of the x coordinate in the scenario.  """ 
        self.ymin = ymin
        """ The minimum limit of the y coordinate in the scenario.  """ 
        self.ymax = ymax
        """ The maximum limit of the y coordinate in the scenario.  """ 
        self.grid = grid
        """ For calculating the spacial distribution of the parameters of the scenario, 
        the scenario is divided by a grid in x and y cordinates. This value is the number of divisions in each coordinate.  """ 
        self.bspos = bspos
        """ The position of the Base Satation in the scenario in the coordinates system [x,y,z]. """ 
        self.ptx_db = ptx_db
        """ The power transmited by the base station in dbm.  """ 
        self.force_los = force_los
        """ This parameter can take value 0 if LOS condition is forced to NLOS, 1 if is forcd to LOS, and 2 if iti is calculated from the probability model. """ 
        self.var = force_los
        """  This variable can take value 0 if LOS condition is forced to NLOS, 1 if is forcd to LOS, and 2 if iti is calculated from the probability model. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.move_probability = move_probability
        """ The probability that each scatter is moving.""" 
        self.v_min_scatters = v_min_scatters
        """ The minimum velocity of one scatter movement.""" 
        self.v_max_scatters = v_max_scatters
        """ The maximum velocity of one scatter movement. """ 
        self.MSs_move = np.empty(shape=(1),dtype= object)
        """ An array of objects where each object has the movement configuration of one MS. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """    
        font = tkfont.Font(family="Helvetica", size=25, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/3)

        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        frm_xmin = tk.Frame(master=self.window)
        self.ent_xmin = tk.Entry(master=frm_xmin, width=12)
        """The tkinter Entry object to enter the x min limit of the scenario"""
        lbl_xmin = tk.Label(master=frm_xmin, text="Scenario x_min in m")
        self.ent_xmin.grid(row=1, column=0, sticky="e")
        lbl_xmin.grid(row=0, column=0, sticky="w")
        self.ent_xmin.insert(tk.END,str(self.xmin))
        frm_xmin.grid(row=0, column=0, padx=10)
        
        frm_xmax = tk.Frame(master=self.window)
        self.ent_xmax = tk.Entry(master=frm_xmax, width=12)
        """The tkinter Entry object to enter the x max limit of the scenario"""
        lbl_xmax = tk.Label(master=frm_xmax, text="Scenario x_max in m")
        self.ent_xmax.grid(row=1, column=0, sticky="e")
        lbl_xmax.grid(row=0, column=0, sticky="w")
        self.ent_xmax.insert(tk.END,str(self.xmax))
        frm_xmax.grid(row=0, column=1, padx=10)
        
        frm_ymin = tk.Frame(master=self.window)
        self.ent_ymin = tk.Entry(master=frm_ymin, width=12)
        """The tkinter Entry object to enter the y min limit of the scenario"""
        lbl_ymin = tk.Label(master=frm_ymin, text="Scenario y_min in m")
        self.ent_ymin.grid(row=1, column=0, sticky="e")
        lbl_ymin.grid(row=0, column=0, sticky="w")
        self.ent_ymin.insert(tk.END,str(self.ymin))
        frm_ymin.grid(row=0, column=2, padx=10)
        
        frm_ymax = tk.Frame(master=self.window)
        self.ent_ymax = tk.Entry(master=frm_ymax, width=12)
        """The tkinter Entry object to enter the y max limit of the scenario"""
        lbl_ymax = tk.Label(master=frm_ymax, text="Scenario y_max in m")
        self.ent_ymax.grid(row=1, column=0, sticky="e")
        lbl_ymax.grid(row=0, column=0, sticky="w")
        self.ent_ymax.insert(tk.END,str(self.ymax))
        frm_ymax.grid(row=0, column=3, padx=10)

        frm_grid = tk.Frame(master=self.window)
        self.ent_grid = tk.Entry(master=frm_grid, width=12)
        """The tkinter Entry object to enter the size of the grid of the scenario"""
        lbl_grid = tk.Label(master=frm_grid, text="number of grid elements")
        self.ent_grid.grid(row=1, column=0, sticky="e")
        lbl_grid.grid(row=0, column=0, sticky="w")
        self.ent_grid.insert(tk.END,str(self.grid))
        frm_grid.grid(row=0, column=4, padx=10)

        frm_fc = tk.Frame(master=self.window)
        self.ent_fc = tk.Entry(master=frm_fc, width=12)
        """The tkinter Entry object to enter the frequency of the scenario"""
        lbl_fc = tk.Label(master=frm_fc, text="Frequency in GHz")
        self.ent_fc.grid(row=1, column=0, sticky="e")
        lbl_fc.grid(row=0, column=0, sticky="w")
        self.ent_fc.insert(tk.END,str(self.fcGHz))
        frm_fc.grid(row=1, column=0, padx=10)
        
        frm_ptx = tk.Frame(master=self.window)
        self.ent_ptx = tk.Entry(master=frm_ptx, width=12)
        """The tkinter Entry object to enter the Tx power of the BS in dbm"""
        lbl_ptx = tk.Label(master=frm_ptx, text="BS Tx power in dbm ")
        self.ent_ptx.grid(row=1, column=0, sticky="e")
        lbl_ptx.grid(row=0, column=0, sticky="w")
        self.ent_ptx.insert(tk.END,str(self.ptx_db))
        frm_ptx.grid(row=1, column=1, padx=10)

        frm_bsx = tk.Frame(master=self.window)
        self.ent_bsx = tk.Entry(master=frm_bsx, width=12)
        """The tkinter Entry object to enter the x coordinate of the BS in the scenario"""
        lbl_bsx = tk.Label(master=frm_bsx, text="BS position x in m")
        self.ent_bsx.grid(row=1, column=0, sticky="e")
        lbl_bsx.grid(row=0, column=0, sticky="w")
        self.ent_bsx.insert(tk.END,str(self.bspos[0]))
        frm_bsx.grid(row=1, column=2, padx=10)

        frm_bsy = tk.Frame(master=self.window)
        self.ent_bsy = tk.Entry(master=frm_bsy, width=12)
        """The tkinter Entry object to enter the y coordinate of the BS in the scenario"""
        lbl_bsy = tk.Label(master=frm_bsy, text="BS position y in m ")
        self.ent_bsy.grid(row=1, column=0, sticky="e")
        lbl_bsy.grid(row=0, column=0, sticky="w")
        self.ent_bsy.insert(tk.END,str(self.bspos[1]))
        frm_bsy.grid(row=1, column=3, padx=10)
        
        frm_bsz = tk.Frame(master=self.window)
        self.ent_bsz = tk.Entry(master=frm_bsz, width=12)
        """The tkinter Entry object to enter the z coordinate of the BS in the scenario"""
        lbl_bsz = tk.Label(master=frm_bsz, text="BS position z in m ")
        self.ent_bsz.grid(row=1, column=0, sticky="e")
        lbl_bsz.grid(row=0, column=0, sticky="w")
        self.ent_bsz.insert(tk.END,str(self.bspos[2]))
        frm_bsz.grid(row=1, column=4, padx=10)
     
        self.v = tk.IntVar(self.window)
        """The tkinter int variable object to LOS condition of the scenario."""
        self.v.set(self.force_los)
        frm_los = tk.Frame(master=self.window)
        lbl_los = tk.Label(master=frm_los, text="Select LOS model")
        rb_prob = tk.Radiobutton(master=frm_los, text='From probability model',variable = self.v,value = 2,command=self.cmd_prob)
        rb_los = tk.Radiobutton(master=frm_los, text='Force LOS',variable = self.v,value =1, command=self.cmd_los)
        rb_nlos = tk.Radiobutton(master=frm_los, text='Force NLOS',variable = self.v,value =0, command=self.cmd_nlos)
        lbl_los.grid(row=0, column=0, sticky="w")
        rb_prob.grid(row=1, column=0, sticky="e")
        rb_los.grid(row=2, column=0, sticky="w")
        rb_nlos.grid(row=3, column=0, sticky="w")
        frm_los.grid(row=3, column=1, padx=10)
        
        
        frm_nMS = tk.Frame(master=self.window)
        self.ent_nMS = tk.Entry(master=frm_nMS, width=12)
        """The tkinter Entry object to enter the number of MSs"""
        lbl_nMS = tk.Label(master=frm_nMS, text="Number of MSs")
        aux1 = tk.Button(master=frm_nMS, text="Configure each \nMS movement" ,font=font, compound=tk.CENTER, command=self.openfrmMS)

        lbl_nMS.grid(row=0, column=0, sticky="w")
        self.ent_nMS.grid(row=1, column=0, sticky="e")
        aux1.grid(row=2, column=0, padx=10)
        aux1.config(width=square_size, height=int(square_size/2))
        self.ent_nMS.insert(tk.END," 1 ")
        frm_nMS.grid(row=3, column=2, padx=10)
        
        self.vmove = tk.IntVar(self.window)
        """The tkinter int variable object to LOS condition of the scenario."""
        self.vmove.set(0)
        frm_scmv = tk.Frame(master=self.window)
        rb_nomove = tk.Radiobutton(master=frm_scmv, text='Scatters no move',variable = self.vmove,value = 0,command=self.cmd_sc_nomove)
        rb_move = tk.Radiobutton(master=frm_scmv, text='Scatters move',variable = self.vmove,value =1, command=self.cmd_sc_move)
        rb_nomove.grid(row=1, column=0, sticky="e")
        rb_move.grid(row=2, column=0, sticky="w")
         
        self.btscmv = tk.Button(master=frm_scmv, text="Configure the \n scatters \n movement" ,font=font, compound=tk.CENTER, command=self.openfrm_scatters_move)
        """ Tk button to open the scatters movement configuration form"""
        self.btscmv.grid(row=3, column=0, padx=10)
        self.btscmv.config(width=square_size, height=int(square_size/2),state = tk.DISABLED)
        frm_scmv.grid(row=3, column=3, padx=10)

        self.vmode = tk.IntVar(self.window)
        """The tkinter int variable object to LOS condition of the scenario."""
        self.vmode.set(2)
        frm_mode = tk.Frame(master=self.window)
        lbl_mode = tk.Label(master=frm_mode, text="Select the ssp \n spatial concictency model")

        rb_nospcons = tk.Radiobutton(master=frm_mode, text='No spatial consistency',variable = self.vmode,value = 0)
        rb_gridssps = tk.Radiobutton(master=frm_mode, text='Grid of ssps ',variable = self.vmode,value =1)
        rb_grid3gpp = tk.Radiobutton(master=frm_mode, text='Grid of ssps and update in drop  ',variable = self.vmode,value =2)        
        lbl_mode.grid(row=0, column=0, sticky="w")
        rb_nospcons.grid(row=1, column=0, sticky="w")
        rb_gridssps.grid(row=2, column=0, sticky="w")
        rb_grid3gpp.grid(row=3, column=0, sticky="w")
        frm_mode.grid(row=3, column=4, padx=10)
    
        
        self.gui_select_scenarios(2,0)    
        
    def gui_select_scenarios(self,row,col):
        """This method builds the select scenario list part of the form.
        
        @type row: int
        @param row: the number of row in the form to insert the select scenario list.
        @type col: int
        @param col: the number of column in the form to insert the select scenario list.
        """         
        font = tkfont.Font(family="Helvetica", size=25, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/3)

        show = tk.Label(self.window, text = "Select scenario", font = ("Times", 14), padx = 10, pady = 10)
        #self.show.pack() 
        show.grid(row=row, column=col, padx=10)
        self.lb = tk.Listbox(self.window,exportselection=False)#, selectmode = "multiple")
        """ A tkinter Listbox object to select the scenario. """ 
        #self.lb.pack(padx = 10, pady = 10, expand = tk.YES, fill = "both") 
        self.lb.grid(row=row+1, column=col, padx=10)
        x =["3gpp Indoor", "3gpp UMA", "3gpp UMI"]

        for item in range(len(x)): 
            self.lb.insert(tk.END, x[item]) 
            self.lb.itemconfig(item, bg="#bdc1d6") 
        if self.cname is not None:
            for i in self.cname:
                self.lb.select_set(i)
                
        self.lb.bind('<<ListboxSelect>>', self.onselect)
        aux1 = tk.Button(self.window, text=self.ok_title,font=font, compound=tk.CENTER, command=self.showSelected)
        aux1.grid(row=row+2, column=col+5, padx=10)
        aux1.config(width=square_size, height=int(square_size/2))
        aux0 = tk.Button(self.window, text="Graph  \n Pathloss", font=font, compound=tk.CENTER,command=self.cmd_gr_pl)
        aux0.grid(row=row+2, column=col, padx=10)
        aux0.config(width=square_size, height=int(square_size/2))

    def openfrmMS(self):
        """This method opens the form to configure one of the MS movement.
        
        """      
        nMS = int(self.ent_nMS.get())
        self.MSs_move = np.empty(shape=(nMS),dtype= object)
        self.windowsMS = np.empty(shape=(nMS),dtype= object)
        """ An array of objects where each object is a tk.TK() window for the AppMSconfig form. """ 
        self.appMS = np.empty(shape=(nMS),dtype= object)
        """ An array of objects where each object is an AppMSconfig form. """         
        self.windowsMS[0] = tk.Tk()
        self.appMS[0] = AppMSconfig(self.windowsMS[0],self.configMS,0," Configure the movement of MS numer: "+str(0) ,"OK",mspos=[5,1,2],msvel=[10,10,0],length=20,samples = 0.1)
        self.windowsMS[0].mainloop()

    def openfrm_scatters_move(self):
        """This method opens the form to configure the scatters movment.
        
        """      
        self.windows_scatters = tk.Tk()
        """ The tk.TK() window for the AppScatterMovement form. """ 
        appscmv = AppScatterMovement(self.windows_scatters,self.config_scatters_move," Configure the movement of the scatters " ,"OK",self.move_probability, self.v_min_scatters,self.v_max_scatters)
        self.windows_scatters.mainloop()


    def config_scatters_move(self,move_probability, v_min_scatters,v_max_scatters):
        """The callback function of the APPScatterMOvment form 
        
        @type move_probability: float
        @param move_probability: The probability that each scatter is moving.
        @type v_min_scatters: float
        @param v_min_scatters: The minimum velocity of one scatter movement.
        @type v_max_scatters: float
        @param v_max_scatters: The maximum velocity of one scatter movement.

        """
        self.windows_scatters.destroy()
        self.move_probability = move_probability
        self.v_min_scatters = v_min_scatters
        self.v_max_scatters = v_max_scatters
      

    def configMS(self,mspos,msvel,length,samples,ms_number):
        """The callback function of the AppMSconfig form. 
        
        @type mspos: Array or list .
        @param mspos: The initial position of the MS in the scenario in the coordinates system [x,y,z].
        @type msvel: Array or list .
        @param msvel: The velocity of the MS in the scenario in the coordinates system [x,y,z].
        @type length: float        
        @param length: The length of the MS route in meters. 
        @type samples: float
        @param samples: The distance between sample points in the MS route. In meters.
        @type ms_number: int.
        @param ms_number: The number of the MS.
        
        """ 
        self.windowsMS[ms_number].destroy()
        self.MSs_move[ms_number] = np.array([mspos,msvel,length,samples],dtype= object) 
        if ms_number < int(self.ent_nMS.get())-1:
            self.windowsMS[ms_number +1] = tk.Tk()
            """ The tk.TK() window for the AppMSconfig form. """ 
            self.appMS[ms_number+1] = AppMSconfig(self.windowsMS[ms_number +1],self.configMS,ms_number+1," Configure the movement of MS numer: "+str(ms_number+1) ,"OK",mspos=[5,1,2],msvel=[10,10,0],length=20,samples = 0.1)
            self.windowsMS[ms_number+1].mainloop()
   
        
    def cmd_sc_move(self):
        """ This method is called when the user click on the 'scatters move' radio button. 
        
        """  
        self.btscmv.config(state = tk.NORMAL)

    def cmd_sc_nomove(self):
        """ This method is called when the user click on the 'scatters no move' radio button. 
        
        """  
        self.btscmv.config(state = tk.DISABLED)


    def cmd_prob(self):
        """ This method is called when the user click on the 'from probability model' radio button. 
        
        """  
        self.var = 2

    def cmd_los(self):
        """ This method is called when the user click on the 'Force LOS' radio button. 
        
        """  
        self.var = 1

    def cmd_nlos(self):
        """ This method is called when the user click on the 'Force NLOS' radio button. 
        
        """  
        self.var = 0
        
    def showSelected(self):
        """ This method is called when the user press the select scenario button.
        
        This method calls the callback function.
        """ 
        try:
            nMS = int(self.ent_nMS.get())
            for i in range(nMS):
                print(self.MSs_move[i])
                if self.MSs_move[i] is None:
                    gum.AppUserMsg("Error message", " You must configure the Movement of all MSs" )
                    return
            cname = self.lb.curselection()
            bspos = np.array([float(self.ent_bsx.get()),float(self.ent_bsy.get()),float(self.ent_bsz.get())])
            self.function_cbk(cname,float(self.ent_fc.get()),float(self.ent_xmin.get()),float(self.ent_xmax.get()),float(self.ent_ymin.get()),float(self.ent_ymax.get()),int(self.ent_grid.get()),bspos,float(self.ent_ptx.get()),self.var,self.MSs_move,self.vmove.get(),self.move_probability,self.v_min_scatters,self.v_max_scatters,self.vmode.get())
        except BaseException as error:
           gum.AppUserMsg('Exception occurred in Scenario!','{}'.format(error)+ " try again please!" )
    
    def cmd_gr_pl(self):
        """ This method is called when the user press the graph path loss button.
        
        This method calls the graph_pathloss mentod in the graph_scenarios module.
        """         
        cname = self.lb.curselection()
        if cname is not None:
            bspos = [float(self.ent_bsx.get()),float(self.ent_bsy.get()),float(self.ent_bsz.get())]
            grs.graph_pathloss(cname,float(self.ent_fc.get()),float(self.ent_xmin.get()),float(self.ent_xmax.get()),float(self.ent_ymin.get()),float(self.ent_ymax.get()),int(self.ent_grid.get()),bspos,float(self.ent_ptx.get()))
        else:
            gum.AppUserMsg("Error message", " Scenarios are not selected! \n Please select one or multiple scenarios " )

        
    def onselect(self,evt):
        """ This method is called when the user click on one element of the selct scenario Listbox
        
        @type evt: tkinter event.
        @param evt: A tkinter event object that indicates the scenario selected.
        """
        w = evt.widget
        index = int(w.curselection()[0])
        if index > 0:
            self.ent_xmin.delete(0,tk.END)
            self.ent_xmin.insert(0,"-1000")
            self.ent_xmax.delete(0,tk.END)
            self.ent_xmax.insert(0,"1000")
            self.ent_ymin.delete(0,tk.END)
            self.ent_ymin.insert(0,"-1000")
            self.ent_ymax.delete(0,tk.END)
            self.ent_ymax.insert(0,"1000") 
        else:
            self.ent_xmin.delete(0,tk.END)
            self.ent_xmin.insert(0,"-100")
            self.ent_xmax.delete(0,tk.END)
            self.ent_xmax.insert(0,"100")
            self.ent_ymin.delete(0,tk.END)
            self.ent_ymin.insert(0,"-100")
            self.ent_ymax.delete(0,tk.END)
            self.ent_ymax.insert(0,"100") 
            
            
class AppMSconfig():
    """ This class is the form for the configuration of the movement of the MSs. """

    def __init__(self,window,function_cbk,MS,title,ok_title,mspos=[5,1,2],msvel=[10,10,0],length=20,samples = 0.1):
        """The constructor of the AppMSconfig Class.
        
        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
 
        @type mspos: Array or list .
        @param mspos: The initial position of the MS in the scenario in the coordinates system [x,y,z].
        @type msvel: Array or list .
        @param msvel: The velocity of the MS in the scenario in the coordinates system [x,y,z].
        @type length: float        
        @param length: The length of the MS route in meters. 
        @type samples: float
        @param samples: The distance between sample points in the MS route. In meters.

        """ 
        self.window = window
        """ The main window of this form. """ 
        self.window.title(title)
        """ The title of this form""" 
        self.ok_title = ok_title
        """  The text to show in the OK button.""" 
        self.MS = MS
        """  The number of the MS.""" 
        self.mspos = mspos
        """ The initial position of the MS in the scenario in the coordinates system [x,y,z]. """ 
        self.msvel = msvel
        """ The velocity of the MS in the scenario in the coordinates system [x,y,z]. """ 
        self.len = length
        """ The length of the MS route in meters.  """ 
        self.samples = samples
        """ The distance between sample points in the MS route. In meters. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the main form to enter the data
        
        """         
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        frm_msx = tk.Frame(master=self.window)
        self.ent_msx = tk.Entry(master=frm_msx, width=12)
        """The tkinter Entry object to enter the x coordinate of the start position of the MS in the scenario"""
        lbl_msx = tk.Label(master=frm_msx, text="MS position x in m ")
        self.ent_msx.grid(row=1, column=0, sticky="e")
        lbl_msx.grid(row=0, column=0, sticky="w")
        self.ent_msx.insert(tk.END,str(self.mspos[0]))
        frm_msx.grid(row=1, column=1, padx=10)

        frm_msy = tk.Frame(master=self.window)
        self.ent_msy = tk.Entry(master=frm_msy, width=12)
        """The tkinter Entry object to enter the y coordinate of the start position of the MS in the scenario"""
        lbl_msy = tk.Label(master=frm_msy, text="MS position y in m ")
        self.ent_msy.grid(row=1, column=0, sticky="e")
        lbl_msy.grid(row=0, column=0, sticky="w")
        self.ent_msy.insert(tk.END,str(self.mspos[1]))
        frm_msy.grid(row=1, column=2, padx=10)
        
        frm_msz = tk.Frame(master=self.window)
        self.ent_msz = tk.Entry(master=frm_msz, width=12)
        """The tkinter Entry object to enter the z coordinate of the start position of the MS in the scenario"""
        lbl_msz = tk.Label(master=frm_msz, text="MS position z in m ")
        self.ent_msz.grid(row=1, column=0, sticky="e")
        lbl_msz.grid(row=0, column=0, sticky="w")
        self.ent_msz.insert(tk.END,str(self.mspos[2]))
        frm_msz.grid(row=1, column=3, padx=10)
        
        frm_msvx = tk.Frame(master=self.window)
        self.ent_msvx = tk.Entry(master=frm_msvx, width=12)
        """The tkinter Entry object to enter the x coordinate of the velocity of the MS in the scenario"""
        lbl_msvx = tk.Label(master=frm_msvx, text="MS speed x in m/s ")
        self.ent_msvx.grid(row=1, column=0, sticky="e")
        lbl_msvx.grid(row=0, column=0, sticky="w")
        self.ent_msvx.insert(tk.END,str(self.msvel[0]))
        frm_msvx.grid(row=2, column=1, padx=10)

        frm_msvy = tk.Frame(master=self.window)
        self.ent_msvy = tk.Entry(master=frm_msvy, width=12)
        """The tkinter Entry object to enter the y coordinate of the velocity of the MS in the scenario"""
        lbl_msvy = tk.Label(master=frm_msvy, text="MS speed y in m/s ")
        self.ent_msvy.grid(row=1, column=0, sticky="e")
        lbl_msvy.grid(row=0, column=0, sticky="w")
        self.ent_msvy.insert(tk.END,str(self.msvel[1]))
        frm_msvy.grid(row=2, column=2, padx=10)
        
        frm_msvz = tk.Frame(master=self.window)
        self.ent_msvz = tk.Entry(master=frm_msvz, width=12)
        """The tkinter Entry object to enter the z coordinate of the velocity of the MS in the scenario"""
        lbl_msvz = tk.Label(master=frm_msvz, text="MS speed z in m/s ")
        self.ent_msvz.grid(row=1, column=0, sticky="e")
        lbl_msvz.grid(row=0, column=0, sticky="w")
        self.ent_msvz.insert(tk.END,str(self.msvel[2]))
        frm_msvz.grid(row=2, column=3, padx=10)
        
        frm_len = tk.Frame(master=self.window)
        self.ent_len = tk.Entry(master=frm_len, width=12)
        """The tkinter Entry object to enter the length of the route of the MS in the scenario"""
        lbl_len = tk.Label(master=frm_len, text="Length of the \n path in m ")
        self.ent_len.grid(row=1, column=0, sticky="e")
        lbl_len.grid(row=0, column=0, sticky="w")
        self.ent_len.insert(tk.END,str(self.len))
        frm_len.grid(row=2, column=4, padx=10)
        
        frm_sample = tk.Frame(master=self.window)
        self.ent_sample = tk.Entry(master=frm_sample, width=12)
        """The tkinter Entry object to enter the distance between samples of the MS route in the scenario"""
        lbl_sample = tk.Label(master=frm_sample, text="Distance between \n samples of the \n channel in m ")
        self.ent_sample.grid(row=1, column=0, sticky="e")
        lbl_sample.grid(row=0, column=0, sticky="w")
        self.ent_sample.insert(tk.END,str(self.samples))
        frm_sample.grid(row=2, column=5, padx=10)
        
        font = tkfont.Font(family="Helvetica", size=25, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/3)

        aux1 = tk.Button(self.window, text=self.ok_title,font=font, compound=tk.CENTER, command=self.cmd_ok)
        aux1.grid(row=3, column=5, padx=10)
        aux1.config(width=square_size, height=int(square_size/2))

        
    def cmd_ok(self):
        """ This method is called when the user press the select scenario button.
        
        This method calls the callback function.
        """ 
        try:
            mspos = np.array([float(self.ent_msx.get()),float(self.ent_msy.get()),float(self.ent_msz.get())])
            msvel = np.array([float(self.ent_msvx.get()),float(self.ent_msvy.get()),float(self.ent_msvz.get())])
            self.function_cbk(mspos,msvel,float(self.ent_len.get()) ,float(self.ent_sample.get()),self.MS)
        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )

class AppScatterMovement():
    """ This class is the form for the configuration of the movement of the scatters. """

    def __init__(self,window,function_cbk,title,ok_title,move_probability=0, v_min_scatters=0,v_max_scatters=0):
        """The constructor of the AppScatterMovement Class.

        @type window: tkinter.Tk() window.
        @param window: The main window of this form.
        @type function_cbk: python function.
        @param function_cbk: The callback function to return when the user press OK button.
        @type title: string
        @param title: The title of the form.
        @type ok_title: string
        @param ok_title: The text to show in the OK button.
        @type move_probability: float
        @param move_probability: The probability that each scatter is moving.
        @type v_min_scatters: float
        @param v_min_scatters: The minimum velocity of one scatter movement.
        @type v_max_scatters: float
        @param v_max_scatters: The maximum velocity of one scatter movement.
     
        """ 
        self.window = window
        """ The main window of this form. """ 
        self.window.title(title)
        """ The title of this form""" 
        self.ok_title = ok_title
        """  The text to show in the OK button.""" 
        self.move_probability = move_probability
        """ The probability that each scatter moves during the simulation. """
        self.v_min_scatters = v_min_scatters
        """ The minimum velocity of one scatter movement.""" 
        self.v_max_scatters = v_max_scatters
        """ The maximum velocity of one scatter movement. """ 
        self.function_cbk = function_cbk
        """ The callback function to return when the user press OK button. """ 
        self.gui_configuration()

    def gui_configuration(self):
        """This method builds the main form to enter the data.
        
        """         
        self.window.rowconfigure(6, minsize=50, weight=1)
        self.window.columnconfigure([0, 1, 2, 3, 4], minsize=50, weight=1) 
        
        frm_mvprob = tk.Frame(master=self.window)
        self.ent_mvprob = tk.Entry(master=frm_mvprob, width=12)
        """The tkinter Entry object to enter the probability that each scatter move or not."""
        lbl_mvprob = tk.Label(master=frm_mvprob, text="Probaility that \n each scatter \n move or not")
        self.ent_mvprob.grid(row=1, column=0, sticky="e")
        lbl_mvprob.grid(row=0, column=0, sticky="w")
        self.ent_mvprob.insert(tk.END,str(self.move_probability))
        frm_mvprob.grid(row=1, column=1, padx=10)

        frm_minv = tk.Frame(master=self.window)
        self.ent_minv = tk.Entry(master=frm_minv, width=12)
        """The tkinter Entry object to enter the minimum velocity of a scatter"""
        lbl_minv = tk.Label(master=frm_minv, text="The minimum speed \n of a scatters in m/s ")
        self.ent_minv.grid(row=1, column=0, sticky="e")
        lbl_minv.grid(row=0, column=0, sticky="w")
        self.ent_minv.insert(tk.END,str(self.v_min_scatters))
        frm_minv.grid(row=1, column=2, padx=10)


        frm_maxv = tk.Frame(master=self.window)
        self.ent_maxv = tk.Entry(master=frm_maxv, width=12)
        """The tkinter Entry object to enter the maximum velocity of a scatter"""
        lbl_maxv = tk.Label(master=frm_maxv, text="The maximum speed \n of a scatters in m/s ")
        self.ent_maxv.grid(row=1, column=0, sticky="e")
        lbl_maxv.grid(row=0, column=0, sticky="w")
        self.ent_maxv.insert(tk.END,str(self.v_max_scatters))
        frm_maxv.grid(row=1, column=3, padx=10)

        
        font = tkfont.Font(family="Helvetica", size=25, weight = tkfont.BOLD)
        square_size = int(font.metrics('linespace')/3)

        aux1 = tk.Button(self.window, text=self.ok_title,font=font, compound=tk.CENTER, command=self.cmd_ok)
        aux1.grid(row=2, column=3, padx=10)
        aux1.config(width=square_size, height=int(square_size/2))

        
    def cmd_ok(self):
        """ This method is called when the user press the select scenario button.
        
        This method calls the callback function.
        """ 
        try:
            self.function_cbk(float(self.ent_mvprob.get()) ,float(self.ent_minv.get()),float(self.ent_maxv.get()))
        except BaseException as error:
            gum.AppUserMsg('Exception occurred!','{}'.format(error)+ " try again please!" )



