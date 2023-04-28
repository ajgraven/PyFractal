from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,NavigationToolbar2Tk)
import tkinter as tk
from tkinter import messagebox
import time
import numpy as np
import cupy as cp
from numpy import arange,exp
import FixedPoint as fp
from pylab import meshgrid,imshow,contour,clabel,colorbar,axis,title,show
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import lambertw
from matplotlib.backend_bases import MouseButton
from matplotlib.widgets import TextBox
import math
from pynput import keyboard
import keyboard as kb
from pylab import  show
import mouse
import matplotlib
import matplotlib.style as mplstyle
import ipywidgets as widgets
#import scipy.optimize as opt




class Fractal:
    '''
    Julia set/dynamical plane fractal object
    Attributes:
    is_julia : bool                : True if dynamical plane, False if parameter space
    f        : compiled expression : Julia set/parameter space function
    exitCond : compiled expression : iteration exit condition
    N        : int                 : iteration cap
    c        : complex number      : Julia set parameter
    Methods:
    f_nits          : n,z0                 -> first n iterates starting at z0
    escape_iter     : z                    -> min(# of iterations to escape from,N)
    get_fract       : xrng,yrng,res        -> grid of iterations to escape at ~res points in (xrng)x(yrng)
    find_per_orbit  : z0,period            -> periodic point near z0, or None if search fails
    find_per_orbits : xrng,yrng,res,period -> list of periodic points in (xrng)x(yrng), searching from res starting points
    '''
    def __init__(self,is_julia,f,exitCond,N=40,c=.4+.6j):
        self.is_julia = is_julia
        self.f        = f
        self.exitCond = exitCond
        self.N        = N
        self.c        = c # Only relevant if is_julia

        # Vectorize some methods to improve performance
        self.escape_iter=np.vectorize(self.escape_iter)
        self.find_per_orbit=np.vectorize(self.find_per_orbit)
        
    def eval_f(self,z,c):
        return eval(self.f,None,{"z":z,"c":c})

    def eval_exitCond(self,z,c):
        return eval(self.exitCond,None,{"z":z,"c":c})

    def f_nits(self,n,z0):
        '''
        Returns first n iterates starting at z0
        Terminates early if exit condition is met
        '''
        pts = [z0]
        c = self.c if self.is_julia else z0
        for i in range(n):
            if self.eval_exitCond(pts[-1],c):
                return pts
            try:
                pts=np.append(pts,[self.eval_f(pts[-1],c)],axis=0)
            except (ValueError,ArithmeticError,RuntimeError,MemoryError):
                return pts
        return pts

    def escape_iter(self,z):
        '''
        Returns number of iterations needed to escape
        starting at z. Returns maximum # of iterations,
        self.N, if error is thrown, or self.N is reached
        '''
        c = self.c if self.is_julia else z
        for i in range(self.N):
            if self.eval_exitCond(z,c):
                return i
            try:
                z=self.eval_f(z,c)
            except (ValueError,ArithmeticError,RuntimeError,MemoryError):
                return self.N
        return self.N

    def get_fract(self,xrng,yrng,res):
        return self.escape_iter(self.complex_grid(xrng,yrng,res))

    def find_per_orbit(self,z0,period):
        # Find periodic orbit, with initial guess z0
        try:
            if self.is_julia:
                return fp.per_point(lambda z:self.eval_f(z,self.c),z0,period)
            else:
                return fp.param_per_point(lambda z,c:self.eval_f(z,c),z0,period)
        except (RuntimeError,ZeroDivisionError):
            pass
        return None
        

    def find_per_orbits(self,xrng,yrng,res,period):
        per_orbs = self.find_per_orbit(self.complex_grid(xrng,yrng,res),period) # search for periodic orbits
        pts = []
        for xrow in per_orbs: # filter out "None"s and duplicate points
            for z in xrow:
                if z != None and (not (True in (abs(ele-z)<1e-5 for ele in pts))):
                    pts.append(z)
        return pts

    # Getters and setters
    def set_f(self,f):
        self.f = f

    def set_exitCond(self,exitCond):
        self.exitCond = exitCond

    def set_c(self,c):
        self.c = c

    def set_N(self,N):
        self.N = N

    def get_f(self):
        return self.f

    def get_exitCond(self):
        return self.exitCond

    def get_c(self):
        return self.c

    def get_N(self):
        return self.N

    # Utility functions
    def complex_grid(self,xrng,yrng,res):
        step = np.sqrt((xrng[1]-xrng[0])*(yrng[1]-yrng[0])/res) # compute step size
        X,Y=meshgrid(arange(xrng[0], xrng[1], step),arange(yrng[0], yrng[1], step)) # generate grid of sample points
        return X+Y*1j

            
            



class FractalPlot:
    def __init__(self,main_window,parent,is_julia,f,exitCond,xrng=[-2.25,.75],yrng=[-1.5,1.5],res=50000,N=40,c=.4+.6j):
        self.main_window = main_window
        self.parent      = parent

        # Create fractal object
        self.fract = Fractal(is_julia,f,exitCond,N,c)
        self.xrng  = xrng
        self.yrng  = yrng
        self.res   = res

        
        # Default values
        self.nits = 6
        self.per  = 1
        self.search_res = 200
        self.iplt_ms = 2 # iteration plot marker size
        self.iplt_lw = 1.0 # iteration plot line width
        self.iplt_alpha = .7 # iteration plot opacity
        self.perplt_ms = 8 # period plot marker size
        
        # Instantiate misc. environment variables
        self.ipts = []
        self.drawn_iplts = [] # list of drawn plots of iterates
        self.drawn_perplts = [] # list of drawn periodic points
        self.drawn_perplt_pers = [] # list storing the period of the each plot in drawn_perplts
        self.perpts = []
        self.per_plot = None
        self.zoompt1 = None
        self.zoompt2 = None
        self.srect_plot = None # Plot of selection rectangle for zooming

        

        # Instantiate plot and tk widget
        self.plot_fig = Figure()
        self.plot_fig.subplots_adjust(left=.05,right=.99,top=.95,bottom=.1)
        self.plot_canvas = FigureCanvasTkAgg(self.plot_fig,master=self.main_window)

        # Set up plot
        self.plot_ax  = self.plot_fig.gca() 
        if self.fract.is_julia:
            self.plot_ax.set_title('Julia Set')
        else:
            self.plot_ax.set_title('Parameter Space')
        self.plot_ax.set_xlabel('Re')
        self.plot_ax.set_ylabel('Im')

        # Make fractal plot
        cmap = plt.cm.get_cmap("twilight_shifted")#, self.fract.get_N()+1)
        self.fractal_grid = self.fract.get_fract(self.xrng,self.yrng,self.res)
        self.fplot = self.plot_ax.imshow(self.fractal_grid, extent=self.get_extent(),
                                         interpolation="gaussian",cmap=cmap)
        self.fplot.set_clim(1,self.fract.get_N()+1)
        self.update_axes()

        # Set up drawing of first several iterates, starting at mouse position and blit
        self.iplt,=self.plot_ax.plot([],[],'-ok',animated=True,ms=self.iplt_ms,
                                     lw=self.iplt_lw,alpha=self.iplt_alpha)
        self.plot_canvas.draw()
        plt.pause(.1)
        self.bg = self.plot_canvas.copy_from_bbox(self.plot_ax.bbox)
        self.plot_fig.draw_artist(self.iplt)
        self.plot_canvas.blit(self.plot_ax.bbox)
        self.fig_size = self.plot_fig.get_size_inches()

    
        # Set up event listeners
        self.click_listener = self.plot_fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.move_listener = self.plot_fig.canvas.mpl_connect('motion_notify_event', self.on_move)
        self.enter_plot_listener = self.plot_fig.canvas.mpl_connect('axes_enter_event', self.on_enter_plot)


    def draw_plot(self):
        self.plot_canvas.draw()
        plt.show(block=False)
        plt.pause(.1)
        self.bg = self.plot_fig.canvas.copy_from_bbox(self.plot_ax.bbox)
        self.plot_fig.canvas.blit(self.plot_ax.bbox)

    def update_plot(self):
        self.update_axes()
        self.update_extent()
        self.fract_grid = self.fract.get_fract(self.xrng,self.yrng,self.res)
        self.fplot.set_data(self.fract_grid)
        self.fplot.set_clim(1,self.fract.get_N()+1)
        self.clear_plot()
        self.draw_plot()

    def clear_plot(self):
        self.clear_iterate_plot()
        self.clear_period_plot()

    def clear_iterate_plot(self):
        if len(self.drawn_iplts)>0:
            for drawn_iplt in self.drawn_iplts:
                drawn_iplt[0].remove()
            self.drawn_iplts = []
        self.ipts = []
        self.update_axes()
        self.update_extent()
        self.draw_plot()

    def clear_period_plot(self):
        self.perpts = []
        if len(self.drawn_perplts)>0:
            for drawn_perplt in self.drawn_perplts:
                drawn_perplt.remove()
            self.drawn_perplts = []
            self.drawn_perplt_pers = []
            self.perpts = []
            self.per_plt_legend.remove()
        self.update_axes()
        self.update_extent()
        self.draw_plot()

    def update_iterates(self,event):
        # Detect window resize, so blitting can be reset
        if (self.fig_size != self.plot_fig.get_size_inches()).any():
            self.bg = self.plot_fig.canvas.copy_from_bbox(self.plot_ax.bbox)
            self.fig_size = self.plot_fig.get_size_inches()
        self.nits = int(self.parent.nits_input_var.get())
        self.ipts=self.fract.f_nits(self.nits,event.xdata+event.ydata*1j)
        self.plot_fig.canvas.restore_region(self.bg)
        self.iplt.set_xdata(np.real(self.ipts))
        self.iplt.set_ydata(np.imag(self.ipts))
        self.plot_fig.draw_artist(self.iplt)
        self.plot_fig.canvas.blit(self.plot_ax.bbox)
        self.plot_fig.canvas.flush_events()

    def update_extent(self):
        self.fplot.set_extent(self.get_extent())

    def update_axes(self):
        self.plot_ax.set_xlim(self.xrng[0],self.xrng[1])
        self.plot_ax.set_ylim(self.yrng[0],self.yrng[1])


    def on_click(self,event):
        if event.inaxes:
            if event.button == 1: # left click
                # Zoom region selection
                if kb.is_pressed("ctrl"):
                    if self.zoompt1 is None:
                        self.zoompt1=[event.xdata,event.ydata]
                        self.srect_plot, = self.plot_ax.plot(event.xdata,event.ydata,
                                                             '-ok',animated=True,ms=1,lw=.5)
                        self.plot_fig.draw_artist(self.srect_plot)
                        self.plot_canvas.blit(self.plot_ax.bbox)
                    else:
                        self.zoompt2 = [event.xdata,event.ydata]
                        self.xrng    = sorted([self.zoompt1[0],self.zoompt2[0]])
                        self.yrng    = sorted([self.zoompt2[1],self.zoompt1[1]])
                        self.zoompt1 = None
                        self.zoompt2 = None
                        self.parent.update_range_text()
                        self.update_plot()
                # Paint current plot of iterates to plot
                if  self.zoompt1 is None and kb.is_pressed("shift"):
                    self.drawn_iplts.append(self.plot_ax.plot(np.real(self.ipts),np.imag(self.ipts),
                                                              '-ok',ms=self.iplt_ms,lw=self.iplt_lw,
                                                              alpha=self.iplt_alpha))
                    self.bg = self.plot_fig.canvas.copy_from_bbox(self.plot_ax.bbox)
            elif event.button == 3 and (not self.fract.is_julia): # right click
                # ask if user wants to generate julia set plot at this value of c
                if event.ydata<0:
                    msg_txt="Generate a Julia set at c="+"{:.3f}".format(event.xdata)+"-"+"{:.3f}".format(abs(event.ydata))+"i?"
                else:
                    msg_txt="Generate a Julia set at c="+"{:.3f}".format(event.xdata)+"+"+"{:.3f}".format(event.ydata)+"i?"
                draw_julia = messagebox.askyesno("Generate Julia Set?", msg_txt)
                if draw_julia:
                    self.parent.add_child_plot(event.xdata+event.ydata*1j,xrng=[-1.5,1.5],yrng=[-1.5,1.5])


    # Attempt to find points of period self.per, and plot them
    def draw_per_plot(self):
        if (not (self.per in self.drawn_perplt_pers)):
            self.perpts = self.fract.find_per_orbits(self.xrng,self.yrng,self.search_res,self.per)
            if len(self.drawn_perplt_pers) == 0:
                insert_index = 0
            elif self.per>self.drawn_perplt_pers[-1]:
                insert_index = len(self.drawn_perplt_pers)
            else:
                insert_index = next(i for i,v in enumerate(self.drawn_perplt_pers) if self.per<v)
            self.drawn_perplts.insert(insert_index,
                                      self.plot_ax.scatter(np.real(self.perpts),np.imag(self.perpts),label=self.per,s=self.perplt_ms))
            self.drawn_perplt_pers.insert(insert_index,self.per)
            self.per_plt_legend = self.plot_ax.legend(
                self.drawn_perplts,["period "+str(p) for p in self.drawn_perplt_pers],frameon=False,framealpha=1,bbox_to_anchor=(1.05, 1.0), loc='upper left',title="Periodic\n Points")
            plt.pause(.1)
            self.bg = self.plot_fig.canvas.copy_from_bbox(self.plot_ax.bbox)
            self.draw_plot()


    # Show sequence of first niter iterates, starting at the mouse position
    # Only do so when the "shift" key is held down
    def on_move(self,event):
        if event.inaxes and kb.is_pressed("shift"):
            self.update_iterates(event)
        elif len(self.ipts) != 0:
            self.iplt.set_xdata([])
            self.iplt.set_ydata([])
            self.draw_plot()
        if event.inaxes and kb.is_pressed("ctrl") and not self.zoompt1 is None:
            self.plot_fig.canvas.restore_region(self.bg)
            self.srect_plot.set_xdata([self.zoompt1[0],event.xdata,    event.xdata,self.zoompt1[0],
                                       self.zoompt1[0]])
            self.srect_plot.set_ydata([self.zoompt1[1],self.zoompt1[1],event.ydata,event.ydata,
                                       self.zoompt1[1]])
            self.plot_fig.draw_artist(self.srect_plot)
            self.plot_fig.canvas.blit(self.plot_ax.bbox)
            self.plot_fig.canvas.flush_events()

            
            
 
    # Event triggered when plot is entered
    def on_enter_plot(self,event):
        if event.inaxes is self.plot_ax:
            self.plot_fig


    # Getters and setters
    def get_canvas(self):
        return self.plot_canvas

    def get_widget(self):
        return self.plot_canvas.get_tk_widget()
    
    def get_extent(self):
        return [self.xrng[0],self.xrng[1],self.yrng[1],self.yrng[0]]

    def set_plot_title(self,title):
        self.plot_ax.set_title(title)

    def set_xlabel(self,xlabel):
        self.plot_ax.set_xlabel(xlabel)

    def set_ylabel(self,ylabel):
        self.plot_ax.set_ylabel(ylabel)

    def set_xrng(self,xrng):
        self.xrng = xrng

    def get_xrng(self):
        return self.xrng

    def set_yrng(self,yrng):
        self.yrng = yrng

    def get_yrng(self):
        return self.yrng

    def set_cmap(self,cmap):
        self.fplot.set_cmap(cmap)

    def set_interp(self,interp):
        self.fplot.set_interpolation(interp)

    def set_period(self,period):
        self.per = period

    def get_period(self):
        return self.per

    def set_nits(self,nits):
        self.nits = nits

    def get_nits(self):
        return self.nits

    def set_res(self,res):
        self.res = res

    def get_res(self):
        return self.res
    
    def set_search_res(self,search_res):
        self.search_res = search_res

    def get_search_res(self):
        return self.search_res

    def set_N(self,N):
        self.fract.set_N(N)

    def get_N(self):
        return self.fract.get_N()

    def set_c(self,c):
        self.fract.set_c(c)

    def get_c(self):
        return self.fract.get_c()

    def set_f(self,f):
        self.fract.set_f(f)

    def get_f(self):
        return self.fract.get_f()

    def set_exitCond(self,exitCond):
        self.fract.set_exitCond(exitCond)

    def get_exitCond(self):
        return self.fract.get_exitCond()


















