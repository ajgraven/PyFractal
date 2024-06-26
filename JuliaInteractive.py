# To Compile:
# (1) Navigate to directory containing this file in the terminal
# (2) run the command "set PYTHONOPTIMIZE=2"
# (3) run the command "PyInstaller --windowed --onefile JuliaInteractive.py"

# Instructions:
#  - Hold down shift and hover your mouse over the plot to see the first several iterates starting at the mouse position
#  - left click to paint the current plot of iterates to the screen
#  - ctrl+click twice to select a region to zoom in to
#  - right click to generate Julia set with c = that point
#  - multiple different parameters may be edited imsave via the text boxes provided
#  - press the update plot button to apply changed parameters

# TODO:
# - Add control over maxmod, niter, bounds of window
# - add parallelization support
# - add method of ploting number of iterates to satisfy a given condition 
# (i.e. entering a certian region)
# - Avoid complete recomputation if only changing color scheme or interpolation.

#from mpmath import *
import PyFractalUtils as pf
import tkinter as tk
import time
import numpy as np
from numpy import arange,exp


############################################
########## Set default parameters ##########
############################################

# Mandelbrot Set
N=35
xrng=[-2,.6]
yrng=[-1.25,1.25]
res=100000
c=1
f_text="z**2+c"
exitCond_text="abs(z)>2"




# Generate main menu window
#main_menu = tk.Tk()



#f_text="z**2+c"
#exitCond_text="abs(z)>2"

main_window = tk.Tk()
dynPlot = pf.DynamicalPlot(main_window,True,N=N,res=res,xrng=xrng,yrng=yrng,f_text=f_text,exitCond_text=exitCond_text)

main_window.mainloop()

'''
class MainMenu:
    def __init__(self,master):
        self.main_window = master
        main_menu.title("Fractal Visualizer Main Menu")
        self.load_fractal_button = tk.Button(self.main_window,
                             text="load fractal",command=self.load_fractal)
        self.new_fractal_button  = tk.Button(self.main_window,
                                text="new fractal",command=self.new_fractal)



    def load_fractal(self):

'''
        

        









