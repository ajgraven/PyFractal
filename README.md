PyFractal: A fractal visualization tool

Dependencies: matplotlib, tkinter, numpy, cupy, pylab, scipy, pyinput, mouse, ipywidgets.
(use "pip install <package name>" to install a package)

To Compile:
(1) Navigate to directory containing this file in the terminal
(2) run the command "set PYTHONOPTIMIZE=2"
(3) run the command "pyinstaller --windowed --onefile JuliaInteractive.py"

Instructions:
 - Hold down shift and hover your mouse over the plot to see the first several iterates starting at the mouse position
 - left click to paint the current plot of iterates to the screen
 - ctrl+click twice to select a region to zoom in to
 - right click to generate Julia set with c = that point
 - multiple different parameters may be edited via the text boxes provided
 - press the update plot button to apply changed parameters
