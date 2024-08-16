# Calcium_OF_Analysis

In this repository, I provide a few scripts that I used to analyze and visualize calcium imaging data. The scripts are not fully polished, but functional(ish). 

**OpenField_Analysis.py**
This script is intended for the analysis of open field experiment data. It includes functions for loading, processing, and analyzing the data. 
- Firing rate
- PSTH
- Place cells
- speed cells
- etc. 

**GUI_Explore_OpenField_Data.py**
This script provides a graphical user interface for exploring open-field data. It is built with qt. The GUI is still a work in progress, but I hope it gives a idea of what I am trying to achieve.

**data/E8KO_1234**
This file is an example dataset that can be used to test the functionality of the GUI and the analysis script. It contains synthetic data (generated with [RatInABox toolbox](https://example.com) - RatInABox toolbox) structured similarly to typical open field experiment datasets.
Results_OLM_Ca_Events.pkl can be visualized with the GUI. 

Requirements : 
- pandas
- numpy 
- PyQt5
- pyqtgraph
- matplotlib
- pathlib
- scipy
- pynapple 
- tqdm
