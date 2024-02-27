import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from Elements import *
from solver import *
from functions import *
import cProfile
mesh = generateMesh11square(1/16)
sol = BiharmHCT(mesh)
sol.PlotSolution()