import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from Elements import *
from solver import *
from functions import *
import cProfile
mesh = generateMesh_UnitSquare(1/8)
s= PoissonHCT(mesh)
s.PlotSolution()

