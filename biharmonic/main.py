import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from Elements import *
from solver import *
from functions import *
import cProfile
import time
mesh = generateMesh11square(1/8)
s = BiharmHCT(mesh)
s.Plot4eigenfunctions()