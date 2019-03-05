from Scene import *
from Visualizer import *
from Skeleton import *
from AnimatorCOCO import *
# from AnimatorTemporal import *
# from AnimatorCSV import *
import sys, os
import subprocess
import numpy as np

if __name__ == "__main__":
    cs = "tcp://localhost:5563"
    # os.system('mkdir -p ../MPII' + sys.argv[1])
    c = b"B"
    sk1 = [Skeleton()]
    for i in range(int(sys.argv[1])-1):
        sk1 += [Skeleton()]
    sc1 = Scene()
    # an1 = AnimatorCSV(sk1, cs, c)
    an1 = AnimatorCSV(sk1, cs, c)
    if sys.argv[1]:
        viz = Visualizer(sk1, sc1, an1, sys.argv[1])
    else:
        viz = Visualizer(sk1, sc1, an1, None)
