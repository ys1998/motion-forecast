from Scene import *
from Visualizer import *
from Skeleton import *
from Animator import *
import sys, os
import subprocess

if __name__ == "__main__":
    cs = "tcp://localhost:5563"
    # os.system('mkdir -p ../MPII' + sys.argv[1])
    c = b"B"
    sk1 = Skeleton()
    sc1 = Scene()
    an1 = Animator(sk1, cs, c)
    if sys.argv[1]:
        viz = Visualizer(sk1, sc1, an1, sys.argv[1])
    else:
        viz = Visualizer(sk1, sc1, an1, None)
