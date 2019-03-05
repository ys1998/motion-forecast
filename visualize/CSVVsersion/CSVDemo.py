from SceneCSV import *
from VisualizerCSV import *
from Skeleton import *
from AnimatorCSV import *
import sys, os
import subprocess

if __name__ == "__main__":
    joints = "./csv_dump.csv"
    screenShotFolder = "./save/"
    sk1 = Skeleton()
    sc1 = SceneCSV()
    an1 = AnimatorCSV(sk1, joints, screenShotFolder)
    viz = VisualizerCSV(sk1, sc1, an1, screenShotFolder)