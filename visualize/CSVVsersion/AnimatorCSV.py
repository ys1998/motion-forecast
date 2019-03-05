import zmq
import vtk
import csv
from datetime import datetime

class AnimatorCSV(object):
    def __init__(self, Skeleton, jointsFile, saveFolder):
        #self.context = zmq.Context()
        #self.subscriber = self.context.socket(zmq.SUB)
        #self.subscriber.connect(ConnectionString)
        #self.subscriber.setsockopt(zmq.SUBSCRIBE, Channel)

        self.csvfile = open(jointsFile)
        self.reader = csv.reader(self.csvfile)
        self.savePath = saveFolder

        self.Skeleton = Skeleton

    def animate(self, obj, event):
        renWin = obj.GetRenderWindow()
        if event == "TimerEvent":
            #[address, contents] = self.subscriber.recv_multipart()
            #contents = str(contents,'utf-8')

            row = next(self.reader)
            #print(len(row))

            # row = row[:-1]
            if len(row) == 48:
                self.Skeleton.updateLimbs(row)

        renWin.Render()
        self.takeScreenShot(renWin)

    def takeScreenShot(self, renWin):
        fileName = datetime.now().strftime('%Y-%m-%d %H.%M.%S.%f')
        w2if = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(self.savePath + fileName + ".png")
        w2if.SetInput(renWin)
        w2if.Update()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
