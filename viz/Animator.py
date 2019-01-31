import zmq
import vtk
from datetime import datetime

class Animator(object):
    def __init__(self, Skeleton, ConnectionString, Channel):
        self.context = zmq.Context()
        self.subscriber = self.context.socket(zmq.SUB)
        self.subscriber.connect(ConnectionString)
        self.subscriber.setsockopt(zmq.SUBSCRIBE, Channel)
        self.Skeleton = Skeleton

    def animate(self, obj, event):
        print(event)
        renWin = obj.GetRenderWindow()
        if event == "TimerEvent":
            [address, contents] = self.subscriber.recv_multipart()
            contents = str(contents,'utf-8')
            row = contents.split(",")[:-1]
            if len(row) == 48:
                self.Skeleton.updateLimbs(row)

        renWin.Render()

    def takeScreenShot(self):
        fileName = datetime.now().strftime('%Y-%m-%d %H.%M.%S.%f')
        w2if = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName("../Linear_912/"+ fileName + ".png")
        w2if.SetInput(self.renWin)
        w2if.Update()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()


