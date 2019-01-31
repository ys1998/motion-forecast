import vtk
import time
from datetime import datetime

class Visualizer(object):
    def __init__(self, Skeleton, Scene, Animator, idx):
        """
            Skeleton will be a list
        """
        self.ren = vtk.vtkRenderer()
        self.renWin = vtk.vtkRenderWindow()
        self.renWin.AddRenderer(self.ren)
        self.iren = vtk.vtkRenderWindowInteractor()
        self.iren.SetRenderWindow(self.renWin)
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(self.style)
        self.ren.GradientBackgroundOn()
        self.ren.SetBackground(0, 0, 0)
        self.ren.SetBackground2(0, 0, 0)
        self.renWin.SetSize(800, 800)


        self.addSkeleton(Skeleton)
        self.addScene(Scene)

        self.cameraP = vtk.vtkCameraPass()
        self.lights = vtk.vtkLightsPass()
        self.shadowsBaker = vtk.vtkShadowMapBakerPass()
        self.shadowsBaker.SetResolution(4096)

        self.shadows = vtk.vtkShadowMapPass()
        self.shadows.SetShadowMapBakerPass(self.shadowsBaker)

        self.seq = vtk.vtkSequencePass()
        self.passes = vtk.vtkRenderPassCollection()
        self.passes.AddItem(self.shadowsBaker)
        self.passes.AddItem(self.shadows)
        self.passes.AddItem(self.lights)
        self.seq.SetPasses(self.passes)
        self.cameraP.SetDelegatePass(self.seq)

        self.ren.SetPass(self.cameraP)
        self.iren.Initialize()
        self.ren.ResetCamera()
        self.ren.GetActiveCamera().ParallelProjectionOff()
        self.ren.GetActiveCamera().Azimuth(180)
        self.ren.GetActiveCamera().Pitch(1)
        self.ren.GetActiveCamera().Elevation(20)
        self.ren.GetActiveCamera().SetViewAngle(55)
        self.renWin.Render()

        self.idx = idx
        self.frame_time = 8
        self.iren.AddObserver("KeyPressEvent",self.keyPress)
        self.iren.AddObserver('TimerEvent', Animator.animate)
        self.timerId = self.iren.CreateRepeatingTimer(self.frame_time)
        self.iren.Start()
        print(self.idx)

    def addSkeleton(self, Skeleton):
        for i in range(len(Skeleton.sphereActors)):
            self.ren.AddActor(Skeleton.sphereActors[i])

        for i in range(len(Skeleton.ellipsoidActors)):
            self.ren.AddActor(Skeleton.ellipsoidActors[i])

    def addScene(self, Scene):
        for i in range(len(Scene.sceneActors)):
            self.ren.AddActor(Scene.sceneActors[i])

        for i in range(len(Scene.sceneLights)):
            self.ren.AddLight(Scene.sceneLights[i])

    def takeScreenShot(self):
        fileName = datetime.now().strftime('/%Y-%m-%d %H.%M.%S.%f')
        w2if = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName("../MPII"+ self.idx  + fileName + ".png")
        w2if.SetInput(self.renWin)
        w2if.Update()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()

    def keyPress(self, obj, event):
        key = obj.GetKeySym()
        print("key %s" % key)
        self.iren.Disable()

        if key == "t":
            for i in range(0, 60, 1):
                print(i)
                self.ren.GetActiveCamera().Azimuth(3)
                self.renWin.Render()
                time.sleep(0.5)
                self.takeScreenShot()
                time.sleep(0.5)
        elif key == "s":
            self.takeScreenShot()
        elif key == "p":
            self.iren.DestroyTimer(self.timerId)
        elif key == "c":
            self.timerId = self.iren.CreateRepeatingTimer(self.frame_time)

        self.iren.Initialize()
        self.iren.Enable()

