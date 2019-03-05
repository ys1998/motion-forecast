import vtk

class Scene(object):
    def __init__(self):

        self.sceneSources = list()
        self.sceneMappers = list()
        self.sceneActors = list()
        self.sceneLights = list()

        self.sceneSources.append(vtk.vtkCubeSource())
        self.sceneSources[-1].SetXLength(50000)
        self.sceneSources[-1].SetYLength(50000)
        self.sceneSources[-1].SetZLength(5)

        # self.sceneMappers.append(vtk.vtkPolyDataMapper())
        # self.sceneMappers[-1].SetInputConnection(self.sceneSources[-1].GetOutputPort())


        reader = vtk.vtkJPEGReader()
        reader.SetFileName("blackandwhite.jpg")
        # reader.SetFileName("white.jpg")

        # Create texture object
        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.RepeatOn()

        #Map texture coordinates
        map_to_plane = vtk.vtkTextureMapToPlane()
        map_to_plane.SetInputConnection(self.sceneSources[-1].GetOutputPort())

        # Create mapper and set the mapped texture as input
        mapperplane = vtk.vtkPolyDataMapper()
        mapperplane.SetInputConnection(map_to_plane.GetOutputPort())

        self.sceneActors.append(vtk.vtkActor())
        self.sceneActors[-1].RotateX(90)
        self.sceneActors[-1].SetPosition(1300,-800,2500) # -1200
        self.sceneActors[-1].SetMapper(mapperplane)
        self.sceneActors[-1].SetTexture(texture)
        # self.sceneActors[-1].GetProperty().SetColor(1,1,1)

        self.addLight(1.0, 1.0, 1.0, 1000, 1000, -1000, 0.75, 180, 0.75)
        self.addLight(1.0, 1.0, 1.0, -1000, 500, 1000, 0.5, 180, 0.0)
        self.addLight(1.0, 1.0, 1.0, -1000, 500,- 1000, 0.5, 180, 0.0)

    def addLight(self, cR, cG, cB, pX, pY, pZ, Intensity, ConeAngle, Attenuation):
        self.sceneLights.append(vtk.vtkLight())
        self.sceneLights[-1].SetColor(cR, cG, cB)
        self.sceneLights[-1].SetPosition(pX, pY, pZ)
        self.sceneLights[-1].SetIntensity(Intensity)
        self.sceneLights[-1].SetConeAngle(ConeAngle)
        self.sceneLights[-1].SetShadowAttenuation(Attenuation)
        self.sceneLights[-1].SetLightTypeToSceneLight()
