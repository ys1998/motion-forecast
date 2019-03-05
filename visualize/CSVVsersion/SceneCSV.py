import vtk

class SceneCSV(object):
    def __init__(self):

        self.sceneSources = list()
        self.sceneMappers = list()
        self.sceneActors = list()
        self.sceneLights = list()

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
