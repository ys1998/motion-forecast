import vtk
import math
import random
from vtk.util.colors import white as color

class Skeleton(object):
    def __init__(self):
        self.sphereSources = list()
        self.sphereMappers = list()
        self.sphereActors = list()

        self.ellipsoidPara = list()
        self.ellipsoidSources = list()
        self.ellipsoidMappers = list()
        self.ellipsoidActors = list()
        self.size1 = 25
        self.size2 = 35
        self.size3 = 60
        self.idxLimbs = [0, 1, 2, 3, 4, 5,
                    3, 4, 5, 6, 7, 8,
                    6, 7, 8, 18, 19, 20,  
                    9, 10, 11, 18, 19, 20,  
                    9, 10, 11, 12, 13, 14,  
                    15, 16, 17, 12, 13, 14,  
                    18, 19, 20, 21, 22, 23, 
                    24, 25, 26, 21, 22, 23,  
                    24, 25, 26, 36, 37, 38,  
                    36, 37, 38, 33, 34, 35,  
                    33, 34, 35, 30, 31, 32,  
                    24, 25, 26, 39, 40, 41,  
                    39, 40, 41, 42, 43, 44,  
                    42, 43, 44, 45, 46, 47,  
                    24, 25, 26, 27, 28, 29]
        self.defaultPose = [173.36344909668,889.08557128906,210.82315063477,136.16815185547,439.6711730957,
                            134.53231811523,119.46765136719,-15.016092300415,39.336120605469,-120.73154449463,
                            13.54674911499,-40.604217529297,-134.08387756348,469.61880493164,50.811546325684,
                            -176.72409057617,920.24377441406,134.7225189209,0,0,0,1.4916274547577,-246.97732543945,
                            -83.619606018066,10.143627166748,-494.11926269531,-145.40014648438,23.438856124878,
                            -694.08026123047,-135.14549255371,629.7666015625,-413.57543945312,169.4983215332,
                            443.78927612305,-418.60272216797,30.647079467773,171.22592163086,-469.2080078125,
                            -118.32019805908,-136.05972290039,-460.2502746582,-212.36334228516,-417.2038269043,
                            -387.82623291016,-267.94668579102,-645.6337890625,-368.27423095703,-256.57174682617]

        j = 0
        for i in range(0, 48, 3):
            self.sphereSources.append(vtk.vtkSphereSource())
            self.sphereSources[j].SetThetaResolution(12)
            self.sphereSources[j].SetPhiResolution(12)
            self.sphereSources[j].SetRadius(35)

            self.sphereMappers.append(vtk.vtkPolyDataMapper())
            self.sphereMappers[j].SetInputConnection(self.sphereSources[j].GetOutputPort())

            self.sphereActors.append(vtk.vtkActor())
            self.sphereActors[j].SetMapper(self.sphereMappers[j])
            self.sphereActors[j].GetProperty().SetColor(color)
            self.sphereActors[j].SetPosition(-float(self.defaultPose[i]), -float(self.defaultPose[i+1]), float(self.defaultPose[i+2]))
            j = j + 1

        self.createLimb(self.defaultPose)
        
    def extractStartEnd(self, listJoints):
        listLimbPoints = []

        for i in range(0, len(self.idxLimbs), 6):
            t = [float(listJoints[self.idxLimbs[i]]), 
                float(listJoints[self.idxLimbs[i+1]]), 
                float(listJoints[self.idxLimbs[i+2]]), 
                float(listJoints[self.idxLimbs[i+3]]), 
                float(listJoints[self.idxLimbs[i+4]]), 
                float(listJoints[self.idxLimbs[i+5]])]
            listLimbPoints.append(t)
        return listLimbPoints
    
    def calcTransform(self, listLimbPoints):

        startPoint = [0 for i in range(3)]
        endPoint = [0 for i in range(3)]
        center = [0 for i in range(3)]
        normalizedX = [0 for i in range(3)]
        normalizedY = [0 for i in range(3)]
        normalizedZ = [0 for i in range(3)]
        math = vtk.vtkMath()
        arbitrary = [0 for i in range(3)]
        matrix = vtk.vtkMatrix4x4()
        arbitrary[0] = random.uniform(-10, 10)
        arbitrary[1] = random.uniform(-10, 10)
        arbitrary[2] = random.uniform(-10, 10)

        transform = vtk.vtkTransform()
        startPoint[0] = -listLimbPoints[0]
        startPoint[1] = -listLimbPoints[1]
        startPoint[2] = listLimbPoints[2]

        endPoint[0] = -listLimbPoints[3]
        endPoint[1] = -listLimbPoints[4]
        endPoint[2] = listLimbPoints[5]

        center[0] = (endPoint[0] + startPoint[0])/2
        center[1] = (endPoint[1] + startPoint[1])/2
        center[2] = (endPoint[2] + startPoint[2])/2

        math.Subtract(endPoint, startPoint, normalizedX)
        length = math.Norm(normalizedX)
        math.Normalize(normalizedX)


        math.Cross(normalizedX, arbitrary, normalizedZ)
        math.Normalize(normalizedZ)

        math.Cross(normalizedZ, normalizedX, normalizedY)

        matrix.Identity()
        for i in range(3):
            matrix.SetElement(i, 0, normalizedY[i])
            matrix.SetElement(i, 1, normalizedX[i])
            matrix.SetElement(i, 2, normalizedZ[i])

        #transform.PostMultiply()
        transform.Translate(center)
        transform.Concatenate(matrix)
        transform.Scale(1, -1, 1)

        return (transform, length)

    def createLimb(self, listJoints):

        listLimbPoints = self.extractStartEnd(listJoints)

        for i in range(15):
            transform, length = self.calcTransform(listLimbPoints[i])
            self.ellipsoidPara.append(vtk.vtkParametricEllipsoid())
            self.ellipsoidPara[-1].SetXRadius(self.size1)
            self.ellipsoidPara[-1].SetZRadius(self.size1)
            self.ellipsoidPara[-1].SetYRadius(length/2)

            self.ellipsoidSources.append(vtk.vtkParametricFunctionSource())
            self.ellipsoidSources[-1].SetParametricFunction(self.ellipsoidPara[-1])

            self.ellipsoidMappers.append(vtk.vtkPolyDataMapper())
            self.ellipsoidMappers[-1].SetInputConnection(self.ellipsoidSources[-1].GetOutputPort())

            self.ellipsoidActors.append(vtk.vtkActor())
            self.ellipsoidActors[-1].SetMapper(self.ellipsoidMappers[-1])

            self.ellipsoidActors[-1].SetUserTransform(transform)
            self.ellipsoidActors[-1].GetProperty().SetColor(color)

        self.ellipsoidPara[0].SetXRadius(self.size3)
        self.ellipsoidPara[0].SetZRadius(self.size3)

        self.ellipsoidPara[1].SetXRadius(self.size3)
        self.ellipsoidPara[1].SetZRadius(self.size3)

        self.ellipsoidPara[4].SetXRadius(self.size3)
        self.ellipsoidPara[4].SetZRadius(self.size3)

        self.ellipsoidPara[5].SetXRadius(self.size3)
        self.ellipsoidPara[5].SetZRadius(self.size3)

        self.ellipsoidPara[9].SetXRadius(self.size2)
        self.ellipsoidPara[9].SetZRadius(self.size2)

        self.ellipsoidPara[10].SetXRadius(self.size2)
        self.ellipsoidPara[10].SetZRadius(self.size2)

        self.ellipsoidPara[12].SetXRadius(self.size2)
        self.ellipsoidPara[12].SetZRadius(self.size2)

        self.ellipsoidPara[13].SetXRadius(self.size2)
        self.ellipsoidPara[13].SetZRadius(self.size2)

    def updateLimbs(self, listJoints):
        j = 0
        for i in range(0, 48, 3):
            self.sphereActors[j].SetPosition(-float(listJoints[i]), -float(listJoints[i+1]), float(listJoints[i+2]))
            j = j + 1

        listLimbPoints = self.extractStartEnd(listJoints)

        for i in range(15):
            transform, length = self.calcTransform(listLimbPoints[i])
            self.ellipsoidActors[i].SetUserTransform(transform)