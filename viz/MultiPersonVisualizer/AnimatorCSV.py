import vtk
# import csv
# from datetime import datetime
import numpy as np

class AnimatorCSV(object):
    def __init__(self, Skeleton, jointsFile, saveFolder):
        #self.context = zmq.Context()
        #self.subscriber = self.context.socket(zmq.SUB)
        #self.subscriber.connect(ConnectionString)
        #self.subscriber.setsockopt(zmq.SUBSCRIBE, Channel)

        # self.csvfile = open(jointsFile)
        # self.reader = csv.reader(self.csvfile)
        self.savePath = saveFolder
        self.frame = 0
        self.sub = 5 # [19,10, 14, 20, 12
        self.n_frames = 261 #[430,251, 625, 500, 250
        self.savePath = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS'+str(self.sub)+'/'
        # self.savePath = '/media/anurag/WareHouse/datasets/coco_vis/'
        self.Skeleton = Skeleton
        self.edges = [[0,10], [1,9], [2,8], [3,11], [4,12], [5,13],\
                    [6,14], [7,15], [8,1], [9,16], [10,4], [11,3], [12,2],\
                    [13,5], [14,6], [15,7]]
        self.counter = 0
        self.context = 10
        self.framebuff = np.zeros((500,10,48))

    def animate(self, obj, event):
        renWin = obj.GetRenderWindow()
        if event == "TimerEvent":
            #[address, contents] = self.subscriber.recv_multipart()
            #contents = str(contents,'utf-8')

            self.frame += 1
            self.counter += 1
            print(self.frame)
            path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mupots_preds_hg/exp/3D_matched_abs/'
            # path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mupots_preds_hg/exp/3D_interp/'
            # path = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS9/'
            poses = np.genfromtxt(path + 'sub_'+str(self.sub)+'_frame_' + str(self.frame % self.n_frames) + '.csv', delimiter=',')

            poses = poses.reshape(-1, 51)
            n_people = poses.shape[0]
            n_people = 2
            self.n_people = n_people
            print(n_people)
            rows = np.zeros((n_people, 48))
            # row = row[:-1]
            for j in range(n_people):
                row_ = poses[j, :]
                row_np = np.array(row_).reshape(17,3)
                row_h = np.zeros((16,3))
                # loc = np.array([j*2000, 0, 0])
                for i in range(len(self.edges)):
                    h = self.edges[i][0]
                    m = self.edges[i][1]
                    row_h[h]  = row_np[m]
                # row_h = row_h + loc.reshape(1,3)
                # row_h = row_.reshape(16,3) + [100*j, 0, 100*j]
                # row_h = self.skeletonFitting(row_h.reshape(16,3))
                # row_ = list(row_h.reshape(48))
                rows[j] = row_h.reshape(48)
                # self.Skeleton[j].updateLimbs(row_)
            for j in range(n_people):
                # row_ = self.smoothPose(rows)
                row_ = rows
                row_ = list(row_[j])
                self.Skeleton[j].updateLimbs(row_)
        renWin.Render()
        # self.takeScreenShot(renWin)

    def smoothPose(self, poses):
        ct = min(self.counter, 3) # 5 is the context size
        i = self.counter
        self.framebuff[i,:self.n_people] = poses
        output = self.framebuff[i-ct: i+1].mean(0)
        return output

    def takeScreenShot(self, renWin):
        # fileName = datetime.now().strftime('%Y-%m-%d %H.%M.%S.%f')
        fileName = 'image_' + str(self.frame % self.n_frames)
        w2if = vtk.vtkWindowToImageFilter()
        writer = vtk.vtkPNGWriter()
        writer.SetFileName(self.savePath + fileName + ".png")
        w2if.SetInput(renWin)
        w2if.Update()
        writer.SetInputConnection(w2if.GetOutputPort())
        writer.Write()
        self.frame += 1

    def skeletonFitting(self, pose):
        tree = [[6,-1,0], [7,6,239.7], [8,7,254.4], [9,8,178.07],
              [12,8,150.85], [11,12,279.66], [10,11,246.43], [13,8,150.85],
              [14,13,279.66], [15,14,246.43], [2,6,139.6], [1,2,448.04],
              [0,1,436.62], [3,6,139.59], [4,3,448.08], [5,4,436.62]
              ]
        output = pose.copy()
        for j in range(len(tree)):
            parent = tree[j][1]
            joint = tree[j][0]
            bone_length = tree[j][2]
            if parent == -1:
                output[joint] = pose[joint]
            else:
                vecnorm = (pose[joint] - pose[parent]) / np.linalg.norm(pose[joint]-pose[parent])
                output[joint] = output[parent] + bone_length * vecnorm

        return output
