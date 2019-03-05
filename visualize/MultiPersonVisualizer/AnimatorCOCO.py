# import zmq
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
        self.frame = 7
        self.sub =  5# [19,10, 14, 20
        self.n_frames = 4000  #[430,251, 625, 500
        self.savePath = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS'+str(self.sub)+'/'
        self.savePath = '/media/anurag/WareHouse/datasets/coco_vis/'
        self.Skeleton = Skeleton
        self.edges = [[0,10], [1,9], [2,8], [3,11], [4,12], [5,13],\
                    [6,14], [7,15], [8,1], [9,16], [10,4], [11,3], [12,2],\
                    [13,5], [14,6], [15,7]]

    def animate(self, obj, event):
        renWin = obj.GetRenderWindow()
        if event == "TimerEvent":
            #[address, contents] = self.subscriber.recv_multipart()
            #contents = str(contents,'utf-8')

            # self.frame += 1
            print(self.frame)
            path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mupots_preds_hg/exp/3D_matched_abs/'
            file_name = 'image_' + str(self.frame)
            file_name = '000000481390'
            file_name = '000000013291'
            path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mscoco/' + file_name + '.csv'
            path_2d = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mscoco/' + file_name + '_2d.csv'
            # path = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS9/'
            # poses = np.genfromtxt(path + 'sub_'+str(self.sub)+'_frame_' + str(self.frame % self.n_frames) + '.csv', delimiter=',')
            poses = np.genfromtxt(path, delimiter=',')
            poses_2d = np.genfromtxt(path_2d, delimiter=',')

            poses = poses.reshape(-1, 48)
            poses_2d = poses_2d.reshape(-1, 32)

            abs_poses, _ = self.get_absolute_pose(poses.reshape(-1,16,3), poses_2d.reshape(-1,16,2), 1000)
            poses = abs_poses.reshape(-1,48)

            print(poses[0, 15:18])
            n_people = min(poses.shape[0], len(self.Skeleton))
            # n_people = 3
            # row = row[:-1]
            print(n_people)
            for j in range(n_people):
                row_ = poses[j, :]
                # if j == 0:
                #     row_ = row_ * 100
                # row_np = np.array(row_).reshape(17,3)
                # row_h = np.zeros((16,3))
                # loc = np.array([j*2000, 0, 0])
                # for i in range(len(self.edges)):
                #     h = self.edges[i][0]
                #     m = self.edges[i][1]
                #     row_h[h]  = row_np[m]
                row_ = self.skeletonFitting(row_.reshape(16,3))
                # row_ = row_.reshape(16,3) + loc.reshape(1,3)
                row_ = list(row_.reshape(48))
                self.Skeleton[j].updateLimbs(row_)

        renWin.Render()
        # self.takeScreenShot(renWin)

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

    def get_absolute_pose(self, return_3d, save_2d, height):
        f = 334 / (2 * np.tan(np.deg2rad(27)))
        # f = 500
        edges = [[0,1], [1,2], [2,6], [8,12], [12,11], [11,10], [6,3], [3,4], \
                        [4,5], [8,13], [13,14], [14,15], [6,7], [7,8], [8,9]]
        # Indexed with 1. Be careful.
        # edges_mpi = [[1,17], [17,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8],\
        #             [2,16], [16,15], [15,9], [9,10], [10,11], [15,12], [12,13], [13,14]]
        # edges = [[15,16], [16,2], [2,1]]
        root = 6
        n_people = return_3d.shape[0]
        for p in range(n_people):
            sbl2d, sbl3d = 0, 0
            for e in range(len(edges)):
                j1 = edges[e][0] #- 1
                j2 = edges[e][1] #- 1
                sbl2d += (((save_2d[p, j1, :] - save_2d[p, j2, :])**2).sum())**0.5
                sbl3d += (((return_3d[p, j1, :] - return_3d[p, j2, :])**2).sum())**0.5
            Z = f * sbl3d / sbl2d
            X = (save_2d[p, root, 0] - 250) * sbl3d / sbl2d
            # Y = (save_2d[p, root, 1] - 640) * sbl3d / sbl2d
            p_root = np.array([X, 0.0, Z]).astype('float32')
            return_3d[p] += p_root
        return_3d = return_3d.reshape(-1, 16*3)
        return return_3d, save_2d


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
