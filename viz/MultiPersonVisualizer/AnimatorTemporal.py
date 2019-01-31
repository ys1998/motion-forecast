# import zmq
import vtk
# import csv
# from datetime import datetime
import numpy as np
import pdb

class AnimatorCSV(object):
    def __init__(self, Skeleton, jointsFile, saveFolder):
        #self.context = zmq.Context()
        #self.subscriber = self.context.socket(zmq.SUB)
        #self.subscriber.connect(ConnectionString)
        #self.subscriber.setsockopt(zmq.SUBSCRIBE, Channel)

        # self.csvfile = open(jointsFile)
        # self.reader = csv.reader(self.csvfile)
        self.savePath = saveFolder
        self.frame = 4500
        self.sub =  5# [19,10, 14, 20
        self.n_frames = 4000  #[430,251, 625, 500
        self.savePath = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS'+str(self.sub)+'/'
        self.savePath = '/media/anurag/WareHouse/datasets/coco_vis/'
        self.Skeleton = Skeleton
        self.edges = [[0,10], [1,9], [2,8], [3,11], [4,12], [5,13],\
                    [6,14], [7,15], [8,1], [9,16], [10,4], [11,3], [12,2],\
                    [13,5], [14,6], [15,7]]
        self.counter = 0
        self.context = 2
        self.framebuff = np.zeros((500,5,48))
        self.framebuff2 = np.ones((500,5,32))

    def animate(self, obj, event):
        renWin = obj.GetRenderWindow()
        if event == "TimerEvent":
            #[address, contents] = self.subscriber.recv_multipart()
            #contents = str(contents,'utf-8')

            self.frame += 1
            self.counter += 1
            print(self.frame)
            path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/mupots_preds_hg/exp/3D_matched_abs/'
            file_name = '000000133969'
            file_name = 'dance3/image_' + str(self.frame)
            path = '/media/anurag/WareHouse/anmol/Detectron.pytorch/' + file_name + '.csv'
            path_2d = '/media/anurag/WareHouse/anmol/Detectron.pytorch/' + file_name + '_2d.csv'
            # path = '/media/anurag/WareHouse/datasets/mupots-3d-eval/TS9/'
            # poses = np.genfromtxt(path + 'sub_'+str(self.sub)+'_frame_' + str(self.frame % self.n_frames) + '.csv', delimiter=',')
            poses = np.genfromtxt(path, delimiter=',')
            poses_2d = np.genfromtxt(path_2d, delimiter=',')

            poses = poses.reshape(-1, 48)
            poses_2d = poses_2d.reshape(-1, 32)

            abs_poses, _ = self.get_absolute_pose(poses.reshape(-1,16,3), poses_2d.reshape(-1,16,2), 1000)
            poses = abs_poses.reshape(-1,48)

            n_people = min(poses.shape[0], len(self.Skeleton))
            self.n_people = n_people
            # n_people = 3
            # row = row[:-1]
            print(n_people)
            rows = np.zeros((n_people, 48))
            for j in range(n_people):
                row_ = poses[j, :]
                # if j == 1:
                #     row_ = row_ * 100
                # row_np = np.array(row_).reshape(16,3)
                # row_h = np.zeros((16,3))
                # loc = np.array([j*2000, 0, 0])
                # for i in range(len(self.edges)):
                #     h = self.edges[i][0]
                #     m = self.edges[i][1]
                #     row_h[h]  = row_np[m]
                # row_ = self.smoothPose(row_.reshape(1,48), j)
                row_ = self.skeletonFitting(row_.reshape(16,3))
                # row_ = row_ + loc.reshape(1,3)
                rows[j] = row_.reshape(48)
            rows = self.smoothPose(rows, poses_2d)
            for j in range(n_people):
                row_ = list(rows[j])
                self.Skeleton[j].updateLimbs(row_)
        renWin.Render()
        self.takeScreenShot(renWin)

    def smoothPose(self, poses, poses_2d):
        c = min(self.counter, 10)
        i = self.counter
        poses, poses_2d, _ = self.match_pose_stream(self.framebuff[i-c: i, :],\
                                self.framebuff2[i-c: i, :], poses, poses_2d, curr_fidel=1,\
                                metric='pck')
        ct = min(self.counter, self.context) # 5 is the context size
        self.framebuff[i,:self.n_people] = poses[:self.n_people]
        self.framebuff2[i,:self.n_people] = poses_2d[:self.n_people]
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

    def get_absolute_pose(self, return_3d, save_2d, height):
        f = 1080 / (2 * np.tan(np.deg2rad(27)))
        # f = 500
        edges = [[1,2], [2,6], [6,3], [3,4], \
                         [6,7], [7,8], [8,9]]
        # edges = [[0,1], [1,2], [2,6], [8,12], [12,11], [11,10], [6,3], [3,4], \
        #                 [4,5], [8,13], [13,14], [14,15], [6,7], [7,8], [8,9]]
        # Indexed with 1. Be careful.
        # edges_mpi = [[1,16], [16,2], [2,3], [3,4], [4,5], [2,6], [6,7], [7,8],\
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
            self.X = (save_2d[p, root+1, 0] - 720) * sbl3d / sbl2d
            X = self.X
            # Y = (save_2d[p, root, 1] - 640) * sbl3d / sbl2d
            p_root = np.array([X, 0.0, Z]).astype('float32')
            return_3d[p] += p_root
        return_3d = return_3d.reshape(-1, 16*3)
        return return_3d, save_2d


    def skeletonFitting(self, pose):
        tree = [[6,-1,0], [7,6,239.7], [8,7,254.4], [9,8,168.07],
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


    def match_pose_stream(self, past_3d, past_2d, curr_3d, curr_2d, curr_fidel=1, metric='pck'):
        """
            The input is a stream of 3D and 2D poses. The expected sizes are:
                3D: context x n_people x 16 x 3
                2D: context x n_people x 16 x 2
            We'll do a matching based on the agreement of the current frame
            with the past (context - 1) frames.
        """
        curr_3d = curr_3d.reshape(-1, 16, 3)
        curr_2d = curr_2d.reshape(-1, 16, 2)
        past_3d = past_3d.reshape(-1, 5, 16, 3)
        past_2d = past_2d.reshape(-1, 5, 16, 2)
        context = past_3d.shape[0]
        n_people = past_3d.shape[1]
        curr_n_people = curr_3d.shape[0]
        n_joints = past_3d.shape[2]
        inp_3d = np.zeros((n_people, n_joints, 3))
        inp_2d = np.zeros((n_people, n_joints, 2))
        inp_3d[:curr_n_people] = curr_3d
        inp_2d[:curr_n_people] = curr_2d
        curr_3d = inp_3d
        curr_2d = inp_2d

        matching = [-1]*n_people
        matched =  [-1]*n_people

        for i in range(n_people):
            pose1 = curr_3d[i]
            kps1 = curr_2d[i]
            # pdb.set_trace()
            matching_score = [0]*n_people
            for j in range(n_people):
                if matched[j] > 0:
                    continue
                for k in range(context):
                    pose2 = past_3d[k, j]
                    kps2 = past_2d[k, j]
                    if metric == 'mpjpe':
                        dist, _ = self.MPJPE(pose1, pose2)
                        matching_score[j] += dist
                    else:
                        # diff = abs(kps1 - kps2).reshape(32,1)
                        # print(diff)
                        diff = abs(kps1[6] - kps2[6])
                        matching_score[j] += 1 /  np.linalg.norm(diff)
                        # matches = np.ma.masked_less(diff,30).mask.astype('float32')
                        # matching_score[j] += np.sum(matches)
            # pdb.set_trace()
            if metric == 'mpjpe':
                score = np.min(matching_score)
            else:
                score = np.max(matching_score)
            if score > 0 and score < 200:
                if metric == 'mpjpe':
                    matching[i] = np.argmin(matching_score)
                else:
                    matching[i] = np.argmax(matching_score)
                matched[matching[i]] = 1

        # print(matching)
        # Now that we have the matches, arrange them correctly and return
        # The output array should be the same size as the previoius frame.
        # If a detection is missed, copy the last frame for the detection.
        # Currently, we're not accounting for new detections
        if curr_fidel == 1:
            output = curr_3d.copy()
            kps_out = curr_2d.copy()
            # kps_out = np.zeros((output.shape[0], 16,2))
            count = 0
            for i in range(n_people):
                if matching[i] > -1:
                    output[i] = curr_3d[matching[i]]
                    kps_out[i] = curr_2d[matching[i]]
                    count += 1
        else:
            output = past_3d[-1].copy()
            kps_out = np.zeros((output.shape[0], n_joints,2))
            for i in range(past_3d[-1].shape[0]):
                if matching[i] > -1:
                    output[i] = curr_3d[matching[i]]
                    kps_out[i] = curr_2d[matching[i]]
                else:
                    output[i] = past_3d[-1,i] * 0.0
                    kps_out[i] = past_2d[-1,i] * 0.0

        return output.reshape(-1,48), kps_out.reshape(-1,32), np.array(matching)


    def MPJPE(self,output3D, meta, norm=False):
      nJoints = output3D.shape[-2]
      output3D = output3D.reshape(1,nJoints,3)
      meta = meta.reshape(1,nJoints,3)
      output = output3D.copy()
      tree = [[6,-1,0], [7,6,239.7], [8,7,254.4], [9,8,178.07],
            [12,8,150.85], [11,12,279.66], [10,11,246.43], [13,8,150.85],
            [14,13,279.66], [15,14,246.43], [2,6,139.6], [1,2,448.04],
            [0,1,436.62], [3,6,139.59], [4,3,448.08], [5,4,436.62]
            ]

      for i in range(meta.shape[0]):
          for j in range(len(tree)):
              parent = tree[j][1]
              joint = tree[j][0]
              bone_length = tree[j][2]
              if parent == -1:
                  output[i,joint] = output3D[i,joint]
              else:
                  vecnorm = (output3D[i,joint] - output3D[i,parent]) / np.linalg.norm(output3D[i,joint]-output3D[i,parent])
                  output[i,joint] = output[i,parent] + bone_length * vecnorm

      p = output3D.copy()
      p = p.reshape(-1, nJoints, 3)
      # p = p * std_3d + mean_3d
      root = 6
      err, num3D = 0, 0
      for i in range(output3D.shape[0]):
          pRoot = p[i, root].copy()
          num3D += 1
          for j in range(nJoints):
            if norm == 1:
                p[i,j] = ((p[i,j] - pRoot) * std_3d[j]) + mean_3d[j]
            else:
                p[i, j] = (p[i, j] - pRoot) + meta[i, root]
          # p[i, 7] = (p[i, 6] + p[i, 8]) / 2
          for j in range(nJoints):
            dis = ((p[i, j] - meta[i, j]) ** 2).sum() ** 0.5
            err += dis / nJoints
      if num3D > 0:
        return err / num3D, num3D
      else:
        return 0, 0
