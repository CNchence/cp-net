
from matplotlib import pylab as plt
import cv2
import numpy as np
import os

from skimage.color.colorlabel import DEFAULT_COLORS
from skimage.color.colorlabel import color_dict

from cp_net.utils import multi_object_renderer
from cp_net.utils import renderer
from cp_net.utils import inout

import glob

def main():
    # Camera parameters
    im_size = (640, 480)
    K = np.array([[572.41140, 0, 325.26110],
                  [0, 573.57043, 242.04899],
                  [0, 0, 0]])

    objs = ['Ape', 'Can', 'Cat', 'Driller', 'Duck', 'Eggbox', 'Glue', 'Holepuncher']
    model_fpath_mask = os.path.join('../../train_data/OcclusionChallengeICCV2015',
                                    'models_ply', '{0}.ply')

    pose_fpath_mask = os.path.join('../../train_data/OcclusionChallengeICCV2015',
                                   'poses', '{0}', '*.txt')
    models = []
    gt_poses = []
    for obj in objs:
        print 'Loading data:', obj
        model_fpath = model_fpath_mask.format(obj)
        models.append(inout.load_ply(model_fpath))

        gt_fpaths = sorted(glob.glob(pose_fpath_mask.format(obj)))
        gt_poses_obj = []
        for gt_fpath in gt_fpaths:
            gt_poses_obj.append(
                inout.load_gt_pose_dresden(gt_fpath))
        gt_poses.append(gt_poses_obj)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 10))
    repeats = 100
    for i in range(repeats):
        print "multi object rendering {} / {}".format(i, repeats)
        rot_list = []
        pos_list = []
        indices = np.random.choice(len(objs), 1)
        model_list = []
        labels = []
        for obj_id, obj_name in enumerate(objs):
            # pos = np.random.rand(3) - 0.5
            # pos = np.zeros(3)
            # pos[2] += -2.0
            # pos = np.array([0,0,-1.5]).T
            # rot = np.eye(3)
            pose = gt_poses[obj_id][int(i)]
            if pose['R'].size != 0 and pose['t'].size != 0:
                pos = pose['t']
                rot = pose['R']
                pos_list.append(pos)
                rot_list.append(rot)
                model_list.append(models[obj_id])
                # rgb_ren, depth_ren = renderer.render(
                #     models[obj_id], im_size, K, rot, pos, 0.1, 4.0, mode='rgb+depth')
                labels.append(obj_id + 1)
        rgb_ren, depth_ren, label_ren  = multi_object_renderer.render(
            model_list, im_size, K, rot_list, pos_list, 0.1, 4.0,
            labels=labels, mode='rgb+depth+label')

        label_img = np.zeros((im_size[1], im_size[0], 3))
        n_colors = len(DEFAULT_COLORS)
        for lbl_id in labels:
            color = color_dict[DEFAULT_COLORS[lbl_id % n_colors]]
            label_img[(label_ren == lbl_id), :] = color

        # Clear axes
        for ax in axes.flatten():
            ax.clear()
        axes[0].imshow(rgb_ren.astype(np.uint8))
        axes[0].set_title('RGB image')
        axes[1].imshow(depth_ren)
        axes[1].set_title('Depth image')
        axes[2].imshow(label_img)
        axes[2].set_title('Label image')
        fig.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95,
                            hspace=0.15, wspace=0.15)
        plt.draw()
        # # plt.pause(0.01)
        plt.waitforbuttonpress()

if __name__ == '__main__':
    main()
