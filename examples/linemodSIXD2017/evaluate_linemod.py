#!/usr/bin/env python

import argparse
import os
import six

import chainer
from chainer import cuda
import numpy as np
import quaternion

from cp_net.models.dual_cp_network_ver2 import DualCenterProposalNetworkRes50_predict7
from datasets.linemod_sixd2017_single_instance import LinemodSIXDSingleInstanceDataset

from cp_net.pose_estimation_interface import SimplePoseEstimationInterface, PoseEstimationInterface
from cp_net.utils import inout

import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file')
    parser.add_argument('-g', '--gpu', default=0, type=int,
                        help='if -1, use cpu only (default: 0)')
    args = parser.parse_args()
    alpha=0.3
    image_alpha=1
    render_eps = 0.015

    data_path = '../../train_data/linemodSIXD2017'

    ## delta for visibility correspondance
    delta = 0.015 # [m]
    objs = np.arange(15) + 1
    n_class = len(objs) + 1
    distance_sanity = 0.02
    min_distance= 0.005
    output_scale = 0.14
    prob_eps = 0.4
    eps = 0.02
    im_size=(640, 480)
    interval = 15

    ## load object models
    obj_model_fpath_mask = os.path.join(data_path, 'models', 'obj_{0:0>2}.ply')
    obj_models = []
    for obj in objs:
        if obj != 'background':
            print 'Loading data: obj_{0}'.format(obj)
            obj_model_fpath = obj_model_fpath_mask.format(obj)
            obj_models.append(inout.load_ply(obj_model_fpath))

    ## load network model
    model = DualCenterProposalNetworkRes50_predict7(n_class=n_class)
    chainer.serializers.load_npz(args.model_file, model)

    if args.gpu >= 0:
        cuda.get_device(args.gpu).use()
        model.to_gpu()

    # pose estimator instance
    # pei = SimplePoseEstimationInterface(distance_sanity=distance_sanity,
    #                                     base_path=data_path,
    #                                     min_distance=min_distance, eps=prob_eps, im_size=im_size)
    pei = PoseEstimationInterface(objs= ['obj_{0:0>2}'.format(i) for i in objs],
                                  base_path=data_path,
                                  distance_sanity=distance_sanity,
                                  model_scale = 1000.0,
                                  model_partial= 1,
                                  min_distance=min_distance, eps=prob_eps, im_size=im_size)
    scores_pos = []
    scores_rot = []
    ids_arr = []
    scores_5cm5deg = []
    test_objs = np.delete(np.arange(15) + 1, [2, 6])
    for obj_id in test_objs:
        test = LinemodSIXDSingleInstanceDataset(data_path, obj_id,
                                                mode='test',
                                                interval=interval,
                                                metric_filter=output_scale + eps)
        im_ids = test.__len__()
        # im_ids = 50
        detect_cnt = 0
        cnt_5cm5deg = 0
        sum_pos = 0
        sum_rot = 0
        for im_id in tqdm.trange(im_ids):
            # print "executing {0} / {1}".format(im_id, test.__len__())
            img_rgb, img_depth, pos, rot, K = test.get_example(im_id)
            x_data = np.expand_dims(img_rgb, axis=0)
            with chainer.no_backprop_mode():
                if args.gpu >= 0:
                    x_data = cuda.to_gpu(x_data)
                    x = chainer.Variable(x_data)
                with chainer.using_config('train', False):
                    y_cls_d, y_cp_d, y_ocp_d = model(x)
                    y_cls = chainer.cuda.to_cpu(y_cls_d.data)[0]
                    y_cp = chainer.cuda.to_cpu(y_cp_d.data)[0]
                    y_ocp = chainer.cuda.to_cpu(y_ocp_d.data)[0]

            y_pos, y_rot = pei.execute(y_cls, y_cp * output_scale, y_ocp * output_scale, img_depth, K, estimate_idx = [obj_id - 1])
            for i in six.moves.range(n_class - 1):
                if np.sum(pos[i]) != 0 and  np.sum(rot[i]) != 0:
                    pos_diff = np.linalg.norm(y_pos[i] - pos[i])
                    quat = quaternion.from_rotation_matrix(np.dot(y_rot[i].T, rot[i]))
                    quat_w = min(1, abs(quat.w))
                    diff_angle = np.rad2deg(np.arccos(quat_w)) * 2
                    # print "obj_{0:0>2} : position_diff = {1}, rotation_diff = {2}".format(i + 1, pos_diff, diff_angle)
                    if i + 1 == obj_id and pos_diff < 1.0:
                        sum_pos += pos_diff
                        sum_rot += diff_angle
                        detect_cnt += 1
                        if  pos_diff < 0.05 and diff_angle < 5:
                            cnt_5cm5deg +=1

        print "obj_{0:0>2} : detection_rate = {1}, position_diff = {2}, rotation_diff = {3}, 5cm5deg = {4}" \
            .format(obj_id, detect_cnt / (1.0 * im_ids), sum_pos / detect_cnt, sum_rot / detect_cnt, cnt_5cm5deg / (1.0 * im_ids))
        scores_pos.append(sum_pos / im_ids)
        scores_rot.append(sum_rot / im_ids)
        scores_5cm5deg.append(cnt_5cm5deg)
        ids_arr.append(im_ids)


    print "-- results --"
    print scores_pos
    print scores_rot
    print "-- total results --"
    ids_arr = np.asarray(ids_arr)
    scores_5cm5deg = np.asarray(scores_5cm5deg)
    print "5cm5deg ratio : {}".format(np.sum(scores_5cm5deg) / (1.0 * np.sum(ids_arr)))

if __name__ == '__main__':
    main()