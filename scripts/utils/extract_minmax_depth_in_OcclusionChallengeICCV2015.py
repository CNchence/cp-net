#!/usr/bin/env python

import numpy as np
import argparse
import os
import glob

from cp_net.utils import inout

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pose_path')
    args = parser.parse_args()
    pose_path = args.pose_path
    paths = glob.glob(os.path.join(pose_path, '*', 'info_*.txt'))
    max_depth = 0
    min_depth = 100
    for path in paths:
        pose = inout.load_gt_pose_dresden(path)
        if pose['R'].size != 0 and pose['t'].size != 0 and pose['t'][2] > 0:
            depth = pose['t'][2]
            max_depth = max(depth, max_depth)
            min_depth = min(depth ,min_depth)

    print "-- results --"
    print "number of sampling data : {}".format(len(paths))
    print "max depth: {}".format(max_depth)
    print "min depth: {}".format(min_depth)


if __name__ == '__main__':
    main()