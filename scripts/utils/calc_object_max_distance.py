#!/usr/bin/env python

import numpy as np
import argparse
import os
import glob
import pyassimp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')
    args = parser.parse_args()
    model_path = args.model_path
    paths = sorted(glob.glob(os.path.join(model_path, "*.ply")))
    max_axis_dist = np.zeros(3)
    for path in paths:
        model = pyassimp.core.load(path)
        max_obj_dist = np.zeros(3)
        for mesh in model.meshes:
            verts = mesh.vertices
            max_verts = np.max(verts, axis=0)
            max_obj_dist = np.max(np.vstack((max_obj_dist, max_verts)),axis=0)
        print "----"
        print "object path : {}".format(path)
        print "max coordinates axis distances: {}".format(max_obj_dist)
        print "max distances: {}".format(np.max(max_obj_dist))
        max_axis_dist = np.max(np.vstack((max_axis_dist, max_obj_dist)),axis=0)
    print "-- results --"
    print "max distances: {}".format(max_axis_dist)


if __name__ == '__main__':
    main()