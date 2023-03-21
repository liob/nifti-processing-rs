#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path
import nibabel as nib
from nibabel.processing import resample_to_output
from tqdm import tqdm
import argparse


parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)
parser.add_argument('input_images', metavar='<fn>', type=str, nargs='*',
                    help='input nifti files')
parser.add_argument('-r', metavar='<float>', type=float, default=1.5,
                    help='target resolution')
parser.add_argument('-order', metavar='<int>', type=int, default=1,
                    help='resampling mode: 0 -> nearest, 1 -> trilinear')
parser.add_argument('-mode', metavar='<str>', type=str, default='constant',
                    help='out of sample strategy: constant, nearest, reflect, wrap')
parser.add_argument('-cval', metavar='<float>', type=int, default=0.,
                    help='Value used for points outside the boundaries of the input if mode=constant')
parser.add_argument('-o', metavar='<dir>', type=str, default='.',
                    help='output directory')
args = parser.parse_args()


for fn in tqdm(args.input_images):
    root, cn = path.split(fn)
    img = nib.load(fn)
    resampled_img = resample_to_output(img, voxel_sizes=args.r, order=args.order, \
                                            mode=args.mode, cval=args.cval)
    resampled_img.to_filename(path.join(args.o, cn))
