import os
import glob
import cv2
import numpy as np
import argparse

parser = argparse.ArgumentParser("prepare scannet")
parser.add_argument("--src_dir")
parser.add_argument("--tgt_dir")
args = parser.parse_args()

sample_ratio = 3
scene_num = 20
first_k = 1500

seqs = sorted(os.listdir(args.src_dir))
for i in range(scene_num):
    seq = seqs[i]
    full_seq = os.path.join(args.src_dir, seq, "color")
    tgt_seq = os.path.join(args.tgt_dir, seq, "images")
    os.makedirs(tgt_seq)
    total_num = len(glob.glob(full_seq + "/*.jpg"))
    total_num = np.minimum(total_num, first_k)
    for j in range(total_num):
        if j % sample_ratio == 0:
            src_name = os.path.join(full_seq, str(j)+".jpg")
            tgt_name = os.path.join(tgt_seq, "{:04d}.jpg".format(j))
            img = cv2.imread(src_name)
            img_r = cv2.resize(img, (640,480))
            cv2.imwrite(tgt_name, img_r)
    print(seq)
