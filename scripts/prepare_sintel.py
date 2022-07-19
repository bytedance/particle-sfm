import os
import argparse

parser = argparse.ArgumentParser("prepare sintel")
parser.add_argument("--src_dir")
parser.add_argument("--tgt_dir")
args = parser.parse_args()

seqs = sorted(os.listdir(args.src_dir))
for seq in seqs:
    full_seq = os.path.join(args.src_dir, seq)
    tgt_seq = os.path.join(args.tgt_dir, seq, "images/")
    os.makedirs(tgt_seq)
    command = "cp " + full_seq + "/* " + tgt_seq
    os.system(command)
    print(seq)