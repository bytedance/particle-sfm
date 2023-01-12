import pickle
import sys
import numpy as np
import os
import shutil
import argparse

# Get utils on path
sys.path.append("../")

from utils import data_utils

parser = argparse.ArgumentParser("Load frames for particlesfm")
parser.add_argument(
    "--track",
    type=int,
    required=True,
    help="Lyft track idx",
)
args = parser.parse_args()

# pkl_path = f"/data2/iphone-tracks//data/track_{args.track}.pkl"
pkl_path = f"/data2/speed-lyft/tracks/track_{args.track}.pkl"

with open(pkl_path, "rb") as f:
    track = pickle.load(f)

data = []
for f in track["track_frames"]:
    data.append(
        {
            "frame_idx": f["frame_idx"],
            "timestamp": f["timestamp"],
            "rgb_path": f["rgb_path"],
            "dep_path": f["dep_path"],
            "mask_locs": np.array(f["mask_locs"]).T[:, [1, 0]],
        }
    )

# f["ts"],
# f"{track['data_path']}/zipped_frames/frame_{f['frame_idx']}.jpg",
# f"{track['data_path']}/zipped_frames/frame_{f['frame_idx']}.dep",
# np.array(f["locs"]).T[:, [1, 0]],

out_path = f"./example/lyft_track_{args.track}/images/"
os.makedirs(out_path, exist_ok=True)

for i, f in enumerate(data):
    shutil.copy(f["rgb_path"], f"{out_path}/{i:05d}.jpg")
