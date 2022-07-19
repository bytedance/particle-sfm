import os
import numpy as np
import argparse
import cv2
from point_trajectory.track import track
from point_trajectory.utils import flow_check
from motion_seg.core.dataset.data_utils import read_flow_png
from third_party.MiDaS import run_midas

# flyingthings3d flow data structure
split = ['TRAIN', 'TEST']
part = ['A', 'B', 'C']
time = ['into_future', 'into_past']
hand = ['left', 'right']

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header.decode("ascii") == 'PF':
        color = True
    elif header.decode("ascii") == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def load_flows_flyingthings3d(flow_dir):
    flos = []
    flows = sorted(os.listdir(flow_dir))
    for flow in flows:
        flo = read_flow_png(os.path.join(flow_dir, flow))
        flos.append(flo)
    return flos

def read_pad_point_traj_flyingthings3d(fname, thres=3, length=10):
    # read and pad the point trajectory data into [N, L, 2] with mask
    trajs, masks = [], []
    # read the point trajectory data
    npy = np.load(fname, allow_pickle=True).item()
    all_trajs, all_times, seq_len = \
        npy['trajs'], npy['times'], npy['leng']

    assert length == seq_len
    traj_num = len(all_trajs)
    # loop through each trajectory
    for i in range(traj_num):
        traj_s, time_s = all_trajs[i], all_times[i]
        traj = np.zeros((length, 2))
        pad_mask = np.ones((length, 1))
        if len(time_s) < thres:
            continue
        for j in range(len(time_s)):
            time = int(time_s[j])
            traj[time][0] = traj_s[j][0]
            traj[time][1] = traj_s[j][1]
            pad_mask[time][0] = 0
        trajs.append(traj)
        masks.append(pad_mask)
    trajs = np.stack(trajs, 0) # [N, L, 2]
    masks = np.stack(masks, 0) # [N, L, 1]
    return trajs, masks

def find_traj_label(traj, mask, gts):
    # traj: [N, L, 2], mask: [N, L, 1]
    # gts: [L, H, W]
    N, L = traj.shape[:2]
    label_cls = np.zeros(N)
    for i in range(N):
        t = traj[i]
        label_num = 0
        total_num = 0
        for j in range(L):
            if mask[i,j]:
                continue
            else:
                x, y = t[j]
                label = gts[j, round(y), round(x)]
                label_num += label
                total_num += 1
        if label_num > total_num // 2:
            label_cls[i] = 1
    return label_cls

def main_compress_flow(root_dir):
    flow_src_dir = os.path.join(root_dir, "optical_flow_raw")
    save_flow_tgt_dir = os.path.join(root_dir, "optical_flow")
    for s in split:
        for p in part:
            cur_dir = os.path.join(flow_src_dir, s, p)
            seqs = sorted(os.listdir(cur_dir))
            for seq in seqs:
                for t in time:
                    for h in hand:
                        img_dir = os.path.join(flow_src_dir, s, p, seq, t, h)
                        save_dir = os.path.join(save_flow_tgt_dir, s, p, seq, t, h)
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)
                        names = sorted(os.listdir(img_dir))
                        for n in names:
                            flow = readPFM(os.path.join(img_dir, n))
                            flow_int = flow * 100 + 32000
                            save_name = n.replace('.pfm', '.png')
                            cv2.imwrite(os.path.join(save_dir, save_name), flow_int.astype(np.uint16))
                print(seq)

def main_track_flyingthings3d(root_dir, sample_ratio):
    # initiate save_dir
    save_dir = os.path.join(root_dir, "point_traj")
    flow_dir = os.path.join(root_dir, "optical_flow")
    os.makedirs(save_dir, exist_ok=True)
    for s in split:
        for p in part:
            cur_dir = os.path.join(flow_dir, s, p)
            seqs = sorted(os.listdir(cur_dir))
            for seq in seqs:
                for h in hand:
                    output_npy_dir = os.path.join(save_dir, s, p, seq, h)
                    os.makedirs(output_npy_dir, exist_ok=True)
                    output_npy_fname = os.path.join(output_npy_dir, "track.npy")

                    flow_seq_dir = os.path.join(flow_dir, s, p, seq, "into_future", h)
                    flow_seq_bk_dir = os.path.join(flow_dir, s, p, seq, "into_past", h)
                    flows_f = load_flows_flyingthings3d(flow_seq_dir)[:-1]
                    flows_b = load_flows_flyingthings3d(flow_seq_bk_dir)[1:]
                    error_maps, occ_maps = flow_check(flows_f, flows_b, thres=1.0)
                    n_images = len(flows_f) + 1

                    # start connecting tracks into point trajectories
                    trajs = track(flows_f, occ_maps, sample_ratio)
                    # save the outputs
                    xys, times = [], []
                    for traj in trajs:
                        xys.append(traj.xys)
                        times.append(traj.times)
                    track_dat = {}
                    track_dat["trajs"], track_dat['times'], track_dat['leng'] = xys, times, n_images
                    np.save(output_npy_fname, track_dat, allow_pickle=True)
                    print(output_npy_dir)

def main_post_process_flyingthings3d(root_dir):
    save_dir = os.path.join(root_dir, "point_traj")
    for s in split:
        for p in part:
            cur_dir = os.path.join(save_dir, s, p)
            seqs = sorted(os.listdir(cur_dir))
            for seq in seqs:
                for h in hand:
                    pt_seq_dir = os.path.join(cur_dir, seq, h)
                    fname = os.path.join(pt_seq_dir, "track.npy")
                    pt, pad_mask = read_pad_point_traj_flyingthings3d(fname)
                    np.savez(os.path.join(pt_seq_dir, 'pt.npz'), pt)
                    np.savez(os.path.join(pt_seq_dir, 'pad_mask.npz'), pad_mask)
                    print(pt_seq_dir)

def main_calculate_motion_labels(root_dir):
    traj_root = os.path.join(root_dir, 'point_traj')
    gt_root = os.path.join(root_dir, 'motion_labels')
    for s in split:
        for p in part:
            cur_dir = os.path.join(traj_root, s, p)
            seqs = sorted(os.listdir(cur_dir))
            for seq in seqs:
                for h in hand:
                    gts = []
                    gt_seq = os.path.join(gt_root, s, p, seq, h)
                    if not os.path.exists(gt_seq):
                        continue
                    gt_names = sorted(os.listdir(gt_seq))
                    if len(gt_names) != 10:
                        continue
                    for gt_name in gt_names:
                        gt = cv2.imread(os.path.join(gt_seq, gt_name))[:,:,0]
                        gts.append(gt)
                    gts = np.stack(gts, 0)
                    pt_seq = os.path.join(cur_dir, seq, h)
                    traj = np.load(os.path.join(pt_seq, 'pt.npz'))['arr_0']
                    mask = np.load(os.path.join(pt_seq, 'pad_mask.npz'))['arr_0']
                    traj_label = find_traj_label(traj, mask, gts)
                    traj_label_name = os.path.join(pt_seq, 'traj_label.npy')
                    np.save(traj_label_name, traj_label)
                print(seq)

def main_run_midas(root_dir):
    input_dir = os.path.join(root_dir, "frames_finalpass")
    save_dir = os.path.join(root_dir, "midas_depth")
    for s in split:
        for p in part:
            cur_dir = os.path.join(input_dir, s, p)
            seqs = sorted(os.listdir(cur_dir))
            for seq in seqs:
                for h in hand:
                    full_image_dir = os.path.join(input_dir, s, p, seq, h)
                    output_dir = os.path.join(save_dir, s, p, seq, h)
                    os.makedirs(output_dir, exist_ok=True)
                    run_midas(full_image_dir, output_dir)
                    print(output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("prepare flyingthings3d")
    parser.add_argument("--src_dir")
    args = parser.parse_args()

    # First compress the groundtruth optical flow from .flo into .png (save disk)
    main_compress_flow(args.src_dir)
    # Then run the tracking
    print("Tracking...")
    main_track_flyingthings3d(args.src_dir, sample_ratio=2)
    # post process the trajectory (pad)
    print("Post processing...")
    main_post_process_flyingthings3d(args.src_dir)
    # calculate the motion labels
    print("Calculate the label...")
    main_calculate_motion_labels(args.src_dir)
    # run the MiDaS
    print("Run the midas depth...")
    main_run_midas(args.src_dir)