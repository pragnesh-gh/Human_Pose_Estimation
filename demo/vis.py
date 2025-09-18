import argparse
import cv2
from lib.preprocess import h36m_coco_format, revise_kpts
# from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import os, sys
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

POSE_DEBUG = os.getenv("POSE_DEBUG", "0") != "0"


try:
    from common.orientation import compute_torso_frame, smooth_quat
except ModuleNotFoundError:
    # As a fallback, also try the current working directory (in case you run from repo root)
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    from common.orientation import compute_torso_frame, smooth_quat
import json, csv
import os 
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy
import shutil
from pathlib import Path
from contextlib import contextmanager
def _dbg(msg, **kwargs):
    print(f"[VIS.DBG] {msg}")
    if kwargs:
        for k, v in kwargs.items():
            print(f"  - {k}: {v}")

@contextmanager
def _strip_argv():
    saved = sys.argv
    try:
        sys.argv = [saved[0]]
        yield
    finally:
        sys.argv = saved

sys.path.append(os.getcwd())
from common.model_poseformer import PoseTransformerV2 as Model
from common.camera import *
from common.camera import normalize_screen_coordinates  # for 2D norm (w,h) → [-1,1]
from common.generators import UnchunkedGenerator       # to handle padding/windowing

import matplotlib
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# suppress warnings while we ALSO properly close figures everywhere
matplotlib.rcParams['figure.max_open_warning'] = 0

def show2Dpose(kps, img):
    # Expect a single frame worth of 2D keypoints (17 joints, x/y)
    if not (isinstance(kps, np.ndarray) and kps.shape == (17, 2)):
        raise ValueError(f"show2Dpose expected kps shape (17,2), got {getattr(kps, 'shape', None)}")

    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j, (a, b) in enumerate(connections):
        p0 = np.asarray(kps[a], dtype=np.int32).ravel()
        p1 = np.asarray(kps[b], dtype=np.int32).ravel()
        start = (int(p0[0]), int(p0[1]))
        end = (int(p1[0]), int(p1[1]))
        cv2.line(img, start, end, lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, start, radius=3, color=(0, 255, 0), thickness=-1)
        cv2.circle(img, end, radius=3, color=(0, 255, 0), thickness=-1)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white) 
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


# def get_pose2D(source_path, output_dir):
#     # cap = cv2.VideoCapture(video_path)
#     # width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
#     # height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
#
#     print('\nGenerating 2D pose...')
#
#
#     #importing here because its classing with the arg parse with image.
#     saved_argv = sys.argv
#     try:
#         sys.argv = [saved_argv[0]]  # strip all external CLI flags
#         from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
#     finally:
#         sys.argv = saved_argv
#
#     keypoints, scores = hrnet_pose(source_path, det_dim=416, num_peroson=1, gen_output=True)
#     keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
#     re_kpts = revise_kpts(keypoints, scores, valid_frames)
#     print('Generating 2D pose successful!')
#
#     output_dir += 'input_2D/'
#     os.makedirs(output_dir, exist_ok=True)
#
#     output_npz = output_dir + 'keypoints.npz'
#     np.savez_compressed(output_npz, reconstruction=keypoints)

def get_pose2D(source_path, output_dir):
    _dbg("Entering get_pose2D", source_path=source_path, output_dir=output_dir)

    print('\nGenerating 2D pose...')

    # --- Shield HRNet import from our CLI flags ---
    with _strip_argv():
        _dbg("Importing hrnet gen_kpts with stripped argv", argv=sys.argv)
        from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
    _dbg("Imported hrnet gen_kpts", func_module=hrnet_pose.__module__, func_name=hrnet_pose.__name__)

    # Small helper to write a fallback npz + log when no person / HRNet fails
    def _write_fallback_npz_and_log():
        # Determine T (frames)
        T = 1
        # Try to detect if it's a video by opening with cv2 and reading frame count
        try:
            cap = cv2.VideoCapture(source_path)
            if cap is not None and cap.isOpened():
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if frames and frames > 0:
                    T = frames
            if cap is not None:
                cap.release()
        except Exception:
            pass

        # Build zeros reconstruction: (1, T, 17, 2) so downstream code is happy
        reconstruction = np.zeros((1, T, 17, 2), dtype=np.float32)
        valid_mask = np.zeros((T,), dtype=np.uint8)

        out2d_dir = os.path.join(output_dir, 'input_2D')
        os.makedirs(out2d_dir, exist_ok=True)
        output_npz = os.path.join(out2d_dir, 'keypoints.npz')
        np.savez_compressed(output_npz, reconstruction=reconstruction, valid_mask=valid_mask)

        # Write a simple log so you know this source was skipped for pose detection
        skip_log = os.path.join(output_dir, 'skip.log')
        with open(skip_log, 'a', encoding='utf-8') as f:
            f.write(f"[SKIP] No person detected or HRNet failed for: {source_path}\n")
            f.write(f"       Wrote fallback npz to: {output_npz}\n")
        _dbg("Wrote fallback npz + skip log", frames=T, npz=output_npz, log=skip_log)

    # --- Shield HRNet call from our CLI flags as well ---
    try:
        with _strip_argv():
            _dbg("Calling hrnet_pose(...) with stripped argv", argv=sys.argv,
                 det_dim=416, num_peroson=1, gen_output=True)
            keypoints, scores = hrnet_pose(source_path, det_dim=416, num_peroson=1, gen_output=True)
    except Exception as e:
        # Typical when no people are present: ValueError from internal transpose
        _dbg("HRNet call failed with Exception; writing fallback npz and continuing", err=str(e))
        _write_fallback_npz_and_log()
        print('Generating 2D pose (fallback, no person) complete!')
        return

    _dbg("HRNet returned", kpts_shape=np.array(keypoints).shape, scores_shape=np.array(scores).shape)

    try:
        keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)
        _dbg("After h36m_coco_format", kpts_shape=np.array(keypoints).shape,
             scores_shape=np.array(scores).shape, valid_frames_len=len(valid_frames))

        re_kpts = revise_kpts(keypoints, scores, valid_frames)
        _dbg("After revise_kpts", re_kpts_shape=np.array(re_kpts).shape)

        print('Generating 2D pose successful!')

        out2d_dir = os.path.join(output_dir, 'input_2D')
        os.makedirs(out2d_dir, exist_ok=True)
        output_npz = os.path.join(out2d_dir, 'keypoints.npz')

        # Build a frame-aligned valid mask from valid_frames (no dropped frames)
        T = int(re_kpts.shape[1]) if re_kpts.ndim >= 2 else len(valid_frames)
        valid_mask = np.zeros((T,), dtype=np.uint8)
        if len(valid_frames) > 0:
            vf = np.array(valid_frames, dtype=int)
            vf = vf[(vf >= 0) & (vf < T)]
            valid_mask[vf] = 1

        np.savez_compressed(
            output_npz,
            reconstruction=re_kpts,       # aligned to full video length
            valid_mask=valid_mask         # 1 = person present on that frame
        )
        if POSE_DEBUG:
            print(f"[2D.DBG] Saved {output_npz}: T={re_kpts.shape[1]}, valid_sum={int(valid_mask.sum())}")
        _dbg("Saved 2D keypoints", output_npz=output_npz)
    except Exception as e:
        # If post-HRNet formatting fails for an empty/odd case, fall back too
        _dbg("Post-processing failed; writing fallback npz and continuing", err=str(e))
        _write_fallback_npz_and_log()
        print('Generating 2D pose (fallback, post-process) complete!')
        return




def img2video(video_path, output_dir):
    _dbg("Entering img2video", video_path=video_path, output_dir=output_dir)
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 25  # fallback
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    if not names:
        _dbg("img2video: no pose frames found", dir=output_dir + 'pose/')
        return

    img0 = cv2.imread(names[0])
    size = (img0.shape[1], img0.shape[0])

    source_name = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, f"{source_name}.mp4")
    vw = cv2.VideoWriter(out_path, fourcc, fps, size)

    for name in names:
        frame = cv2.imread(name)
        vw.write(frame)

    vw.release()
    cap.release()
    _dbg("img2video: wrote", path=out_path, fps=fps, nframes=len(names))



def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([]) 
    plt.axis('off')
    ax.imshow(img)


def get_pose3D(video_path, output_dir, is_image=False, args=None):
    # args, _ = argparse.ArgumentParser().parse_known_args()
    _dbg("Entering get_pose3D", video_path=video_path, output_dir=output_dir, is_image=is_image)

    if args is None:
        class Dummy: pass
        args = Dummy()
    args.embed_dim_ratio, args.depth, args.frames = 32, 4, 243
    args.number_of_kept_frames, args.number_of_kept_coeffs = 27, 27
    args.pad = (args.frames - 1) // 2
    args.previous_dir = 'checkpoint/'
    args.n_joints, args.out_joints = 17, 17

    ## Reload
    model = nn.DataParallel(Model(args=args)).cuda()
    # model_dict = model.state_dict()
    # Put the pretrained model of PoseFormerV2 in 'checkpoint/']
    model_path = sorted(glob.glob(os.path.join(args.previous_dir, '27_243_45.2.bin')))[0]
    _dbg("Model & checkpoint", model_path=model_path)
    pre_dict = torch.load(model_path)
    model.load_state_dict(pre_dict['model_pos'], strict=True)

    model.eval()

    ## input
    # --- NEW: also load valid_mask (frame-aligned presence) ---
    _npz = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)
    keypoints = _npz['reconstruction']
    valid_mask = _npz['valid_mask'] if 'valid_mask' in _npz.files else None  # 1=person present

    # source handling
    if is_image:
        video_length = 1
        img0 = cv2.imread(video_path)
        if img0 is None:
            raise FileNotFoundError(f"Cannot read image: {video_path}")
    else:
        cap = cv2.VideoCapture(video_path)
        video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if POSE_DEBUG:
        kp_T = keypoints.shape[1]
        print(f"[3D.DBG] video_len={video_length}, keypoints_T={kp_T}, "
              f"valid_mask_len={(len(valid_mask) if valid_mask is not None else 'None')}")

    # --- NEW: cache base frame size & last good frame for robust timeline ---
    if not is_image:
        base_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
        base_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    else:
        base_h, base_w = img0.shape[:2]
    if base_w <= 0 or base_h <= 0:
        base_w, base_h = 640, 480  # safe fallback
    # last_valid_img = None

    # --- Orientation buffers (only used if enabled) ---
    orientation_rows = []
    prev_quat = None
    prev_post_out = None # remember previous 3D pose to hold-through gaps

    ## 3D
    print('\nGenerating 3D pose...')
    for i in tqdm(range(video_length)):
        # Getting the frame
        if is_image:
            img = img0
        else:
            ret, img = cap.read()
            if not ret or img is None:
                if POSE_DEBUG:
                    print(f"[3D.DBG] Frame {i}: cap.read() failed")
                # still produce empty outputs to keep timeline
                img = np.zeros((base_h, base_w, 3), dtype=np.uint8)
        # img_size = img.shape

        # --- decide if this frame has a person (avoid “speed-up” on gaps) ---
        has_person = True
        if valid_mask is not None and i < len(valid_mask):
            has_person = bool(valid_mask[i])
        elif valid_mask is not None and i >= len(valid_mask):
            # if keypoints shorter than video (should not happen after patch), treat as no person
            has_person = False
        # --- NEW END ---

        ## input frames
        start = max(0, i - args.pad)
        end = min(i + args.pad, len(keypoints[0]) - 1)

        input_2D_no = keypoints[0][start:end + 1]

        left_pad, right_pad = 0, 0
        if input_2D_no.shape[0] != args.frames:
            if i < args.pad:
                left_pad = args.pad - i
            if i > len(keypoints[0]) - args.pad - 1:
                right_pad = i + args.pad - (len(keypoints[0]) - 1)

            input_2D_no = np.pad(input_2D_no, ((left_pad, right_pad), (0, 0), (0, 0)), 'edge')

        if POSE_DEBUG:
            print(f"[3D.DBG] Frame {i}: has_person={has_person} win=[{start},{end}] padL={left_pad} padR={right_pad}")



        # --- NEW: if no person, skip model forward; save raw 2D and held 3D ---
        if not has_person:
            # Save plain 2D frame (no overlay)
            output_dir_2D = output_dir + 'pose2D/'
            os.makedirs(output_dir_2D, exist_ok=True)
            cv2.imwrite(output_dir_2D + f"{i:04d}_2D.png", img)

            # Hold last 3D pose or zeros
            post_out = prev_post_out if prev_post_out is not None else np.zeros((17, 3), dtype=np.float32)
            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1);
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)
            output_dir_3D = output_dir + 'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            plt.savefig(output_dir_3D + f"{i:04d}_3D.png", dpi=200, format='png', bbox_inches='tight')
            plt.clf();
            plt.close(fig)
            continue
        # Center frame's 2D in pixels for overlay
        overlay_2D_px = input_2D_no[args.pad]

        # input_2D_no += np.random.normal(loc=0.0, scale=5, size=input_2D_no.shape)
        # 1) temporal window is already built: input_2D_no (243,17,2) in *pixel* coords
        input_2D = normalize_screen_coordinates(input_2D_no, w=img.shape[1], h=img.shape[0])

        # 2) flipped copy in 2D (exactly once) + swap left/right joints
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]

        input_2D_aug = input_2D.copy()
        input_2D_aug[:, :, 0] *= -1
        input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
        input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
        # (2, 243, 17, 2)

        input_2D = input_2D[np.newaxis, :, :, :, :]

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()

        _dbg("Loaded input_2D", npz_path=output_dir + 'input_2D/keypoints.npz',
             shape=np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction'].shape)

        # N = input_2D.size(0)

        ## estimation
        output_3D_non_flip = model(input_2D[:, 0])
        output_3D_flip = model(input_2D[:, 1])
        # [1, 1, 17, 3]

        output_3D_flip[:, :, :, 0] *= -1
        output_3D_flip[:, :, joints_left + joints_right, :] = output_3D_flip[:, :, joints_right + joints_left, :]

        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        output_3D[:, :, 0, :] = 0
        post_out = output_3D[0, 0].cpu().detach().numpy()
        prev_post_out = post_out #remember for gap frames

        rot = np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],dtype='float32')
        post_out = camera_to_world(post_out, R=rot, t=0)
        post_out[:, 2] -= np.min(post_out[:, 2])

        # input_2D_no = input_2D_no[args.pad]

        overlay_src_idx = min(i, keypoints.shape[1] - 1)
        overlay_2D_px = keypoints[0, overlay_src_idx]  # shape (17, 2)

        # Frame timing introspection from OpenCV
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if not is_image else 0.0
        pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC)) if not is_image else 0.0
        est_ms = (i / fps * 1000.0) if (not is_image and fps > 0) else 0.0

        if POSE_DEBUG:
            print(
                "[SYNC.DBG] "
                f"i={i} start={start} end={end} padL={left_pad} padR={right_pad} "
                f"overlay_src_idx={overlay_src_idx} "
                f"pos_ms={pos_ms:.2f} est_ms={est_ms:.2f} fps={fps:.2f}"
            )

        ## 2D
        image = show2Dpose(overlay_2D_px, copy.deepcopy(img))
        cv2.putText(image, f"frame={i}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)

        output_dir_2D = output_dir + 'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + f"{i:04d}_2D.png", image)

        ## 3D
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05)
        ax = plt.subplot(gs[0], projection='3d')
        show3Dpose(post_out, ax)

        # --- Optional: compute torso orientation & overlay triad ---
        if getattr(args, "estimate_orientation", False):
            layout = getattr(args, "orientation_layout",
                             "h36m")  # Default H36M; for COCO-17 pass --orientation-layout coco17
            alpha = float(getattr(args, "orientation_alpha", 0.6))
            conf_min = float(getattr(args, "orientation_conf_min", 0.5))

            # Compute torso frame (camera/world consistent with post_out)
            quat_wxyz, fwd, right, up, conf = compute_torso_frame(post_out, layout=layout)
            conf = 0.0 if conf is None else float(conf)

            # Confidence-gated SLERP-EMA
            cur_quat = quat_wxyz
            if conf < conf_min and prev_quat is not None:
                smoothed = prev_quat
            else:
                smoothed = smooth_quat(prev_quat, cur_quat, alpha) if prev_quat is not None else cur_quat
            prev_quat = smoothed

            # Persist a tidy row for Colab
            orientation_rows.append({
                "frame_index": int(i),
                "quat_w": float(smoothed[0]),
                "quat_x": float(smoothed[1]),
                "quat_y": float(smoothed[2]),
                "quat_z": float(smoothed[3]),
                "confidence": conf,
                "forward_x": float(fwd[0]), "forward_y": float(fwd[1]), "forward_z": float(fwd[2]),
                "right_x": float(right[0]), "right_y": float(right[1]), "right_z": float(right[2]),
                "up_x": float(up[0]), "up_y": float(up[1]), "up_z": float(up[2]),
            })

            # Optional overlay on the same 3D axes
            if getattr(args, "orientation_overlay", False):
                # Anchor at pelvis/root. For COCO-17, change root index accordingly.
                root_idx = 0  # H36M pelvis
                origin = post_out[root_idx]  # [3]

                # Triad length relative to the scene size
                radius = max(1e-6, (post_out.max(axis=0) - post_out.min(axis=0)).max() / 2.0)
                s = float(getattr(args, "orientation_overlay_scale", 1.0)) * radius * 0.2

                # Draw F (red), R (green), U (blue)
                ax.plot([origin[0], origin[0] + s * fwd[0]],
                        [origin[1], origin[1] + s * fwd[1]],
                        [origin[2], origin[2] + s * fwd[2]], lw=2, c='r')
                ax.plot([origin[0], origin[0] + s * right[0]],
                        [origin[1], origin[1] + s * right[1]],
                        [origin[2], origin[2] + s * right[2]], lw=2, c='g')
                ax.plot([origin[0], origin[0] + s * up[0]],
                        [origin[1], origin[1] + s * up[1]],
                        [origin[2], origin[2] + s * up[2]], lw=2, c='b')

                # add legend once per figure
                ax.plot([], [], c='r', label='Forward (Z)')
                ax.plot([], [], c='g', label='Right (X)')
                ax.plot([], [], c='b', label='Up (Y)')
                ax.legend(loc='upper right')

        output_dir_3D = output_dir + 'pose3D/'
        os.makedirs(output_dir_3D, exist_ok=True)
        plt.savefig(output_dir_3D + str(('%04d' % i)) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
        plt.clf()
        plt.close(fig)

    print('Generating 3D pose successful!')

    if not is_image:
        cap.release()

    # --- Optional: save per-frame orientations for Colab ---
    if getattr(args, "estimate_orientation", False) and getattr(args, "orientation_save", None):
        out_path = Path(args.orientation_save)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ext = out_path.suffix.lower()
        if ext == ".csv":
            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "frame_index",
                        "quat_w", "quat_x", "quat_y", "quat_z",
                        "confidence",
                        "forward_x", "forward_y", "forward_z",
                        "right_x", "right_y", "right_z",
                        "up_x", "up_y", "up_z",
                    ]
                )
                writer.writeheader()
                writer.writerows(orientation_rows)
            print(f"[orientation] Saved CSV: {out_path}")
        else:
            p = out_path if ext == ".json" else out_path.with_suffix(".json")
            with p.open("w", encoding="utf-8") as f:
                json.dump(orientation_rows, f, ensure_ascii=False)
            print(f"[orientation] Saved JSON: {p}")

    ## all
    image_dir = 'results/'
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    if POSE_DEBUG:
        print(f"[COMPOSE.DBG] pose2D_frames={len(image_2d_dir)} pose3D_frames={len(image_3d_dir)}")

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        # ## crop
        # edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        # image_2d = image_2d[:, edge:image_2d.shape[1] - edge]
        #
        # edge = 130
        # image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize=font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize=font_size)

        ## save
        output_dir_pose = output_dir + 'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d' % i)) + '_pose.png', dpi=200, bbox_inches='tight')
        plt.clf()
        plt.close(fig)
        if (i % 50) == 0:
            plt.close('all')  # just in case

    if POSE_DEBUG:
        print(f"[COMPOSE.DBG] wrote {len(image_2d_dir)} composite frames to {output_dir + 'pose/'}")


if __name__ == "__main__":
    _dbg("Raw sys.argv at start", argv=sys.argv)

    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--video', type=str, help='input video')
    group.add_argument('--image', type=str, help='input image or absolute path')
    group.add_argument('--image-dir', type=str, help='directory containing images')
    parser.add_argument('--gpu', type=str, default='0', help='Gpu id')
    # --- Orientation (all no-ops unless estimate is enabled) ---
    parser.add_argument('--estimate-orientation', action='store_true',
                        help='Compute torso orientation per frame (camera-frame).') #These flags are ignored unless --estimate-orientation is set
    parser.add_argument('--orientation-layout', type=str, default='h36m', choices=['h36m', 'coco17'],
                        help='Joint layout used to compute torso frame. Default: h36m.')
    parser.add_argument('--orientation-alpha', type=float, default=0.6,
                        help='SLERP-EMA smoothing factor ∈ [0,1], higher = smoother.')
    parser.add_argument('--orientation-conf-min', type=float, default=0.5,
                        help='If confidence below this, hold previous quat.')
    parser.add_argument('--orientation-overlay', action='store_true',
                        help='Draw orientation triad on 3D view.')
    parser.add_argument('--orientation-overlay-scale', type=float, default=1.0,
                        help='Scale of triad relative to scene size.')
    parser.add_argument('--orientation-save', type=str, default=None,
                        help='Path to save per-frame orientations (.csv or .json).')



    args = parser.parse_args()
    _dbg("Parsed args", video=args.video, image=args.image, gpu=args.gpu)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    _dbg("CUDA_VISIBLE_DEVICES set", value=os.environ.get("CUDA_VISIBLE_DEVICES"))


    def _resolve_image_path(p: str) -> str:
        # absolute stays; otherwise prefix to demo/image/
        return p if os.path.isabs(p) else os.path.join('./demo/image/', p)


    def _list_images_in_dir(d: str):
        p = Path(d) if os.path.isabs(d) else Path('./demo/image') / d
        # common extensions
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
        files = []
        for e in exts:
            files.extend(sorted(p.glob(e)))
        return [str(f) for f in files]


    if args.video:
        # ---- VIDEO MODE (unchanged) ----
        source_path = args.video if os.path.isabs(args.video) else './demo/video/' + args.video
        is_image = False
        _dbg("Resolved source (video)", source_path=source_path, is_image=is_image)
        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_dir = './demo/output/' + source_name + '/'
        _dbg("Output directory", output_dir=output_dir)

        get_pose2D(source_path, output_dir)
        get_pose3D(source_path, output_dir, is_image=is_image, args=args)
        img2video(source_path, output_dir)
        print('Generating demo successful!')

    elif args.image:
        # ---- SINGLE IMAGE MODE (unchanged) ----
        source_path = _resolve_image_path(args.image)
        is_image = True
        _dbg("Resolved source (single image)", source_path=source_path, is_image=is_image)

        source_name = os.path.splitext(os.path.basename(source_path))[0]
        output_dir = './demo/output/' + source_name + '/'
        _dbg("Output directory", output_dir=output_dir)

        get_pose2D(source_path, output_dir)
        get_pose3D(source_path, output_dir, is_image=is_image, args=args)

        one_frame = os.path.join(output_dir, 'pose', '0000_pose.png')
        final_png = os.path.join(output_dir, f'{source_name}.png')
        shutil.copyfile(one_frame, final_png)
        print(f'Saved final image: {final_png}')
        print('Generating demo successful!')

    else:
        # ---- FOLDER MODE (new) ----
        images = _list_images_in_dir(args.image_dir)
        if not images:
            print(f'No images found in folder: {args.image_dir}')
            sys.exit(1)

        print(f'Found {len(images)} images in folder: {args.image_dir}')
        for idx, source_path in enumerate(images, 1):
            is_image = True
            source_name = os.path.splitext(os.path.basename(source_path))[0]
            output_dir = './demo/output/' + source_name + '/'
            print(f'[{idx}/{len(images)}] Processing: {source_name}')

            get_pose2D(source_path, output_dir)
            get_pose3D(source_path, output_dir, is_image=is_image, args=args)

            one_frame = os.path.join(output_dir, 'pose', '0000_pose.png')
            final_png = os.path.join(output_dir, f'{source_name}.png')
            shutil.copyfile(one_frame, final_png)
            print(f'  ↳ Saved: {final_png}')

        print('Folder processing complete!')



#Command: image-dir, image, video
# python demo/vis.py --video output_video.mp4 --gpu 0 --estimate-orientation --orientation-overlay --orientation-overlay-scale 1.0
