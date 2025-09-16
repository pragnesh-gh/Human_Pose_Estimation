# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import matplotlib

matplotlib.use('Agg')


import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, writers
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import subprocess as sp
import cv2


def get_resolution(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=width,height', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            w, h = line.decode().strip().split(',')
            return int(w), int(h)


def get_fps(filename):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=r_frame_rate', '-of', 'csv=p=0', filename]
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        for line in pipe.stdout:
            a, b = line.decode().strip().split('/')
            return int(a) / int(b)


def read_video(filename, skip=0, limit=-1):
    # w, h = get_resolution(filename)
    w = 1000
    h = 1002

    command = ['ffmpeg',
               '-i', filename,
               '-f', 'image2pipe',
               '-pix_fmt', 'rgb24',
               '-vsync', '0',
               '-vcodec', 'rawvideo', '-']

    i = 0
    with sp.Popen(command, stdout=sp.PIPE, bufsize=-1) as pipe:
        while True:
            data = pipe.stdout.read(w * h * 3)
            if not data:
                break
            i += 1
            if i > limit and limit != -1:
                continue
            if i > skip:
                yield np.frombuffer(data, dtype='uint8').reshape((h, w, 3))


def downsample_tensor(X, factor):
    length = X.shape[0] // factor * factor
    return np.mean(X[:length].reshape(-1, factor, *X.shape[1:]), axis=1)


# def render_animation(keypoints, keypoints_metadata, poses, skeleton, fps, bitrate, azim, output, viewport,
#                      limit=-1, downsample=1, size=6, input_video_path=None, input_video_skip=0):
#     """
#     TODO
#     Render an animation. The supported output modes are:
#      -- 'interactive': display an interactive figure
#                        (also works on notebooks if associated with %matplotlib inline)
#      -- 'html': render the animation as HTML5 video. Can be displayed in a notebook using HTML(...).
#      -- 'filename.mp4': render and export the animation as an h264 video (requires ffmpeg).
#      -- 'filename.gif': render and export the animation a gif file (requires imagemagick).
#     """
#     plt.ioff()
#     fig = plt.figure(figsize=(size * (1 + len(poses)), size))
#     ax_in = fig.add_subplot(1, 1 + len(poses), 1)
#     ax_in.get_xaxis().set_visible(False)
#     ax_in.get_yaxis().set_visible(False)
#     ax_in.set_axis_off()
#     ax_in.set_title('Input')
#
#     ax_3d = []
#     lines_3d = []
#     trajectories = []
#     radius = 1.7
#     for index, (title, data) in enumerate(poses.items()):
#         ax = fig.add_subplot(1, 1 + len(poses), index + 2, projection='3d')
#         ax.view_init(elev=15., azim=azim)
#         ax.set_xlim3d([-radius / 2, radius / 2])
#         ax.set_zlim3d([0, radius])
#         ax.set_ylim3d([-radius / 2, radius / 2])
#         try:
#             ax.set_aspect('equal')
#         except NotImplementedError:
#             ax.set_aspect('auto')
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_zticklabels([])
#         ax.dist = 7.5
#         ax.set_title(title)  # , pad=35
#         ax_3d.append(ax)
#         lines_3d.append([])
#         trajectories.append(data[:, 0, [0, 1]])
#     poses = list(poses.values())
#
#     # Decode video
#     if input_video_path is None:
#         # Black background
#         all_frames = np.zeros((keypoints.shape[0], viewport[1], viewport[0]), dtype='uint8')
#     else:
#         # Load video using ffmpeg
#         all_frames = []
#         for f in read_video(input_video_path, skip=input_video_skip, limit=limit):
#             all_frames.append(f)
#         effective_length = min(keypoints.shape[0], len(all_frames))
#         all_frames = all_frames[:effective_length]
#
#         keypoints = keypoints[input_video_skip:]  # todo remove
#         for idx in range(len(poses)):
#             poses[idx] = poses[idx][input_video_skip:]
#
#         if fps is None:
#             fps = get_fps(input_video_path)
#
#     if downsample > 1:
#         keypoints = downsample_tensor(keypoints, downsample)
#         all_frames = downsample_tensor(np.array(all_frames), downsample).astype('uint8')
#         for idx in range(len(poses)):
#             poses[idx] = downsample_tensor(poses[idx], downsample)
#             trajectories[idx] = downsample_tensor(trajectories[idx], downsample)
#         fps /= downsample
#
#     initialized = False
#     image = None
#     lines = []
#     points = None
#
#     if limit < 1:
#         limit = len(all_frames)
#     else:
#         limit = min(limit, len(all_frames))
#
#     parents = skeleton.parents()
#
#     def update_video(i):
#         nonlocal initialized, image, lines, points
#
#         for n, ax in enumerate(ax_3d):
#             ax.set_xlim3d([-radius / 2 + trajectories[n][i, 0], radius / 2 + trajectories[n][i, 0]])
#             ax.set_ylim3d([-radius / 2 + trajectories[n][i, 1], radius / 2 + trajectories[n][i, 1]])
#
#         # Update 2D poses
#         joints_right_2d = keypoints_metadata['keypoints_symmetry'][1]
#         colors_2d = np.full(keypoints.shape[1], 'black')
#         colors_2d[joints_right_2d] = 'red'
#         if not initialized:
#             image = ax_in.imshow(all_frames[i], aspect='equal')
#
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     # Draw skeleton only if keypoints match (otherwise we don't have the parents definition)
#                     lines.append(ax_in.plot([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                             [keypoints[i, j, 1], keypoints[i, j_parent, 1]], color='pink'))
#
#                 col = 'red' if j in skeleton.joints_right() else 'black'
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n].append(ax.plot([pos[j, 0], pos[j_parent, 0]],
#                                                [pos[j, 1], pos[j_parent, 1]],
#                                                [pos[j, 2], pos[j_parent, 2]], zdir='z', c=col))
#
#             points = ax_in.scatter(*keypoints[i].T, 10, color=colors_2d, edgecolors='white', zorder=10)
#
#             initialized = True
#         else:
#             image.set_data(all_frames[i])
#
#             for j, j_parent in enumerate(parents):
#                 if j_parent == -1:
#                     continue
#
#                 if len(parents) == keypoints.shape[1] and keypoints_metadata['layout_name'] != 'coco':
#                     lines[j - 1][0].set_data([keypoints[i, j, 0], keypoints[i, j_parent, 0]],
#                                              [keypoints[i, j, 1], keypoints[i, j_parent, 1]])
#
#                 for n, ax in enumerate(ax_3d):
#                     pos = poses[n][i]
#                     lines_3d[n][j - 1][0].set_xdata(np.array([pos[j, 0], pos[j_parent, 0]]))
#                     lines_3d[n][j - 1][0].set_ydata(np.array([pos[j, 1], pos[j_parent, 1]]))
#                     lines_3d[n][j - 1][0].set_3d_properties(np.array([pos[j, 2], pos[j_parent, 2]]), zdir='z')
#
#             points.set_offsets(keypoints[i])
#
#         print('{}/{}      '.format(i, limit), end='\r')
#
#     fig.tight_layout()
#
#     anim = FuncAnimation(fig, update_video, frames=np.arange(0, limit), interval=1000 / fps, repeat=False)
#     if output.endswith('.mp4'):
#         Writer = writers['ffmpeg']
#         writer = Writer(fps=fps, metadata={}, bitrate=bitrate)
#         anim.save(output, writer=writer)
#     elif output.endswith('.gif'):
#         anim.save(output, dpi=80, writer='imagemagick')
#     else:
#         raise ValueError('Unsupported output format (only .mp4 and .gif are supported)')
#     plt.close()

def render_animation(keypoints, kps_lines, joints_left, joints_right,
                     input_keypoints=None, fps=25, bitrate=3000, azim=70, elev=15,
                     output='animation.mp4', viewport=(3, 3), downsample=1,
                     limit=None, input_video_path=None, input_video_skip=0,
                     orientations=None,              # NEW: optional per-frame triads
                     orientation_scale=1.0,          # NEW: triad length multiplier
                     orientation_root_joint=0        # NEW: where to anchor the triad (0=pivot/pelvis in H36M)
                     ):
    """
    Renders the 3D skeleton animation. If `orientations` is provided (a list/seq
    of dicts with keys 'forward','right','up', each R^3), a triad is drawn
    per frame, anchored at `orientation_root_joint`.

    Notes:
    - Default joint layout assumed: H36M (pelvis = 0). If you later switch to COCO-17,
      pass the appropriate root index (e.g., 11 or 12 depending on your mapping) and
      feed `orientations` built with layout='coco17'.
    - Triad colors: Forward=red, Right=green, Up=blue.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter

    assert keypoints.ndim == 3 and keypoints.shape[-1] == 3, "keypoints must be [T, J, 3]"
    T, J, _ = keypoints.shape

    if limit is not None:
        T = min(T, limit)
        keypoints = keypoints[:T]
        if input_keypoints is not None:
            input_keypoints = input_keypoints[:T]
        if orientations is not None:
            orientations = orientations[:T]

    # Matplotlib fig/axes setup (unchanged from your style)
    fig = plt.figure(figsize=viewport)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=elev, azim=azim)

    # Set equal aspect and reasonable bounds
    # (Keep your existing bounds logic if you already have one.)
    kpt_min = keypoints.reshape(-1, 3).min(axis=0)
    kpt_max = keypoints.reshape(-1, 3).max(axis=0)
    center = (kpt_min + kpt_max) / 2.0
    radius = (kpt_max - kpt_min).max() / 2.0
    radius = max(radius, 1e-3)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

    # Draw skeleton lines once; update their data each frame
    lines_3d = []
    for (i, j) in kps_lines:
        line, = ax.plot([keypoints[0, i, 0], keypoints[0, j, 0]],
                        [keypoints[0, i, 1], keypoints[0, j, 1]],
                        [keypoints[0, i, 2], keypoints[0, j, 2]],
                        lw=2, c='k')
        lines_3d.append(line)

    # Left/right coloring (optional; preserve your existing styling if present)
    for i in joints_left:
        for j in joints_left:
            pass  # keep your existing color handling if you already do this
    for i in joints_right:
        for j in joints_right:
            pass

    # --- NEW: orientation triad artists (created once, updated per frame) ---
    draw_triad = orientations is not None
    if draw_triad:
        # Create 3 line artists for forward/right/up
        triad_forward, = ax.plot([0, 0], [0, 0], [0, 0], lw=2, c='r')  # Forward
        triad_right,   = ax.plot([0, 0], [0, 0], [0, 0], lw=2, c='g')  # Right
        triad_up,      = ax.plot([0, 0], [0, 0], [0, 0], lw=2, c='b')  # Up
    else:
        triad_forward = triad_right = triad_up = None

    writer = FFMpegWriter(fps=fps, bitrate=bitrate)
    with writer.saving(fig, output, dpi=100):
        for t in range(T):
            P = keypoints[t]  # [J, 3]

            # Update skeleton segments
            for seg_idx, (i, j) in enumerate(kps_lines):
                lines_3d[seg_idx].set_data([P[i, 0], P[j, 0]], [P[i, 1], P[j, 1]])
                lines_3d[seg_idx].set_3d_properties([P[i, 2], P[j, 2]])

            # --- NEW: update triad if provided for this frame ---
            if draw_triad:
                ori = orientations[t]
                # Expecting dict with 'forward','right','up' keys
                fwd = np.asarray([ori['forward_x'], ori['forward_y'], ori['forward_z']]) \
                      if 'forward_x' in ori else np.asarray(ori['forward'])
                right = np.asarray([ori['right_x'], ori['right_y'], ori['right_z']]) \
                        if 'right_x' in ori else np.asarray(ori['right'])
                up = np.asarray([ori['up_x'], ori['up_y'], ori['up_z']]) \
                     if 'up_x' in ori else np.asarray(ori['up'])

                # Anchor (pelvis/root). For COCO-17, pass the correct root id and
                # feed orientations computed with layout='coco17'.
                origin = P[orientation_root_joint]

                # Scale the triad for visibility relative to the scene radius
                s = orientation_scale * radius * 0.2

                # Forward (red)
                triad_forward.set_data([origin[0], origin[0] + s * fwd[0]],
                                       [origin[1], origin[1] + s * fwd[1]])
                triad_forward.set_3d_properties([origin[2], origin[2] + s * fwd[2]])

                # Right (green)
                triad_right.set_data([origin[0], origin[0] + s * right[0]],
                                     [origin[1], origin[1] + s * right[1]])
                triad_right.set_3d_properties([origin[2], origin[2] + s * right[2]])

                # Up (blue)
                triad_up.set_data([origin[0], origin[0] + s * up[0]],
                                  [origin[1], origin[1] + s * up[1]])
                triad_up.set_3d_properties([origin[2], origin[2] + s * up[2]])

            writer.grab_frame()

    plt.close(fig)
