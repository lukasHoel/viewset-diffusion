# Copyright (c) Meta Platforms, Inc. and affiliates.

import logging
import os
import shutil
import uuid
import socket
from typing import Optional, List
from tqdm.auto import tqdm

import imageio

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


# We globally define the path manager.
_HOSTNAME = socket.gethostname()
if _HOSTNAME.endswith(".facebook.com") or _HOSTNAME.endswith(".fbinfra.net"):
    LOGGER.info("facebook environment detected.")
    FBENV = True
else:
    FBENV = False
    LOGGER.info("not in facebook environment.")


from iopath.common.file_io import PathManager, get_cache_dir

pmgr = PathManager()

if FBENV:
    from iopath.fb.manifold import ManifoldPathHandler

    pmgr.register_handler(ManifoldPathHandler())


def is_manifold_path(data_fp: str):
    return data_fp.startswith("manifold://")


def get_summary_writer(data_fp: str, comment: Optional[str] = None):
    """Get a tensorboard summary writer."""
    if FBENV and is_manifold_path(data_fp):
        from fblearner.flow.util.visualization_utils import summary_writer

        LOGGER.info(
            (
                "Storing tensorboard data on manifold. "
                "You can use 'Tensorboard On Demand' to look at your results at this URL: `%s`."
            ),
            "https://our.intern.facebook.com/intern/tensorboard/?dir=%s" % (data_fp),
        )
        return summary_writer(log_dir=data_fp, comment=comment)
    else:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(log_dir=data_fp, comment=comment)


def store_mesh_obj(mesh_path, vertices, triangles):
    with pmgr.open(mesh_path, "w") as fp:
        for v in vertices:
            fp.write("v {0} {1} {2}\n".format(v[0], v[1], v[2]))

        for t in triangles:
            fp.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(t[0] + 1, t[1] + 1, t[2] + 1))


def load_image(data_fp: str) -> torch.Tensor:
    """Load an image.

    Returns the image as a PyTorch tensor. No normalization is performed,
    no device offloading done.
    """
    with pmgr.open(data_fp, "rb") as instream:
        return torch.from_numpy(imageio.imread(instream))


def save_image(
    data_fp: str,
    image: np.ndarray,
    ttl: Optional[int] = None,
    has_user_data: Optional[bool] = None,
):
    """Save an image."""
    with pmgr.open(data_fp, "wb", ttl=ttl, has_user_data=has_user_data) as outstream:
        # imageio does not automatically detect the right format
        imageio.imwrite(outstream, image, format=".{:}".format(data_fp.split(".")[-1]))


def store_images(image_dir, imgs):
    num_frames = len(imgs)

    for frame_idx in range(num_frames):
        save_image(os.path.join(image_dir, "{0:06d}.png".format(frame_idx)), imgs[frame_idx])


def store_video(video_path, imgs, ffmpeg_bin=None, fps=2):
    num_frames = len(imgs)
    from pyvideo.ffmpeg.helpers import which

    if ffmpeg_bin is None:
        ffmpeg_bin = which("ffmpeg")
    os.environ["IMAGEIO_FFMPEG_EXE"] = ffmpeg_bin

    with pmgr.open(video_path, "wb") as outf, imageio.get_writer(
        outf,
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="libx264",
        quality=10,
        output_params=["-f", "mp4"],
    ) as out_writer:
        for frame_idx in range(num_frames):
            out_writer.append_data(imgs[frame_idx])


def get_local_cache_dir(cache_dir: str):
    if FBENV and is_manifold_path(cache_dir):
        local_cache_dir = os.path.join(get_cache_dir(), str(uuid.uuid4()))
        pmgr.mkdirs(local_cache_dir)
        return local_cache_dir
    else:
        return cache_dir


def upload_and_remove_folder(local_folder: str, remote_root: str, file_suffix: str = None, file_patterns: List[str] = None):
    # if we call this method on a local file-system there is nothing to do
    if local_folder == remote_root:
        return

    local_files = [f for f in pmgr.ls(local_folder)]

    pmgr.mkdirs(remote_root)
    for f in tqdm(local_files, desc="Upload files"):
        if file_suffix is not None and file_suffix not in f:
            continue
        if file_patterns is not None and not any([p in f for p in file_patterns]):
            continue
        local_file_path = os.path.join(local_folder, f)

        # copy file to remote & delete locally
        if pmgr.isfile(local_file_path):
            remote_file_path = os.path.join(remote_root, f)
            pmgr.copy_from_local(local_file_path, remote_file_path)
            pmgr.rm(local_file_path)

        # recursively copy subfolder
        if pmgr.isdir(local_file_path):
            remote_subfolder = os.path.join(remote_root, f)
            upload_and_remove_folder(local_file_path, remote_subfolder)


def remove_folder(folder: str):
    if FBENV and is_manifold_path(folder):
        files = [f for f in pmgr.ls(folder)]

        for f in files:
            file_path = os.path.join(folder, f)

            # delete file
            if pmgr.isfile(file_path):
                pmgr.rm(file_path)

            # recursively delete subfolder-content
            if pmgr.isdir(file_path):
                remove_folder(file_path)

            # pmgr does not offer a method to delete folders, so we don't
    else:
        shutil.rmtree(folder)
