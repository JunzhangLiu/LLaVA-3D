import argparse
import os

# Default: ScanNet scans root (each scene is a subdir with `<scene_id>.sens`)
DEFAULT_SCENE_PATH = (
    "/mnt/disk4/chenyt/LLaVA-3D/playground/data/scanet_v2/"
    "OpenDataLab___ScanNet_v2/raw/scans"
)
import os, struct
import numpy as np
import zlib
import imageio
import cv2
import png

COMPRESSION_TYPE_COLOR = {-1:'unknown', 0:'raw', 1:'png', 2:'jpeg'}
COMPRESSION_TYPE_DEPTH = {-1:'unknown', 0:'raw_ushort', 1:'zlib_ushort', 2:'occi_ushort'}

class RGBDFrame():

  def load(self, file_handle):
    self.camera_to_world = np.asarray(struct.unpack('f'*16, file_handle.read(16*4)), dtype=np.float32).reshape(4, 4)
    self.timestamp_color = struct.unpack('Q', file_handle.read(8))[0]
    self.timestamp_depth = struct.unpack('Q', file_handle.read(8))[0]
    self.color_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.depth_size_bytes = struct.unpack('Q', file_handle.read(8))[0]
    self.color_data = b''.join(struct.unpack('c'*self.color_size_bytes, file_handle.read(self.color_size_bytes)))
    self.depth_data = b''.join(struct.unpack('c'*self.depth_size_bytes, file_handle.read(self.depth_size_bytes)))


  def decompress_depth(self, compression_type):
    if compression_type == 'zlib_ushort':
       return self.decompress_depth_zlib()
    else:
       raise


  def decompress_depth_zlib(self):
    return zlib.decompress(self.depth_data)


  def decompress_color(self, compression_type):
    if compression_type == 'jpeg':
       return self.decompress_color_jpeg()
    else:
       raise


  def decompress_color_jpeg(self):
    return imageio.imread(self.color_data)


class SensorData:

  def __init__(self, filename):
    self.version = 4
    self.load(filename)


  def load(self, filename):
    with open(filename, 'rb') as f:
      version = struct.unpack('I', f.read(4))[0]
      assert self.version == version
      strlen = struct.unpack('Q', f.read(8))[0]
      self.sensor_name = b''.join(struct.unpack('c'*strlen, f.read(strlen)))
      self.intrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_color = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.intrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.extrinsic_depth = np.asarray(struct.unpack('f'*16, f.read(16*4)), dtype=np.float32).reshape(4, 4)
      self.color_compression_type = COMPRESSION_TYPE_COLOR[struct.unpack('i', f.read(4))[0]]
      self.depth_compression_type = COMPRESSION_TYPE_DEPTH[struct.unpack('i', f.read(4))[0]]
      self.color_width = struct.unpack('I', f.read(4))[0]
      self.color_height =  struct.unpack('I', f.read(4))[0]
      self.depth_width = struct.unpack('I', f.read(4))[0]
      self.depth_height =  struct.unpack('I', f.read(4))[0]
      self.depth_shift =  struct.unpack('f', f.read(4))[0]
      num_frames =  struct.unpack('Q', f.read(8))[0]
      self.frames = []
      for i in range(num_frames):
        frame = RGBDFrame()
        frame.load(f)
        self.frames.append(frame)
        if num_frames > 200 and (i + 1) % max(200, num_frames // 20) == 0:
          print(f"  .sens decode progress: {i + 1}/{num_frames} frames", flush=True)


  def export_depth_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, ' depth frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      depth_data = self.frames[f].decompress_depth(self.depth_compression_type)
      depth = np.fromstring(depth_data, dtype=np.uint16).reshape(self.depth_height, self.depth_width)
      if image_size is not None:
        depth = cv2.resize(depth, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      #imageio.imwrite(os.path.join(output_path, str(f) + '.png'), depth)
      with open(os.path.join(output_path, str(f) + '.png'), 'wb') as f: # write 16-bit
        writer = png.Writer(width=depth.shape[1], height=depth.shape[0], bitdepth=16)
        depth = depth.reshape(-1, depth.shape[1]).tolist()
        writer.write(f, depth)

  def export_color_images(self, output_path, image_size=None, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    print('exporting', len(self.frames)//frame_skip, 'color frames to', output_path)
    for f in range(0, len(self.frames), frame_skip):
      color = self.frames[f].decompress_color(self.color_compression_type)
      if image_size is not None:
        color = cv2.resize(color, (image_size[1], image_size[0]), interpolation=cv2.INTER_NEAREST)
      imageio.imwrite(os.path.join(output_path, str(f) + '.jpg'), color)


  def save_mat_to_file(self, matrix, filename):
    with open(filename, 'w') as f:
      for line in matrix:
        np.savetxt(f, line[np.newaxis], fmt='%f')


  def export_poses(self, output_path, frame_skip=1):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    total = len(self.frames)
    n = (total + frame_skip - 1) // frame_skip
    print(f"exporting {n} pose files to {output_path}", flush=True)
    step = max(500, total // 10)
    for f in range(0, total, frame_skip):
      if total > 500 and f > 0 and f % step == 0:
        print(f"  pose progress: {f}/{total}", flush=True)
      self.save_mat_to_file(self.frames[f].camera_to_world, os.path.join(output_path, str(f) + '.txt'))


  def export_intrinsics(self, output_path):
    if not os.path.exists(output_path):
      os.makedirs(output_path)
    self.save_mat_to_file(self.intrinsic_color, os.path.join(output_path, 'intrinsic_color.txt'))
    self.save_mat_to_file(self.extrinsic_color, os.path.join(output_path, 'extrinsic_color.txt'))
    self.save_mat_to_file(self.intrinsic_depth, os.path.join(output_path, 'intrinsic_depth.txt'))
    self.save_mat_to_file(self.extrinsic_depth, os.path.join(output_path, 'extrinsic_depth.txt'))


def main():
    parser = argparse.ArgumentParser(
        description="Decode ScanNet .sens into depth PNGs and color JPGs under each scene's video/."
    )
    parser.add_argument(
        "--scene-path",
        type=str,
        default=DEFAULT_SCENE_PATH,
        help="Directory containing sceneXXXX_xx folders, each with sceneXXXX_xx.sens",
    )
    args = parser.parse_args()
    scene_path = os.path.abspath(os.path.expanduser(args.scene_path))
    print(f"Scene root: {scene_path}", flush=True)
    print(
        "Tip: run without `| head` and use `python -u extract_sens_data.py ...` if logs look stuck (pipe buffering).",
        flush=True,
    )

    for scene_id in sorted(os.listdir(scene_path)):
        scene_dir = os.path.join(scene_path, scene_id)
        if not os.path.isdir(scene_dir):
            continue
        # Same as original: scans/<scene_id>/video/ (depth *.png, color *.jpg)
        video_dir = os.path.join(scene_dir, "video")
        sens_path = os.path.join(scene_dir, f"{scene_id}.sens")
        if not os.path.isfile(sens_path):
            print(f"skip (no .sens): {scene_id}", flush=True)
            continue

        def _needs_pose_export(vdir):
            jpgs = [f for f in os.listdir(vdir) if f.endswith(".jpg")]
            if not jpgs:
                return False
            for j in jpgs:
                stem, _ = os.path.splitext(j)
                if not os.path.isfile(os.path.join(vdir, f"{stem}.txt")):
                    return True
            return False

        if os.path.isdir(video_dir) and len(os.listdir(video_dir)) > 0:
            if _needs_pose_export(video_dir):
                sz_gb = os.path.getsize(sens_path) / (1024**3)
                print(
                    f"backfill poses: {scene_id} (loading .sens, ~{sz_gb:.2f} GiB — often several minutes, no hang)",
                    flush=True,
                )
                try:
                    sd = SensorData(sens_path)
                except struct.error as e:
                    print(
                        f"skip backfill (corrupt or truncated .sens): {scene_id} — {e}. "
                        "Re-download the scene archive or that .sens file.",
                        flush=True,
                    )
                    continue
                print(f"loaded {len(sd.frames)} frames", flush=True)
                sd.export_poses(video_dir)
                print(f"done backfill poses: {scene_id}", flush=True)
            else:
                print(f"skip (already has frames + poses): {scene_id}", flush=True)
            continue

        sz_gb = os.path.getsize(sens_path) / (1024**3)
        print(
            f"{scene_id}: loading .sens (~{sz_gb:.2f} GiB), then exporting — this can take a long time",
            flush=True,
        )
        try:
            sd = SensorData(sens_path)
        except struct.error as e:
            print(
                f"skip export (corrupt or truncated .sens): {scene_id} — {e}. "
                "Re-download the scene archive or that .sens file.",
                flush=True,
            )
            continue
        print(f"loaded {len(sd.frames)} frames", flush=True)
        os.makedirs(video_dir, exist_ok=True)
        print("saving to", video_dir, flush=True)
        sd.export_depth_images(video_dir)
        print("saved depth", flush=True)
        sd.export_color_images(video_dir)
        print("saved color", flush=True)
        sd.export_poses(video_dir)
        print("saved poses", flush=True)


if __name__ == "__main__":
    main()

