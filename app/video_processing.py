# video_processing.py
import cv2
import numpy as np
from multiprocessing import Pool, Event
import os
import subprocess
import uuid
import tempfile
from utils import temp_dir, log_session
import psutil
try:
    import torch
except ImportError:
    torch = None

def compute_motion_score(prev_frame, curr_frame, threshold):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold)
    return score

def optical_flow_motion(prev_frame, curr_frame, flow_threshold=0.5):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag) > flow_threshold

def is_white_or_black_frame(frame, white_threshold, black_threshold):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    return mean > white_threshold or mean < black_threshold

def get_selected_indices(video_path, motion_threshold, white_threshold, black_threshold, flow_threshold, total_frames):
    cap = cv2.VideoCapture(video_path)
    indices = []
    prev_frame = None
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        if prev_frame is not None:
            score = compute_motion_score(prev_frame, frame, motion_threshold)
            if score > motion_threshold or optical_flow_motion(prev_frame, frame, flow_threshold):
                if not is_white_or_black_frame(frame, white_threshold, black_threshold):
                    indices.append(i)
        prev_frame = frame
    cap.release()
    return indices

def normalize_frame(frame, resolution, clip_limit, saturation_boost):
    frame = cv2.resize(frame, resolution)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = hsv[..., 1] * saturation_boost
    frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return frame

def process_frame_batch(args):
    video_path, indices, resolution, clip_limit, saturation_boost = args
    cap = cv2.VideoCapture(video_path)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            if cv2.cuda.getCudaEnabledDeviceCount() > 0:
                gpu_frame = cv2.cuda_GpuMat()
                gpu_frame.upload(frame)
                # GPU operations if possible
                frame = gpu_frame.download()  # Placeholder
            frame = normalize_frame(frame, resolution, clip_limit, saturation_boost)
            frames.append(frame)
    cap.release()
    return frames

def generate_output_video(processed_frames, output_path, fps, music_path, volume, watermark, custom_args, fade_in_out=0):
    if len(processed_frames) == 0:
        return
    height, width = processed_frames[0].shape[:2]
    use_ffmpeg = True
    if not music_path and not watermark:  # Simple case, use OpenCV
        use_ffmpeg = False
        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in processed_frames:
            writer.write(frame)
        writer.release()
    if use_ffmpeg:
        temp_video = os.path.join(temp_dir, f"{uuid.uuid4()}.mp4")
        writer = cv2.VideoWriter(temp_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        for frame in processed_frames:
            writer.write(frame)
        writer.release()
        cmd = ["ffmpeg", "-i", temp_video]
        if music_path:
            cmd += ["-stream_loop", "-1", "-i", music_path, "-filter_complex", f"[1:a]volume={volume},afade=t=in:st=0:d={fade_in_out},afade=t=out:st={len(processed_frames)/fps - fade_in_out}:d={fade_in_out}[a]", "-map", "0:v", "-map", "[a]", "-shortest"]
        else:
            cmd += ["-an"]
        if watermark:
            cmd += ["-vf", f"drawtext=text='{watermark}':x=10:y=10:fontsize=24:fontcolor=white"]
        cmd += custom_args.split() if custom_args else []
        cmd += ["-c:v", "h264_nvenc" if subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True).stdout.find("nvenc") != -1 else "libx264", "-y", output_path]
        subprocess.run(cmd, check=True)
        os.remove(temp_video)

def process_single_video(video_path, settings, progress_callback, cancel_event, pause_event):
    if not validate_video_file(video_path):
        raise ValueError("Invalid video")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    indices = get_selected_indices(video_path, settings['motion_threshold'], settings['white_threshold'], settings['black_threshold'], settings['flow_threshold'], total_frames)

    # Subsample to fit duration
    target_frames = int(settings['duration'] * fps)
    if len(indices) > target_frames:
        step = len(indices) // target_frames
        indices = indices[::step][:target_frames]

    batch_size = settings['batch_size']
    resolution = tuple(map(int, settings['resolution'].split('x')))
    # Auto-detect input res
    cap = cv2.VideoCapture(video_path)
    input_res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    if input_res[0] > 1920:  # 4K+
        # Offer upscale/downscale, but for now downscale
        resolution = (1920, 1080)

    # Detect video type if torch
    if torch:
        # Simple ML detection, placeholder
        pass

    batches = [indices[i:i+batch_size] for i in range(0, len(indices), batch_size)]
    pool_args = [(video_path, batch, resolution, settings['clip_limit'], settings['saturation_boost']) for batch in batches]

    processed_frames = []
    with Pool(processes=min(settings['workers'], psutil.cpu_count() - 1)) as pool:
        for i, batch_frames in enumerate(pool.imap_unordered(process_frame_batch, pool_args)):
            if cancel_event.is_set():
                raise Exception("Cancelled")
            while pause_event.is_set():
                time.sleep(1)
            processed_frames.extend(batch_frames)
            progress = (i + 1) / len(batches) * 100
            progress_callback(progress)

    output_path = os.path.join(settings['output_dir'], f"{os.path.basename(video_path)}_processed.mp4")
    music_path = settings['music_paths'].get(str(settings['duration']), None)
    generate_output_video(processed_frames, output_path, fps, music_path, settings['music_volume'], settings['watermark'], settings['custom_ffmpeg_args'], settings['fade_in_out'])

    # Analytics
    stats = {'frames_processed': len(processed_frames), 'motion_events': len(indices), 'time': time.time()}
    return output_path, stats

# Debug functions
def debug_get_selected_indices(video_path, *args):
    return [0, 10, 20]

def debug_normalize_frame(frame, *args):
    return frame

def debug_generate_output_video(*args):
    logging.info("Simulating video generation")