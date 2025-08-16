import cv2
import numpy as np
import os
import time
import subprocess
import tempfile
import uuid
from multiprocessing import Pool
import functools
from PIL import Image
from utils import log_session, cancel_events, pause_event, BATCH_SIZE, WORKER_PROCESSES, thread_lock


def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute the motion score between two frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold, dtype=np.uint32)
    log_session(f"Motion score: {score}")
    return score


def is_white_or_black_frame(frame, white_threshold=200, black_threshold=50):
    """Check if a frame is predominantly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > white_threshold or mean_brightness < black_threshold


def normalize_frame(frame, output_resolution, clip_limit=1.0, saturation_multiplier=1.1):
    """Normalize and enhance a frame's visual quality."""
    try:
        frame = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        frame_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(frame_clahe, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = cv2.multiply(s, saturation_multiplier, dtype=cv2.CV_8U)
        hsv_enhanced = cv2.merge((h, s, v))
        return cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2BGR)
    except Exception as e:
        log_session(f"Error in normalize_frame: {str(e)}")
        return None


def validate_video_file(file_path):
    """Validate the integrity of a video file using FFmpeg."""
    try:
        cmd = ['ffmpeg', '-i', file_path, '-f', 'null', '-']
        subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE, check=True)
        log_session(f"Validated video file: {file_path}")
        return True
    except subprocess.CalledProcessError as e:
        log_session(f"Validation failed for {file_path}: {e.stderr.decode()}")
        return False


def check_network_stability():
    """Check network stability for YouTube upload."""
    try:
        response = requests.get("https://www.google.com", timeout=5)
        if response.status_code != 200:
            log_session(f"Network ping failed: Status {response.status_code}")
            return False
        if speedtest is None:
            log_session("No speedtest module, using ping only")
            return True
        st = speedtest.Speedtest()
        st.get_best_server()
        upload_speed = st.upload() / 1_000_000
        if upload_speed < 1.0:
            log_session(f"Upload speed too low: {upload_speed:.2f} Mbps")
            return False
        log_session(f"Network stable: Upload speed {upload_speed:.2f} Mbps")
        return True
    except Exception as e:
        log_session(f"Network check failed: {str(e)}")
        return False


def detect_orientation(video_path):
    """Detect if video is portrait or landscape."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    ret, frame = cap.read()
    if not ret:
        cap.release()
        return None
    height, width = frame.shape[:2]
    cap.release()
    return 'portrait' if height > width else 'landscape'


def get_selected_indices(input_path, motion_threshold, white_threshold, black_threshold, progress_callback=None, task_id=None):
    """Identify frames with significant motion."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log_session(f"Error: Cannot open video file: {input_path}")
        return None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    log_session(f"Total frames to process: {total_frames}")
    prev_frame_resized = None
    selected_indices = []
    start_time = time.time()
    for frame_idx in range(total_frames):
        if task_id in cancel_events and cancel_events[task_id].is_set():
            cap.release()
            log_session("Motion detection canceled by user")
            return None
        ret, frame = cap.read()
        if not ret:
            log_session(f"Failed to read frame {frame_idx} from {input_path}")
            break
        frame_resized = cv2.resize(frame, (640, 360))
        if prev_frame_resized is not None:
            motion_score = compute_motion_score(prev_frame_resized, frame_resized)
            if motion_score > motion_threshold and not is_white_or_black_frame(frame_resized, white_threshold, black_threshold):
                selected_indices.append(frame_idx)
        prev_frame_resized = frame_resized
        if frame_idx % 100 == 0 and progress_callback:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            progress = (frame_idx / total_frames) * 100
            progress_callback(progress, frame_idx, total_frames, remaining)
    cap.release()
    log_session(f"Motion detection completed for {input_path}, {len(selected_indices)} frames selected")
    return selected_indices


def process_frame_batch(input_path, clip_limit, saturation_multiplier, rotate, temp_dir, tasks, output_resolution, task_id):
    """Process a batch of frames in parallel."""
    if task_id in cancel_events and cancel_events[task_id].is_set():
        return []
    results = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log_session(f"Cannot open video file in process_frame_batch: {input_path}")
        return results
    for frame_idx, order in tasks:
        if task_id in cancel_events and cancel_events[task_id].is_set():
            cap.release()
            return results
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        normalized_frame = normalize_frame(frame, output_resolution, clip_limit, saturation_multiplier)
        if normalized_frame is None:
            continue
        if rotate:
            normalized_frame = cv2.rotate(normalized_frame, cv2.ROTATE_90_CLOCKWISE)
        temp_path = os.path.join(temp_dir, f"frame_{order:04d}.jpg")
        cv2.imwrite(temp_path, normalized_frame)
        results.append(order)
    cap.release()
    return results


def probe_video_resolution(video_path):
    """Probe the resolution of a video file using FFmpeg."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
    output = subprocess.check_output(cmd).decode().strip()
    width, height = map(int, output.split(','))
    return width, height


def generate_output_video(input_path, output_path, desired_duration, selected_indices, output_resolution, clip_limit=1.0, saturation_multiplier=1.1,
                          output_format='mp4', progress_callback=None, music_paths=None, music_volume=1.0,
                          status_callback=None, custom_ffmpeg_args=None, watermark_text=None, task_id=None):
    """Generate a video from selected frames with optional enhancements."""
    try:
        log_session(f"Generating video: {output_path} with resolution {output_resolution}")
        if status_callback:
            status_callback("Opening video file...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            log_session(f"Error: Cannot open video file: {input_path}")
            return "Failed to open video file", 0, 0, 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if total_frames <= 0 or fps <= 0:
            log_session(f"Error: Invalid video properties: total_frames={total_frames}, fps={fps}")
            return "Invalid video properties", 0, 0, 0

        orientation = detect_orientation(input_path)
        rotate = (orientation == 'portrait') or (desired_duration <= 60 and orientation != 'landscape')
        start_time = time.time()

        with tempfile.TemporaryDirectory() as temp_dir:
            target_frames_count = int(desired_duration * fps)
            if len(selected_indices) > target_frames_count:
                step = len(selected_indices) / target_frames_count
                final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
            else:
                final_indices = selected_indices

            if status_callback:
                status_callback("Saving processed frames...")
            log_session("Saving processed frames")
            frame_tasks = [(idx, i) for i, idx in enumerate(final_indices)]
            task_batches = [frame_tasks[i:i + BATCH_SIZE] for i in range(0, len(frame_tasks), BATCH_SIZE)]

            with Pool(processes=WORKER_PROCESSES) as pool:
                partial_process_frame_batch = functools.partial(
                    process_frame_batch,
                    input_path,
                    clip_limit,
                    saturation_multiplier,
                    rotate,
                    temp_dir,
                    output_resolution=output_resolution,
                    task_id=task_id
                )
                results = pool.map(partial_process_frame_batch, task_batches)
                frame_counter = sum(len(batch) for batch in results)

            log_session(f"Saved {frame_counter} frames to temporary directory")
            if status_callback:
                status_callback("Frames saved")

            if frame_counter == 0:
                log_session("Warning: No frames saved for the final video")
                return "No frames written after processing", 0, 0, 0

            # Verify first frame size
            if frame_counter > 0:
                first_frame_path = os.path.join(temp_dir, 'frame_0000.jpg')
                with Image.open(first_frame_path) as img:
                    log_session(f"First frame size: {img.size}")

            num_frames = frame_counter
            new_fps = num_frames / desired_duration if num_frames < target_frames_count else fps
            log_session(f"Parameters: num_frames={num_frames}, target_frames_count={target_frames_count}, new_fps={new_fps:.2f}")

            if status_callback:
                status_callback("Creating video from frames...")
            log_session("Creating video with FFmpeg")
            temp_final_path = f"temp_final_{uuid.uuid4().hex}.{output_format}"

            cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg'), '-s', f"{output_resolution[0]}x{output_resolution[1]}"]
            try:
                subprocess.run(['ffmpeg', '-hwaccels'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
            except subprocess.CalledProcessError:
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
            cmd.extend(['-pix_fmt', 'yuv420p', '-r', str(new_fps)])
            if watermark_text:
                cmd.extend(['-vf', f'drawtext=text={watermark_text}:fontcolor=white:fontsize=24:x=10:y=10'])
            if custom_ffmpeg_args:
                cmd.extend(custom_ffmpeg_args)
            cmd.extend(['-y', temp_final_path])
            subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            log_session("Video created from frames")

            # Probe temp video resolution
            width, height = probe_video_resolution(temp_final_path)
            log_session(f"Temp video resolution: {width}x{height}")

            music_path = music_paths.get(desired_duration, music_paths.get("default")) if music_paths else None
            if music_path and os.path.exists(music_path):
                if status_callback:
                    status_callback("Adding music...")
                log_session("Adding music with FFmpeg")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-stream_loop', '-1', '-i', music_path,
                    '-filter_complex', f"[1:a]volume={music_volume}[a]",
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                log_session(f"Music added to {output_path}")
            else:
                if status_callback:
                    status_callback("Adding silent audio...")
                log_session("Adding silent audio with FFmpeg")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                log_session(f"Silent audio added to {output_path}")

            # Probe final video resolution
            width, height = probe_video_resolution(output_path)
            log_session(f"Final video resolution: {width}x{height}")

            os.remove(temp_final_path)
            if progress_callback:
                progress_callback(100, frame_counter, len(final_indices), 0)
            log_session(f"Video generation completed: {output_path}")
            return None, frame_counter, len(selected_indices), time.time() - start_time

    except Exception as e:
        log_session(f"Error in generate_output_video: {str(e)}")
        return f"Error: {str(e)}", 0, 0, 0