import cv2
import numpy as np
import os
import time
import tempfile
import subprocess
import uuid
import functools
from multiprocessing import Pool
from utils import log_session

def process_single_video(app, input_file, selected_videos, output_format, total_tasks, task_count_queue):
    """Process single video."""
    try:
        base, _ = os.path.splitext(input_file)
        output_files = {}
        task_count = task_count_queue.get()
        app.queue.put(("task_start", input_file, "Motion Detection", task_count / total_tasks * 100))
        task_count += 1
        def motion_progress_callback(progress, current, total, remaining):
            app.queue.put(("progress", input_file, "Motion Detection", progress, current, total, remaining))
        selected_indices, motion_scores = get_selected_indices(app, input_file, motion_progress_callback)
        if selected_indices is None:
            app.queue.put(("canceled", input_file, "Processing canceled"))
            return None
        if not selected_indices:
            app.queue.put(("canceled", input_file, "No frames selected"))
            return None
        for task_name, duration in selected_videos:
            if app.cancel_event.is_set():
                app.queue.put(("canceled", input_file, "Canceled"))
                return None
            output_file = f"{base}_{task_name}.{output_format}"
            if app.output_dir:
                output_file = os.path.join(app.output_dir, os.path.basename(output_file))
            app.queue.put(("task_start", input_file, f"Generating {task_name}", task_count / total_tasks * 100))
            task_count += 1
            def progress_callback(progress, current, total, remaining):
                app.queue.put(("progress", input_file, f"Generating {task_name}", progress, current, total, remaining))
            def status_callback(status):
                app.queue.put(("status", input_file, status))
            error, frames_processed, motion_events, proc_time = generate_output_video(
                app, input_file, output_file, duration, selected_indices, progress_callback, status_callback
            )
            if error:
                app.queue.put(("canceled", input_file, error))
                return None
            output_files[task_name] = output_file
            app.analytics_data.append({
                "file": os.path.basename(input_file),
                "duration": duration,
                "frames_processed": frames_processed,
                "motion_events": motion_events,
                "processing_time": proc_time,
                "motion_scores": motion_scores
            })
        task_count_queue.put(task_count)
        return output_files
    except Exception as e:
        app.queue.put(("canceled", input_file, str(e)))
        return None

def get_selected_indices(app, input_path, progress_callback=None):
    """Identify frames with motion."""
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log_session(f"Cannot open {input_path}")
        return None, None
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev_frame_resized = None
    selected_indices = []
    motion_scores = []
    start_time = time.time()
    for frame_idx in range(total_frames):
        if app.cancel_event.is_set():
            cap.release()
            return None, None
        while app.pause_event.is_set() and not app.cancel_event.is_set():
            time.sleep(0.1)
        ret, frame = cap.read()
        if not ret:
            break
        frame_resized = cv2.resize(frame, (640, 360))
        if prev_frame_resized is not None:
            score = compute_motion_score(prev_frame_resized, frame_resized)
            motion_scores.append(score)
            if score > app.motion_threshold and not is_white_or_black_frame(app, frame_resized):
                selected_indices.append(frame_idx)
        prev_frame_resized = frame_resized
        if frame_idx % 100 == 0 and progress_callback:
            elapsed = time.time() - start_time
            rate = frame_idx / elapsed if elapsed > 0 else 0
            remaining = (total_frames - frame_idx) / rate if rate > 0 else 0
            progress = (frame_idx / total_frames) * 100
            progress_callback(progress, frame_idx, total_frames, remaining)
    cap.release()
    return selected_indices, motion_scores

def compute_motion_score(prev_frame, current_frame, threshold=30):
    """Compute motion score between frames."""
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold, dtype=np.uint32)
    return score

def is_white_or_black_frame(app, frame):
    """Check if frame is overly white or black."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > app.white_threshold or mean_brightness < app.black_threshold

def normalize_frame(app, frame, output_resolution, clip_limit, saturation_multiplier):
    """Normalize and enhance frame."""
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
        log_session(f"Frame normalization error: {str(e)}")
        return None

def process_frame_batch(app, input_path, clip_limit, saturation_multiplier, rotate, temp_dir, tasks, output_resolution):
    """Process batch of frames."""
    results = []
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return results
    for frame_idx, order in tasks:
        if app.cancel_event.is_set():
            cap.release()
            return results
        while app.pause_event.is_set() and not app.cancel_event.is_set():
            time.sleep(0.1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
        normalized_frame = normalize_frame(app, frame, output_resolution, clip_limit, saturation_multiplier)
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
    """Probe video resolution using FFmpeg."""
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
    output = subprocess.check_output(cmd).decode().strip()
    width, height = map(int, output.split(','))
    return width, height

def generate_output_video(app, input_path, output_path, desired_duration, selected_indices, progress_callback=None, status_callback=None):
    """Generate output video."""
    try:
        if status_callback:
            status_callback("Opening video...")
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            return "Failed to open video", 0, 0, 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if total_frames <= 0 or fps <= 0:
            return "Invalid video properties", 0, 0, 0
        width, height = probe_video_resolution(input_path)
        rotate = desired_duration <= 60 and width > height
        start_time = time.time()
        with tempfile.TemporaryDirectory() as temp_dir:
            target_frames_count = int(desired_duration * fps)
            if len(selected_indices) > target_frames_count:
                step = len(selected_indices) / target_frames_count
                final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
            else:
                final_indices = selected_indices
            if status_callback:
                status_callback("Processing frames...")
            frame_tasks = [(idx, i) for i, idx in enumerate(final_indices)]
            task_batches = [frame_tasks[i:i + app.batch_size] for i in range(0, len(frame_tasks), app.batch_size)]
            with Pool(processes=app.worker_processes) as pool:
                partial_process = functools.partial(
                    process_frame_batch,
                    app,
                    input_path,
                    app.clip_limit,
                    app.saturation_multiplier,
                    rotate,
                    temp_dir,
                    output_resolution=app.output_resolution
                )
                results = pool.map(partial_process, task_batches)
                frame_counter = sum(len(batch) for batch in results)
            if frame_counter == 0:
                return "No frames processed", 0, 0, 0
            num_frames = frame_counter
            new_fps = num_frames / desired_duration if num_frames < target_frames_count else fps
            if status_callback:
                status_callback("Creating video...")
            temp_final_path = f"temp_final_{uuid.uuid4().hex}.{app.output_format_var.get()}"
            cmd = ['ffmpeg', '-framerate', str(new_fps), '-i', os.path.join(temp_dir, 'frame_%04d.jpg'), '-s', f"{app.output_resolution[0]}x{app.output_resolution[1]}"]
            try:
                subprocess.run(['ffmpeg', '-hwaccels'], check=True)
                cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
            except:
                cmd.extend(['-c:v', 'libx264', '-preset', 'fast'])
            cmd.extend(['-pix_fmt', 'yuv420p', '-r', str(new_fps)])
            if app.watermark_text:
                cmd.extend(['-vf', f'drawtext=text={app.watermark_text}:fontcolor=white:fontsize=24:x=10:y=10'])
            if app.custom_ffmpeg_args:
                cmd.extend(app.custom_ffmpeg_args)
            cmd.extend(['-y', temp_final_path])
            subprocess.run(cmd, check=True)
            music_path = app.music_paths.get(desired_duration, app.music_paths.get("default"))
            if music_path and os.path.exists(music_path):
                if status_callback:
                    status_callback("Adding music...")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-stream_loop', '-1', '-i', music_path,
                    '-filter_complex', f"[1:a]volume={app.music_volume}[a]",
                    '-map', '0:v', '-map', '[a]', '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True)
            else:
                if status_callback:
                    status_callback("Adding silent audio...")
                cmd = [
                    'ffmpeg', '-i', temp_final_path, '-f', 'lavfi', '-i', 'anullsrc=channel_layout=stereo:sample_rate=44100',
                    '-c:v', 'copy', '-c:a', 'aac', '-shortest', '-y', output_path
                ]
                subprocess.run(cmd, check=True)
            os.remove(temp_final_path)
            if progress_callback:
                progress_callback(100, frame_counter, len(final_indices), 0)
            return None, frame_counter, len(selected_indices), time.time() - start_time
    except Exception as e:
        log_session(f"Video generation error: {str(e)}")
        return str(e), 0, 0, 0