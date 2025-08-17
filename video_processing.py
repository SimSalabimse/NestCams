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

def debug_get_selected_indices(app):
    start_time = time.time()
    log_session("Debug: Simulating motion detection")
    total_frames = 1000
    selected_indices = [i for i in range(total_frames) if i % 10 == 0]
    motion_scores = [app.motion_threshold + (i % 100) * 100 for i in range(total_frames)]
    for i in range(0, total_frames, 100):
        progress = (i / total_frames) * 100
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        remaining = (total_frames - i) / rate if rate > 0 else 0
        app.queue.put(("progress", "debug_motion_test.mp4", "Motion Detection", progress, i, total_frames, remaining))
    log_session(f"Debug: Motion detection simulated with {len(selected_indices)} indices")
    return selected_indices, motion_scores

def get_selected_indices(app, input_path, progress_callback=None):
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
    if prev_frame is None or current_frame is None:
        return 0
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(prev_gray, curr_gray)
    score = np.sum(diff > threshold, dtype=np.uint32)
    return score

def is_white_or_black_frame(app, frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    return mean_brightness > app.white_threshold or mean_brightness < app.black_threshold

def normalize_frame(app, frame, output_resolution, clip_limit, saturation_multiplier):
    try:
        frame = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * saturation_multiplier, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return frame
    except Exception as e:
        log_session(f"Error normalizing frame: {str(e)}")
        return None

def debug_normalize_frame(app, frame):
    try:
        log_session("Debug: Simulating frame normalization")
        frame = cv2.resize(frame, app.output_resolution, interpolation=cv2.INTER_AREA)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=app.clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s * app.saturation_multiplier, 0, 255).astype(np.uint8)
        hsv = cv2.merge((h, s, v))
        frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        log_session("Debug: Frame normalization successful")
        return frame
    except Exception as e:
        log_session(f"Debug: Error normalizing frame: {str(e)}")
        return None

def process_frame_batch(app, input_path, clip_limit, saturation_multiplier, rotate, temp_dir, frame_tasks, output_resolution):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        log_session(f"Cannot open {input_path} in process_frame_batch")
        return []
    results = []
    for frame_idx, order in frame_tasks:
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
    cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height', '-of', 'csv=p=0', video_path]
    output = subprocess.check_output(cmd).decode().strip()
    width, height = map(int, output.split(','))
    return width, height

def debug_generate_output_video(app, input_path, desired_duration):
    start_time = time.time()
    log_session("Debug: Simulating video generation")
    fps = 30
    total_frames = 1000
    selected_indices = [i for i in range(0, total_frames, 10)]
    target_frames_count = int(desired_duration * fps)
    if len(selected_indices) > target_frames_count:
        step = len(selected_indices) / target_frames_count
        final_indices = [selected_indices[int(i * step)] for i in range(target_frames_count)]
    else:
        final_indices = selected_indices
    app.queue.put(("status", input_path, "Debug: Processing frames..."))
    frame_counter = len(final_indices)
    for i in range(0, frame_counter, app.batch_size):
        progress = (i / frame_counter) * 100
        app.queue.put(("progress", input_path, f"Generating {desired_duration}s", progress, i, frame_counter, 0))
    app.queue.put(("status", input_path, "Debug: Creating video..."))
    if app.music_paths.get(desired_duration, app.music_paths.get("default")):
        app.queue.put(("status", input_path, "Debug: Adding music..."))
    else:
        app.queue.put(("status", input_path, "Debug: Adding silent audio..."))
    app.queue.put(("progress", input_path, f"Generating {desired_duration}s", 100, frame_counter, frame_counter, 0))
    log_session(f"Debug: Video generation simulated with {frame_counter} frames")
    return None, frame_counter, len(selected_indices), time.time() - start_time

def generate_output_video(app, input_path, output_path, desired_duration, selected_indices, progress_callback=None, status_callback=None):
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