# import subprocess
# import argparse
# import os
# import sys
# import time
# import re

# def get_duration(input_file):
#     """Get the duration of the input video using FFprobe."""
#     cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
#     return float(result.stdout)

# def parse_time(time_str):
#     """Parse time string from FFmpeg output (e.g., '00:01:23.45') to seconds."""
#     h, m, s = time_str.split(':')
#     return int(h) * 3600 + int(m) * 60 + float(s)

# def run_ffmpeg_with_progress(cmd, total_duration):
#     """Run FFmpeg command with progress feedback."""
#     process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
#     start_time = time.time()
#     time_pattern = re.compile(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})')

#     print("Starting FFmpeg process...")
#     for line in process.stderr:
#         match = time_pattern.search(line)
#         if match:
#             current_time = parse_time(match.group(1))
#             progress = (current_time / total_duration) * 100
#             elapsed = time.time() - start_time
#             if progress > 0:
#                 estimated_total = elapsed / (progress / 100)
#                 remaining = estimated_total - elapsed
#                 sys.stdout.write(f"\rProgress: {progress:.2f}% | Estimated remaining: {remaining:.0f}s")
#                 sys.stdout.flush()
#     process.wait()
#     print("\nFFmpeg process completed.")

# def create_timelapse(input_file, output_file, interval, fps=24):
#     """Create a timelapse video by extracting frames at the specified interval."""
#     cmd = [
#         'ffmpeg',
#         '-i', input_file,
#         '-vf', f'fps=1/{interval}',
#         '-r', str(fps),
#         '-an',
#         output_file
#     ]
#     return cmd

# def create_motion_video(input_file, output_file, target_length, fps=24, threshold=0.1):
#     """Create a video with motion-detected frames, aiming for target length."""
#     cmd = [
#         'ffmpeg',
#         '-i', input_file,
#         '-vf', f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
#         '-r', str(fps),
#         '-an',
#         output_file
#     ]
#     return cmd

# def main():
#     parser = argparse.ArgumentParser(
#         description='Process a raw video (e.g., 12-hour bird nesting stream) into short and long timelapse videos.'
#     )
#     parser.add_argument('--input', required=True, help='Input video file (e.g., 2025-03-09_full.mp4)')
#     parser.add_argument('--short_length', type=int, default=60, help='Desired length of short video in seconds (default: 60)')
#     parser.add_argument('--long_length', type=int, default=360, help='Desired length of long video in seconds (default: 360)')
#     parser.add_argument('--short_speed', type=float, help='Speed factor for short video (e.g., 720 for 720x)')
#     parser.add_argument('--long_speed', type=float, help='Speed factor for long video (e.g., 120 for 120x)')
#     parser.add_argument('--fps', type=int, default=24, help='Output frame rate (default: 24)')
#     parser.add_argument('--motion_detection', action='store_true', help='Enable motion detection for long video (default: False)')
#     args = parser.parse_args()

#     # Get input video duration
#     total_seconds = get_duration(args.input)
#     print(f"Input video duration: {total_seconds:.0f} seconds")

#     # Determine output paths in the same directory as input
#     input_dir = os.path.dirname(args.input)
#     input_base = os.path.basename(args.input)
#     short_output = os.path.join(input_dir, 'short_' + input_base)
#     long_output = os.path.join(input_dir, 'long_' + input_base)

#     # Short video: Calculate interval from length or speed
#     if args.short_speed:
#         interval_short = total_seconds / (args.short_speed * args.fps)
#         short_length = total_seconds / args.short_speed
#     else:
#         short_length = args.short_length
#         N_short = short_length * args.fps
#         interval_short = total_seconds / N_short
#     cmd_short = create_timelapse(args.input, short_output, interval_short, args.fps)
#     print(f"Generating short video: {short_output}")
#     run_ffmpeg_with_progress(cmd_short, total_seconds)

#     # Long video: Calculate interval or use motion detection
#     if args.long_speed:
#         interval_long = total_seconds / (args.long_speed * args.fps)
#         long_length = total_seconds / args.long_speed
#     else:
#         long_length = args.long_length
#         N_long = long_length * args.fps
#         interval_long = total_seconds / N_long

#     if args.motion_detection:
#         cmd_long = create_motion_video(args.input, long_output, long_length, args.fps)
#         print(f"Generating long video with motion detection: {long_output}")
#     else:
#         cmd_long = create_timelapse(args.input, long_output, interval_long, args.fps)
#         print(f"Generating long video: {long_output}")
#     run_ffmpeg_with_progress(cmd_long, total_seconds)

#     print("All processing completed.")

# if __name__ == '__main__':
#     main()

















# import subprocess
# import argparse
# import os
# import sys
# import time
# import re

# def get_duration(input_file):
#     """Get the duration of the input video using FFprobe."""
#     cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
#     return float(result.stdout)

# def parse_time(time_str):
#     """Parse time string from FFmpeg output (e.g., '00:01:23.45') to seconds."""
#     h, m, s = time_str.split(':')
#     return int(h) * 3600 + int(m) * 60 + float(s)

# def run_ffmpeg_with_progress(cmd, total_duration):
#     """Run FFmpeg command with progress feedback."""
#     process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
#     start_time = time.time()
#     time_pattern = re.compile(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})')

#     print("Starting FFmpeg process...")
#     for line in process.stderr:
#         match = time_pattern.search(line)
#         if match:
#             current_time = parse_time(match.group(1))
#             progress = (current_time / total_duration) * 100
#             elapsed = time.time() - start_time
#             if progress > 0:
#                 estimated_total = elapsed / (progress / 100)
#                 remaining = estimated_total - elapsed
#                 sys.stdout.write(f"\rProgress: {progress:.2f}% | Estimated remaining: {remaining:.0f}s")
#                 sys.stdout.flush()
#     process.wait()
#     print("\nFFmpeg process completed.")

# def create_timelapse(input_file, output_file, speed_factor, fps=24):
#     """Create a timelapse video by speeding up the input video."""
#     cmd = [
#         'ffmpeg',
#         '-i', input_file,
#         '-vf', f'setpts=PTS/{speed_factor}',
#         '-r', str(fps),
#         '-an',
#         output_file
#     ]
#     return cmd

# def create_motion_video(input_file, output_file, target_length, fps=24, threshold=0.1):
#     """Create a video with motion-detected frames, aiming for target length."""
#     cmd = [
#         'ffmpeg',
#         '-i', input_file,
#         '-vf', f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
#         '-r', str(fps),
#         '-an',
#         output_file
#     ]
#     return cmd

# def main():
#     parser = argparse.ArgumentParser(
#         description='Process a raw video (e.g., 12-hour bird nesting stream) into short and long timelapse videos.'
#     )
#     parser.add_argument('--input', required=True, help='Input video file (e.g., 2025-03-09_full.mp4)')
#     parser.add_argument('--short_length', type=int, default=60, help='Desired length of short video in seconds (default: 60)')
#     parser.add_argument('--long_length', type=int, default=360, help='Desired length of long video in seconds (default: 360)')
#     parser.add_argument('--short_speed', type=float, help='Speed factor for short video (e.g., 720 for 720x)')
#     parser.add_argument('--long_speed', type=float, help='Speed factor for long video (e.g., 120 for 120x)')
#     parser.add_argument('--fps', type=int, default=24, help='Output frame rate (default: 24)')
#     parser.add_argument('--motion_detection', action='store_true', help='Enable motion detection for long video (default: False)')
#     args = parser.parse_args()

#     # Get input video duration
#     total_seconds = get_duration(args.input)
#     print(f"Input video duration: {total_seconds:.0f} seconds")

#     # Determine output paths in the same directory as input
#     input_dir = os.path.dirname(args.input)
#     input_base = os.path.basename(args.input)
#     short_output = os.path.join(input_dir, 'short_' + input_base)
#     long_output = os.path.join(input_dir, 'long_' + input_base)

#     # Short video: Calculate speed_factor
#     if args.short_speed:
#         speed_factor_short = args.short_speed
#         short_length = total_seconds / speed_factor_short
#     else:
#         short_length = args.short_length
#         speed_factor_short = total_seconds / short_length
#     cmd_short = create_timelapse(args.input, short_output, speed_factor_short, args.fps)
#     print(f"Generating short video: {short_output} with speed factor {speed_factor_short:.2f}")
#     run_ffmpeg_with_progress(cmd_short, total_seconds)

#     # Long video: Calculate speed_factor
#     if args.long_speed:
#         speed_factor_long = args.long_speed
#         long_length = total_seconds / speed_factor_long
#     else:
#         long_length = args.long_length
#         speed_factor_long = total_seconds / long_length

#     if args.motion_detection:
#         print("Note: Motion detection is enabled, but the current implementation may not speed up the video as expected.")
#         cmd_long = create_motion_video(args.input, long_output, long_length, args.fps)
#         print(f"Generating long video with motion detection: {long_output}")
#     else:
#         cmd_long = create_timelapse(args.input, long_output, speed_factor_long, args.fps)
#         print(f"Generating long video: {long_output} with speed factor {speed_factor_long:.2f}")
#     run_ffmpeg_with_progress(cmd_long, total_seconds)

#     print("All processing completed.")

# if __name__ == '__main__':
#     main()






# import subprocess
# import argparse
# import os
# import sys
# import time
# import re

# def get_duration(input_file):
#     """Get the duration of the input video using FFprobe."""
#     cmd = [
#         'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
#         '-of', 'default=noprint_wrappers=1:nokey=1', input_file
#     ]
#     result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
#     return float(result.stdout)

# def parse_time(time_str):
#     """Parse time string from FFmpeg output (e.g., '00:01:23.45') to seconds."""
#     h, m, s = time_str.split(':')
#     return int(h) * 3600 + int(m) * 60 + float(s)

# def run_ffmpeg_with_progress(cmd, total_duration, speed_factor=None):
#     """
#     Run FFmpeg command with progress feedback.
    
#     Args:
#         cmd (list): FFmpeg command as a list.
#         total_duration (float): Input video duration in seconds.
#         speed_factor (float, optional): Speed factor for timelapse; if None, use fallback method.
#     """
#     process = subprocess.Popen(cmd, stderr=subprocess.PIPE, universal_newlines=True)
#     start_time = time.time()
#     time_pattern = re.compile(r'time=(\d{2}:\d{2}:\d{2}\.\d{2})')

#     print("Starting FFmpeg process...")
#     for line in process.stderr:
#         match = time_pattern.search(line)
#         if match:
#             current_time = parse_time(match.group(1))
#             if speed_factor:
#                 # Adjusted progress: input time processed = output time * speed_factor
#                 t_input_processed = current_time * speed_factor
#                 progress = (t_input_processed / total_duration) * 100
#                 method = "adjusted"
#             else:
#                 # Fallback: use output time directly (less accurate)
#                 progress = (current_time / total_duration) * 100
#                 method = "fallback"
#             elapsed = time.time() - start_time
#             if progress > 0:
#                 estimated_total = elapsed / (progress / 100)
#                 remaining = estimated_total - elapsed
#                 sys.stdout.write(
#                     f"\rProgress: {progress:.2f}% ({method}) | Estimated remaining: {remaining:.0f}s"
#                 )
#                 sys.stdout.flush()
#     process.wait()
#     print("\nFFmpeg process completed.")

# def create_timelapse(input_file, output_file, speed_factor, fps=24):
#     """Create a timelapse video by speeding up the input video."""
#     cmd = [
#         'ffmpeg', '-i', input_file,
#         '-vf', f'setpts=PTS/{speed_factor}',
#         '-r', str(fps), '-an', output_file
#     ]
#     return cmd

# def create_motion_video(input_file, output_file, target_length, fps=24, threshold=0.1):
#     """Create a video with motion-detected frames, aiming for target length."""
#     cmd = [
#         'ffmpeg', '-i', input_file,
#         '-vf', f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
#         '-r', str(fps), '-an', output_file
#     ]
#     return cmd

# def main():
#     parser = argparse.ArgumentParser(
#         description='Process a video into short and long timelapse videos.'
#     )
#     parser.add_argument('--input', required=True, help='Input video file')
#     parser.add_argument('--short_length', type=int, default=60, help='Short video length in seconds')
#     parser.add_argument('--long_length', type=int, default=360, help='Long video length in seconds')
#     parser.add_argument('--short_speed', type=float, help='Speed factor for short video')
#     parser.add_argument('--long_speed', type=float, help='Speed factor for long video')
#     parser.add_argument('--fps', type=int, default=24, help='Output frame rate')
#     parser.add_argument('--motion_detection', action='store_true', help='Enable motion detection for long video')
#     args = parser.parse_args()

#     # Get input video duration
#     total_seconds = get_duration(args.input)
#     print(f"Input video duration: {total_seconds:.0f} seconds")

#     # Determine output paths
#     input_dir = os.path.dirname(args.input)
#     input_base = os.path.basename(args.input)
#     short_output = os.path.join(input_dir, 'short_' + input_base)
#     long_output = os.path.join(input_dir, 'long_' + input_base)

#     # Short video: Calculate speed_factor
#     if args.short_speed:
#         speed_factor_short = args.short_speed
#     else:
#         speed_factor_short = total_seconds / args.short_length
#     cmd_short = create_timelapse(args.input, short_output, speed_factor_short, args.fps)
#     print(f"Generating short video: {short_output} with speed factor {speed_factor_short:.2f}")
#     run_ffmpeg_with_progress(cmd_short, total_seconds, speed_factor=speed_factor_short)

#     # Long video: Calculate speed_factor or use motion detection
#     if args.long_speed:
#         speed_factor_long = args.long_speed
#     else:
#         speed_factor_long = total_seconds / args.long_length

#     if args.motion_detection:
#         cmd_long = create_motion_video(args.input, long_output, args.long_length, args.fps)
#         print(f"Generating long video with motion detection: {long_output}")
#         run_ffmpeg_with_progress(cmd_long, total_seconds)  # No speed_factor
#     else:
#         cmd_long = create_timelapse(args.input, long_output, speed_factor_long, args.fps)
#         print(f"Generating long video: {long_output} with speed factor {speed_factor_long:.2f}")
#         run_ffmpeg_with_progress(cmd_long, total_seconds, speed_factor=speed_factor_long)

#     print("All processing completed.")

# if __name__ == '__main__':
#     main()










