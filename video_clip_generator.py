import subprocess
import random
import sys
import os

# Check for correct command-line arguments
if len(sys.argv) != 2:
    print("Usage: python video_clip_generator.py input_video.mp4")
    sys.exit(1)

input_file = sys.argv[1]

# Ensure the input file exists
if not os.path.isfile(input_file):
    print(f"Error: Input file '{input_file}' does not exist.")
    sys.exit(1)

# Generate output file names based on input file
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_60s = f"{base_name}_60s.mp4"
output_12m = f"{base_name}_12m.mp4"

# Get video duration using ffprobe
ffprobe_cmd = [
    'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
    '-of', 'default=noprint_wrappers=1:nokey=1', input_file
]
try:
    duration_str = subprocess.check_output(ffprobe_cmd).decode().strip()
    duration = float(duration_str)
except subprocess.CalledProcessError as e:
    print(f"Error getting video duration: {e}")
    sys.exit(1)

print(f"Video duration: {duration:.2f} seconds")

# Generate 60-second clip
if duration < 60:
    print("Error: Video is too short for a 60-second clip.")
    sys.exit(1)

max_start_60 = duration - 60
start_time_60 = random.uniform(0, max_start_60)
print(f"Generating 60-second clip starting at {start_time_60:.2f} seconds")

ffmpeg_cmd_60s = [
    'ffmpeg', '-i', input_file, '-ss', str(start_time_60), '-t', '60',
    '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '128k', '-y', output_60s
]
try:
    subprocess.run(ffmpeg_cmd_60s, check=True)
    print(f"Done generating {output_60s}")
except subprocess.CalledProcessError as e:
    print(f"Error generating 60-second clip: {e}")
    sys.exit(1)

# Generate 12-minute (720-second) clip
if duration < 720:
    print("Error: Video is too short for a 12-minute clip.")
    sys.exit(1)

max_start_12m = duration - 720
start_time_12m = random.uniform(0, max_start_12m)
print(f"Generating 12-minute clip starting at {start_time_12m:.2f} seconds")

ffmpeg_cmd_12m = [
    'ffmpeg', '-i', input_file, '-ss', str(start_time_12m), '-t', '720',
    '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
    '-c:a', 'aac', '-b:a', '128k', '-y', output_12m
]
try:
    subprocess.run(ffmpeg_cmd_12m, check=True)
    print(f"Done generating {output_12m}")
except subprocess.CalledProcessError as e:
    print(f"Error generating 12-minute clip: {e}")
    sys.exit(1)