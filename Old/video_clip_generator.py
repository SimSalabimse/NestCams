import subprocess
import random
import sys
import os
import time
import re

# Check for correct command-line arguments
if len(sys.argv) != 2:
    print("Usage: python video_clip_generator.py input_video.mp4")
    sys.exit(1)

input_file = sys.argv[1]

# Ensure the input file exists
if not os.path.isfile(input_file):
    print(f"Error: Input file '{input_file}' does not exist.")
    sys.exit(1)

# Get the directory of the input file
input_dir = os.path.dirname(input_file) if os.path.dirname(input_file) else '.'

# Generate output file names based on input file
base_name = os.path.splitext(os.path.basename(input_file))[0]
output_60s = os.path.join(input_dir, f"{base_name}_60s.mp4")
output_12m = os.path.join(input_dir, f"{base_name}_12m.mp4")

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

# Function to generate a clip with progress bar
def generate_clip(input_file, output_file, start_time, clip_duration):
    print(f"Generating clip: {output_file}")
    cmd = [
        'ffmpeg', '-i', input_file, '-ss', str(start_time), '-t', str(clip_duration),
        '-c:v', 'libx264', '-crf', '23', '-preset', 'medium',
        '-c:a', 'aac', '-b:a', '128k', '-progress', 'pipe:1', '-y', output_file
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    
    total_seconds = clip_duration
    pattern = re.compile(r'out_time_ms=(\d+)')
    
    while True:
        line = process.stdout.readline()
        if not line:
            break
        match = pattern.search(line)
        if match:
            out_time_ms = int(match.group(1))
            out_time_seconds = out_time_ms / 1_000_000
            percentage = (out_time_seconds / total_seconds) * 100
            sys.stdout.write(f"\rProgress: {percentage:.2f}%")
            sys.stdout.flush()
    process.wait()
    print()  # New line after progress
    if process.returncode != 0:
        print(f"Error generating clip: {output_file}")
        sys.exit(1)

# Record start time for total process
start_time_total = time.time()

# Generate 60-second clip
if duration < 60:
    print("Error: Video is too short for a 60-second clip.")
    sys.exit(1)
max_start_60 = duration - 60
start_time_60 = random.uniform(0, max_start_60)
generate_clip(input_file, output_60s, start_time_60, 60)

# Generate 12-minute (720-second) clip
if duration < 720:
    print("Error: Video is too short for a 12-minute clip.")
    sys.exit(1)
max_start_12m = duration - 720
start_time_12m = random.uniform(0, max_start_12m)
generate_clip(input_file, output_12m, start_time_12m, 720)

# Calculate and print total process time
end_time_total = time.time()
total_time = end_time_total - start_time_total
print(f"Total process time: {total_time:.2f} seconds")