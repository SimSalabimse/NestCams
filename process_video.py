import subprocess
import argparse
import os

def get_duration(input_file):
    """Get the duration of the input video using FFprobe."""
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_file]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, text=True)
    return float(result.stdout)

def create_timelapse(input_file, output_file, interval, fps=24):
    """Create a timelapse video by extracting frames at the specified interval."""
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f'fps=1/{interval}',
        '-r', str(fps),
        '-an',
        output_file
    ]
    subprocess.run(cmd, check=True)

def create_motion_video(input_file, output_file, target_length, fps=24, threshold=0.1):
    """Create a video with motion-detected frames, aiming for target length."""
    cmd = [
        'ffmpeg',
        '-i', input_file,
        '-vf', f"select='gt(scene,{threshold})',setpts=N/FRAME_RATE/TB",
        '-r', str(fps),
        '-an',
        output_file
    ]
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(
        description='Process a raw video (e.g., 12-hour bird nesting stream) into short and long timelapse videos.'
    )
    parser.add_argument('--input', required=True, help='Input video file (e.g., 2025-03-09_full.mp4)')
    parser.add_argument('--short_length', type=int, default=60, help='Desired length of short video in seconds (default: 60)')
    parser.add_argument('--long_length', type=int, default=360, help='Desired length of long video in seconds (default: 360)')
    parser.add_argument('--short_speed', type=float, help='Speed factor for short video (e.g., 720 for 720x)')
    parser.add_argument('--long_speed', type=float, help='Speed factor for long video (e.g., 120 for 120x)')
    parser.add_argument('--fps', type=int, default=24, help='Output frame rate (default: 24)')
    parser.add_argument('--motion_detection', action='store_true', help='Enable motion detection for long video (default: False)')
    args = parser.parse_args()

    # Get input video duration
    total_seconds = get_duration(args.input)
    print(f"Input video duration: {total_seconds:.0f} seconds")

    # Determine output paths in the same directory as input
    input_dir = os.path.dirname(args.input)
    input_base = os.path.basename(args.input)
    short_output = os.path.join(input_dir, 'short_' + input_base)
    long_output = os.path.join(input_dir, 'long_' + input_base)

    # Short video: Calculate interval from length or speed
    if args.short_speed:
        interval_short = total_seconds / (args.short_speed * args.fps)
        short_length = total_seconds / args.short_speed
    else:
        short_length = args.short_length
        N_short = short_length * args.fps
        interval_short = total_seconds / N_short
    create_timelapse(args.input, short_output, interval_short, args.fps)
    print(f"Generated {short_output} ({short_length:.0f}s, interval {interval_short:.2f}s)")

    # Long video: Calculate interval or use motion detection
    if args.long_speed:
        interval_long = total_seconds / (args.long_speed * args.fps)
        long_length = total_seconds / args.long_speed
    else:
        long_length = args.long_length
        N_long = long_length * args.fps
        interval_long = total_seconds / N_long

    if args.motion_detection:
        create_motion_video(args.input, long_output, long_length, args.fps)
        print(f"Generated {long_output} with motion detection (target {long_length}s)")
    else:
        create_timelapse(args.input, long_output, interval_long, args.fps)
        print(f"Generated {long_output} ({long_length:.0f}s, interval {interval_long:.2f}s)")

if __name__ == '__main__':
    main()