"""
Video Processor Module
Creates time-lapse videos from motion segments with speed adjustment
"""

import cv2
import subprocess
import tempfile
import os
import logging
from typing import List, Tuple, Callable, Optional
from pathlib import Path
import platform

logger = logging.getLogger(__name__)


class VideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.target_length = config.get('target_length', 60)
        self.smoothing = config.get('smoothing', 5)
        self.use_gpu = config.get('use_gpu', True)
        self.cpu_threads = config.get('cpu_threads', os.cpu_count() - 1)
        self.quality = config.get('quality', 2)  # 0=low, 1=med, 2=high, 3=max
        self.add_music = config.get('add_music', False)
        self.music_path = config.get('music_path')
        
        # Detect hardware capabilities
        self.hw_accel = self._detect_hardware_acceleration()
        
        logger.info(f"Video processor initialized: target={self.target_length}s, "
                   f"quality={self.quality}, hw_accel={self.hw_accel}")
    
    def _detect_hardware_acceleration(self) -> str:
        """Detect available hardware acceleration"""
        system = platform.system()
        
        if not self.use_gpu:
            return 'cpu'
        
        # Check for NVIDIA GPU
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                logger.info("NVIDIA GPU detected")
                return 'cuda'
        except:
            pass
        
        # Check for Intel Quick Sync (Windows/Linux)
        if system in ['Windows', 'Linux']:
            try:
                # Check for Intel GPU
                result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                      capture_output=True, text=True, timeout=2)
                if 'h264_qsv' in result.stdout:
                    logger.info("Intel Quick Sync detected")
                    return 'qsv'
            except:
                pass
        
        # Check for Apple VideoToolbox (Mac)
        if system == 'Darwin':
            try:
                result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                      capture_output=True, text=True, timeout=2)
                if 'h264_videotoolbox' in result.stdout:
                    logger.info("Apple VideoToolbox detected")
                    return 'videotoolbox'
            except:
                pass
        
        logger.info("No hardware acceleration detected, using CPU")
        return 'cpu'
    
    def create_timelapse(self, input_path: str, motion_segments: List[Tuple[float, float]], 
                        output_path: str, progress_callback: Optional[Callable] = None) -> bool:
        """
        Create time-lapse video from motion segments
        
        Args:
            input_path: Path to input video
            motion_segments: List of (start, end) time tuples with motion
            output_path: Path for output video
            progress_callback: Optional progress callback
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Calculate total motion duration
            total_motion_time = sum(end - start for start, end in motion_segments)
            
            if total_motion_time == 0:
                logger.error("No motion segments to process")
                return False
            
            # Calculate required speedup
            speedup_factor = total_motion_time / self.target_length
            
            # Warn if speedup is less than 1 (would need to slow down)
            if speedup_factor < 1.0:
                logger.warning(f"Motion duration ({total_motion_time:.1f}s) is shorter than "
                             f"target ({self.target_length}s). Video will not be sped up.")
                speedup_factor = 1.0
            
            logger.info(f"Speedup factor: {speedup_factor:.2f}x")
            
            # Extract motion segments to temporary files
            temp_dir = tempfile.mkdtemp()
            segment_files = []
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            for i, (start, end) in enumerate(motion_segments):
                if progress_callback:
                    progress = (i / len(motion_segments)) * 50
                    progress_callback(progress)
                
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                self._extract_segment(input_path, start, end, segment_path, speedup_factor)
                segment_files.append(segment_path)
            
            # Concatenate segments
            if progress_callback:
                progress_callback(60)
            
            concat_path = os.path.join(temp_dir, "concat.mp4")
            self._concatenate_segments(segment_files, concat_path)
            
            # Add music if requested
            final_input = concat_path
            if self.add_music and self.music_path and os.path.exists(self.music_path):
                if progress_callback:
                    progress_callback(80)
                music_path = os.path.join(temp_dir, "with_music.mp4")
                self._add_music(concat_path, self.music_path, music_path)
                final_input = music_path
            
            # Apply final encoding with quality settings
            if progress_callback:
                progress_callback(90)
            
            self._encode_final(final_input, output_path)
            
            # Cleanup
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            logger.info(f"Time-lapse created successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.exception("Error creating time-lapse")
            return False
    
    def _extract_segment(self, input_path: str, start: float, end: float, 
                        output_path: str, speedup: float):
        """Extract and speed up a video segment"""
        duration = end - start
        
        # Build FFmpeg command
        cmd = ['ffmpeg', '-y', '-ss', str(start), '-i', input_path, 
               '-t', str(duration)]
        
        # Apply speedup using setpts filter
        if speedup > 1.0:
            # Calculate new PTS (presentation timestamp)
            pts_factor = 1.0 / speedup
            cmd.extend(['-filter:v', f'setpts={pts_factor}*PTS'])
        
        # Hardware acceleration
        if self.hw_accel == 'cuda':
            cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'fast'])
        elif self.hw_accel == 'qsv':
            cmd.extend(['-c:v', 'h264_qsv', '-preset', 'fast'])
        elif self.hw_accel == 'videotoolbox':
            cmd.extend(['-c:v', 'h264_videotoolbox'])
        else:
            cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast'])
        
        cmd.extend(['-an', output_path])  # Remove audio for now
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _concatenate_segments(self, segment_files: List[str], output_path: str):
        """Concatenate video segments"""
        # Create concat file
        concat_file = output_path.replace('.mp4', '_list.txt')
        with open(concat_file, 'w') as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")
        
        # Concatenate
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
               '-i', concat_file, '-c', 'copy', output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)
    
    def _add_music(self, video_path: str, music_path: str, output_path: str):
        """Add background music to video"""
        # Get video duration
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        # Add music, loop if needed, fade out at end
        cmd = ['ffmpeg', '-y', '-i', video_path, '-stream_loop', '-1', 
               '-i', music_path, '-t', str(video_duration),
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
               '-af', f'afade=t=out:st={video_duration-2}:d=2',
               '-shortest', output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _encode_final(self, input_path: str, output_path: str):
        """Final encoding with quality settings"""
        # Quality presets
        quality_settings = [
            {'crf': '28', 'preset': 'veryfast'},  # Low
            {'crf': '23', 'preset': 'medium'},    # Medium
            {'crf': '18', 'preset': 'slow'},      # High
            {'crf': '15', 'preset': 'veryslow'}   # Maximum
        ]
        
        settings = quality_settings[self.quality]
        
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # Hardware-specific encoding
        if self.hw_accel == 'cuda':
            cmd.extend(['-c:v', 'h264_nvenc', '-preset', 'slow', 
                       '-b:v', '5M', '-maxrate', '8M'])
        elif self.hw_accel == 'qsv':
            cmd.extend(['-c:v', 'h264_qsv', '-preset', settings['preset'],
                       '-global_quality', settings['crf']])
        elif self.hw_accel == 'videotoolbox':
            cmd.extend(['-c:v', 'h264_videotoolbox', '-b:v', '5M'])
        else:
            cmd.extend(['-c:v', 'libx264', '-crf', settings['crf'],
                       '-preset', settings['preset'], '-threads', str(self.cpu_threads)])
        
        # Audio (copy if exists)
        cmd.extend(['-c:a', 'copy', '-movflags', '+faststart', output_path])
        
        subprocess.run(cmd, check=True, capture_output=True)
