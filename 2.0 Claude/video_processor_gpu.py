"""
GPU-Optimized Video Processor Module
Forces NVENC encoding and optimized FFmpeg settings for Windows
"""

import cv2
import subprocess
import tempfile
import os
import logging
import time
from typing import List, Tuple, Callable, Optional, Dict
from pathlib import Path
import platform
import numpy as np

logger = logging.getLogger(__name__)


class OptimizedVideoProcessor:
    def __init__(self, config: dict):
        self.config = config
        self.target_lengths = config.get('target_lengths', [60])
        self.smoothing = config.get('smoothing', 5)
        self.use_gpu = config.get('use_gpu', True)
        self.cpu_threads = config.get('cpu_threads', os.cpu_count() - 1)
        self.quality = config.get('quality', 2)
        self.add_music = config.get('add_music', False)
        self.music_paths = config.get('music_paths', {})
        self.motion_blur = config.get('motion_blur', True)
        self.smooth_transitions = config.get('smooth_transitions', True)
        self.color_correction = config.get('color_correction', True)
        
        # Cancellation flag
        self.cancelled = False
        
        # Detect hardware capabilities
        self.hw_accel = self._detect_hardware_acceleration()
        
        # Force NVENC if available (Windows optimization)
        if self.hw_accel == 'cuda':
            logger.info("✅ NVENC encoder available - will use for 3-10x faster encoding")
        
        logger.info(f"Video processor initialized: hw_accel={self.hw_accel}")
    
    def cancel(self):
        """Cancel ongoing processing"""
        self.cancelled = True
        logger.info("Processing cancelled by user")
    
    def _detect_hardware_acceleration(self) -> str:
        """Detect available hardware acceleration"""
        # Check for CUDA/NVENC first (highest priority on Windows)
        try:
            # Check NVIDIA GPU
            result = subprocess.run(['nvidia-smi'], capture_output=True, timeout=2)
            if result.returncode == 0:
                # Verify NVENC encoder is available
                result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                      capture_output=True, text=True, timeout=2)
                if 'h264_nvenc' in result.stdout:
                    logger.info("🚀 NVIDIA GPU with NVENC detected")
                    return 'cuda'
        except:
            pass
        
        system = platform.system()
        
        # Check for Intel Quick Sync
        if system in ['Windows', 'Linux']:
            try:
                result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                      capture_output=True, text=True, timeout=2)
                if 'h264_qsv' in result.stdout:
                    logger.info("Intel Quick Sync detected")
                    return 'qsv'
            except:
                pass
        
        # Check for Apple VideoToolbox
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
    
    def create_timelapse_batch(self, input_path: str, motion_segments: List[Tuple[float, float]], 
                               output_base_path: str, progress_callback: Optional[Callable] = None,
                               time_estimate_callback: Optional[Callable] = None) -> Dict[int, str]:
        """Create multiple time-lapses at different lengths"""
        results = {}
        start_time = time.time()
        
        for i, target_length in enumerate(self.target_lengths):
            if self.cancelled:
                break
                
            # Calculate output path
            base_name = Path(output_base_path).stem
            base_dir = Path(output_base_path).parent
            
            if target_length == 60:
                output_path = base_dir / f"{base_name}_60s_vertical.mp4"
            elif target_length == 600:
                output_path = base_dir / f"{base_name}_10min.mp4"
            elif target_length == 3600:
                output_path = base_dir / f"{base_name}_1hour.mp4"
            else:
                output_path = base_dir / f"{base_name}_{target_length}s.mp4"
            
            logger.info(f"Creating {target_length}s time-lapse ({i+1}/{len(self.target_lengths)})")
            
            # Update progress base
            batch_progress = i / len(self.target_lengths)
            
            def batch_progress_callback(p):
                if progress_callback:
                    total_progress = (batch_progress + (p / 100) / len(self.target_lengths)) * 100
                    progress_callback(total_progress)
            
            # Create time-lapse
            success = self.create_timelapse(
                input_path, motion_segments, str(output_path), target_length,
                batch_progress_callback, time_estimate_callback
            )
            
            if success:
                results[target_length] = str(output_path)
                
                # Update time estimate
                if time_estimate_callback and i < len(self.target_lengths) - 1:
                    elapsed = time.time() - start_time
                    avg_time_per_video = elapsed / (i + 1)
                    remaining = (len(self.target_lengths) - i - 1) * avg_time_per_video
                    time_estimate_callback(remaining)
        
        return results
    
    def create_timelapse(self, input_path: str, motion_segments: List[Tuple[float, float]], 
                        output_path: str, target_length: Optional[int] = None,
                        progress_callback: Optional[Callable] = None,
                        time_estimate_callback: Optional[Callable] = None) -> bool:
        """Create optimized time-lapse with NVENC encoding"""
        try:
            if target_length is None:
                target_length = self.target_lengths[0] if isinstance(self.target_lengths, list) else self.target_lengths
            
            start_time = time.time()
            
            # Calculate total motion duration
            total_motion_time = sum(end - start for start, end in motion_segments)
            
            if total_motion_time == 0:
                logger.error("No motion segments to process")
                return False
            
            # Calculate required speedup - ALWAYS respect target as MAX length
            speedup_factor = max(1.0, total_motion_time / target_length)
            
            logger.info(f"Speedup factor: {speedup_factor:.2f}x (target={target_length}s MAX, motion={total_motion_time:.1f}s)")
            
            # Extract motion segments with optimized settings
            temp_dir = tempfile.mkdtemp()
            segment_files = []
            
            cap = cv2.VideoCapture(input_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            for i, (start, end) in enumerate(motion_segments):
                if self.cancelled:
                    self._cleanup(temp_dir)
                    return False
                    
                if progress_callback:
                    progress = (i / len(motion_segments)) * 50
                    progress_callback(progress)
                
                # Time estimate
                if time_estimate_callback and i > 0:
                    elapsed = time.time() - start_time
                    avg_time = elapsed / i
                    remaining = (len(motion_segments) - i) * avg_time
                    time_estimate_callback(remaining)
                
                segment_path = os.path.join(temp_dir, f"segment_{i:04d}.mp4")
                self._extract_and_process_segment_optimized(
                    input_path, start, end, segment_path, speedup_factor, fps
                )
                segment_files.append(segment_path)
            
            if self.cancelled:
                self._cleanup(temp_dir)
                return False
            
            # Concatenate segments
            if progress_callback:
                progress_callback(60)
            
            concat_path = os.path.join(temp_dir, "concat.mp4")
            self._concatenate_segments(segment_files, concat_path)
            
            if self.cancelled:
                self._cleanup(temp_dir)
                return False
            
            # Add music if requested
            final_input = concat_path
            music_path = None
            if self.add_music:
                if isinstance(self.music_paths, dict):
                    music_path = self.music_paths.get(target_length)
                elif isinstance(self.config.get('music_path'), str):
                    music_path = self.config.get('music_path')
                
                if music_path and os.path.exists(music_path):
                    if progress_callback:
                        progress_callback(80)
                    with_music_path = os.path.join(temp_dir, "with_music.mp4")
                    self._add_music_smooth(concat_path, music_path, with_music_path)
                    final_input = with_music_path
            
            if self.cancelled:
                self._cleanup(temp_dir)
                return False
            
            # Apply rotation for 60s videos (vertical format)
            if target_length == 60:
                if progress_callback:
                    progress_callback(90)
                rotated_path = os.path.join(temp_dir, "rotated.mp4")
                self._rotate_video(final_input, rotated_path, 90)
                final_input = rotated_path
            
            # Final encoding with NVENC optimization
            if progress_callback:
                progress_callback(95)
            
            self._encode_final_optimized(final_input, output_path)
            
            # Cleanup
            self._cleanup(temp_dir)
            
            logger.info(f"Time-lapse created successfully: {output_path}")
            return True
            
        except Exception as e:
            logger.exception("Error creating time-lapse")
            return False
    
    def _extract_and_process_segment_optimized(self, input_path: str, start: float, end: float, 
                                               output_path: str, speedup: float, fps: float):
        """Extract and process segment with optimized FFmpeg settings"""
        duration = end - start
        
        # Build FFmpeg command with GPU optimization
        cmd = ['ffmpeg', '-y', '-ss', str(start), '-i', input_path, 
               '-t', str(duration)]
        
        # Build complex filter
        filters = []
        
        # Speed up video
        if speedup > 1.0:
            pts_factor = 1.0 / speedup
            filters.append(f'setpts={pts_factor}*PTS')
        
        # Motion blur for smoothness (optimized)
        if self.motion_blur and speedup > 2.0:
            blur_amount = min(20, int(speedup / 2))
            filters.append(f'tmix=frames={blur_amount}:weights="1"')
        
        # Combine filters
        if filters:
            filter_str = ','.join(filters)
            cmd.extend(['-filter:v', filter_str])
        
        # Use NVENC for segment processing (fast preset)
        if self.hw_accel == 'cuda':
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'fast',
                '-cq', '18',  # Quality
                '-pix_fmt', 'yuv420p'
            ])
        elif self.hw_accel == 'qsv':
            cmd.extend(['-c:v', 'h264_qsv', '-preset', 'fast'])
        elif self.hw_accel == 'videotoolbox':
            cmd.extend(['-c:v', 'h264_videotoolbox', '-b:v', '5M'])
        else:
            cmd.extend(['-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '18'])
        
        # Remove original audio
        cmd.extend(['-an', output_path])
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _concatenate_segments(self, segment_files: List[str], output_path: str):
        """Concatenate segments"""
        concat_file = output_path.replace('.mp4', '_list.txt')
        with open(concat_file, 'w') as f:
            for seg in segment_files:
                f.write(f"file '{seg}'\n")
        
        cmd = ['ffmpeg', '-y', '-f', 'concat', '-safe', '0', 
               '-i', concat_file, '-c', 'copy', output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(concat_file)
    
    def _add_music_smooth(self, video_path: str, music_path: str, output_path: str):
        """Add background music with smooth looping"""
        # Get video duration
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    video_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_duration = float(result.stdout.strip())
        
        # Get music duration
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 
                    'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                    music_path]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        music_duration = float(result.stdout.strip())
        
        # Calculate loops needed
        loops_needed = int(np.ceil(video_duration / music_duration))
        
        # Create audio filter for smooth looping
        audio_filter = f'aloop=loop={loops_needed}:size={int(music_duration * 48000)},atrim=0:{video_duration}'
        audio_filter += f',afade=t=in:st=0:d=0.5,afade=t=out:st={video_duration-2}:d=2'
        
        # Mix video (no audio) with looped music
        cmd = ['ffmpeg', '-y', '-i', video_path, '-i', music_path,
               '-filter_complex', audio_filter,
               '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k',
               '-shortest', output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _rotate_video(self, input_path: str, output_path: str, angle: int):
        """Rotate video for vertical format"""
        transpose_code = 1 if angle == 90 else 2
        
        cmd = ['ffmpeg', '-y', '-i', input_path,
               '-vf', f'transpose={transpose_code}',
               '-c:a', 'copy', output_path]
        
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _encode_final_optimized(self, input_path: str, output_path: str):
        """
        Final encoding with OPTIMIZED NVENC settings for Windows
        Based on hardware acceleration advice
        """
        cmd = ['ffmpeg', '-y', '-i', input_path]
        
        # FORCE NVENC if available (Windows optimization)
        if self.hw_accel == 'cuda':
            # NVENC optimized settings
            quality_map = ['18', '23', '28']  # CQ values for NVENC
            cq_value = quality_map[min(self.quality, 2)]
            
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'slow',  # NVENC preset (p1-p7, slow is p4)
                '-cq', cq_value,  # Constant quality mode
                '-b:v', '5M',  # Target bitrate
                '-maxrate', '8M',  # Max bitrate
                '-bufsize', '10M',  # Buffer size
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ])
            
        elif self.hw_accel == 'qsv':
            # Intel Quick Sync
            cmd.extend([
                '-c:v', 'h264_qsv',
                '-preset', 'medium',
                '-global_quality', '18',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ])
            
        elif self.hw_accel == 'videotoolbox':
            # Apple VideoToolbox
            cmd.extend([
                '-c:v', 'h264_videotoolbox',
                '-b:v', '5M',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ])
            
        else:
            # CPU fallback with optimized settings
            quality_settings = ['18', '23', '28', '32']
            crf = quality_settings[self.quality]
            
            cmd.extend([
                '-c:v', 'libx264',
                '-crf', crf,
                '-preset', 'slow',  # Better quality
                '-threads', str(self.cpu_threads),
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ])
        
        # Audio (copy if exists)
        cmd.extend(['-c:a', 'copy', output_path])
        
        logger.info(f"Encoding with: {cmd[7]}")  # Log encoder used
        subprocess.run(cmd, check=True, capture_output=True)
    
    def _cleanup(self, temp_dir: str):
        """Clean up temporary files"""
        try:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
        except:
            pass


# Factory function for backwards compatibility
def VideoProcessor(config: dict):
    """Factory function"""
    return OptimizedVideoProcessor(config)
