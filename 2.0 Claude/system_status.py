"""
System Status Checker and Resource Monitor
Checks GPU availability, FFmpeg, and monitors system resources
"""

import subprocess
import platform
import os
import logging
import psutil
from typing import Dict, List, Tuple, Optional
import cv2

logger = logging.getLogger(__name__)


class SystemStatus:
    """Comprehensive system status checker"""
    
    def __init__(self):
        self.status = {}
        self.refresh()
    
    def refresh(self):
        """Refresh all status checks"""
        self.status = {
            'python': self._check_python(),
            'opencv': self._check_opencv(),
            'cuda': self._check_cuda(),
            'ffmpeg': self._check_ffmpeg(),
            'nvenc': self._check_nvenc(),
            'gpu': self._check_gpu_hardware(),
            'memory': self._check_memory(),
            'disk': self._check_disk(),
            'system': self._get_system_info()
        }
        return self.status
    
    def _check_python(self) -> Dict:
        """Check Python version"""
        import sys
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return {
            'installed': True,
            'version': version,
            'ok': sys.version_info >= (3, 8),
            'message': f"Python {version}" if sys.version_info >= (3, 8) else f"Python {version} (3.8+ required)"
        }
    
    def _check_opencv(self) -> Dict:
        """Check OpenCV installation"""
        try:
            version = cv2.__version__
            return {
                'installed': True,
                'version': version,
                'ok': True,
                'message': f"OpenCV {version}"
            }
        except:
            return {
                'installed': False,
                'version': None,
                'ok': False,
                'message': "OpenCV not installed"
            }
    
    def _check_cuda(self) -> Dict:
        """Check CUDA support in OpenCV"""
        try:
            if hasattr(cv2, 'cuda'):
                count = cv2.cuda.getCudaEnabledDeviceCount()
                if count > 0:
                    return {
                        'available': True,
                        'devices': count,
                        'ok': True,
                        'message': f"CUDA available ({count} device{'s' if count > 1 else ''})"
                    }
                else:
                    return {
                        'available': False,
                        'devices': 0,
                        'ok': False,
                        'message': "CUDA OpenCV installed but no GPU detected",
                        'fix': "Install NVIDIA drivers from nvidia.com/drivers"
                    }
            else:
                return {
                    'available': False,
                    'devices': 0,
                    'ok': False,
                    'message': "CPU-only OpenCV (no CUDA)",
                    'fix': "Install opencv-python-cuda for GPU acceleration"
                }
        except Exception as e:
            return {
                'available': False,
                'devices': 0,
                'ok': False,
                'message': f"CUDA check failed: {str(e)}",
                'fix': "Reinstall OpenCV"
            }
    
    def _check_ffmpeg(self) -> Dict:
        """Check FFmpeg installation"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                # Extract version
                version_line = result.stdout.split('\n')[0]
                version = version_line.split('version')[1].split()[0] if 'version' in version_line else 'unknown'
                return {
                    'installed': True,
                    'version': version,
                    'ok': True,
                    'message': f"FFmpeg {version}"
                }
            else:
                return {
                    'installed': False,
                    'version': None,
                    'ok': False,
                    'message': "FFmpeg not found",
                    'fix': "Download from ffmpeg.org and add to PATH"
                }
        except FileNotFoundError:
            return {
                'installed': False,
                'version': None,
                'ok': False,
                'message': "FFmpeg not found in PATH",
                'fix': "Install FFmpeg from ffmpeg.org"
            }
        except Exception as e:
            return {
                'installed': False,
                'version': None,
                'ok': False,
                'message': f"FFmpeg check failed: {str(e)}"
            }
    
    def _check_nvenc(self) -> Dict:
        """Check NVENC encoder availability"""
        try:
            result = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                has_nvenc = 'h264_nvenc' in result.stdout
                if has_nvenc:
                    return {
                        'available': True,
                        'ok': True,
                        'message': "NVENC encoder available"
                    }
                else:
                    return {
                        'available': False,
                        'ok': False,
                        'message': "NVENC not available",
                        'fix': "Update NVIDIA drivers or use full FFmpeg build"
                    }
            else:
                return {
                    'available': False,
                    'ok': False,
                    'message': "Cannot check NVENC (FFmpeg error)"
                }
        except:
            return {
                'available': False,
                'ok': False,
                'message': "Cannot check NVENC (FFmpeg not found)"
            }
    
    def _check_gpu_hardware(self) -> Dict:
        """Check NVIDIA GPU hardware"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', 
                                   '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                name = parts[0].strip() if len(parts) > 0 else "Unknown"
                memory = parts[1].strip() if len(parts) > 1 else "Unknown"
                driver = parts[2].strip() if len(parts) > 2 else "Unknown"
                
                return {
                    'detected': True,
                    'name': name,
                    'memory': memory,
                    'driver': driver,
                    'ok': True,
                    'message': f"{name} ({memory})"
                }
            else:
                return {
                    'detected': False,
                    'ok': False,
                    'message': "No NVIDIA GPU detected",
                    'fix': "AMD/Intel GPUs not supported for GPU acceleration"
                }
        except FileNotFoundError:
            return {
                'detected': False,
                'ok': False,
                'message': "nvidia-smi not found",
                'fix': "Install NVIDIA drivers from nvidia.com/drivers"
            }
        except Exception as e:
            return {
                'detected': False,
                'ok': False,
                'message': f"GPU check failed: {str(e)}"
            }
    
    def _check_memory(self) -> Dict:
        """Check system memory"""
        try:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            available_gb = mem.available / (1024**3)
            used_percent = mem.percent
            
            return {
                'total_gb': total_gb,
                'available_gb': available_gb,
                'used_percent': used_percent,
                'ok': available_gb > 2.0,  # At least 2GB free
                'message': f"{total_gb:.1f}GB total, {available_gb:.1f}GB available ({used_percent:.0f}% used)"
            }
        except:
            return {
                'total_gb': 0,
                'available_gb': 0,
                'used_percent': 0,
                'ok': False,
                'message': "Cannot check memory"
            }
    
    def _check_disk(self) -> Dict:
        """Check disk space"""
        try:
            disk = psutil.disk_usage('.')
            free_gb = disk.free / (1024**3)
            total_gb = disk.total / (1024**3)
            used_percent = disk.percent
            
            return {
                'total_gb': total_gb,
                'free_gb': free_gb,
                'used_percent': used_percent,
                'ok': free_gb > 5.0,  # At least 5GB free
                'message': f"{free_gb:.1f}GB free of {total_gb:.1f}GB ({used_percent:.0f}% used)"
            }
        except:
            return {
                'total_gb': 0,
                'free_gb': 0,
                'used_percent': 0,
                'ok': False,
                'message': "Cannot check disk"
            }
    
    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'cpu_count': os.cpu_count(),
            'message': f"{platform.system()} {platform.release()} ({platform.machine()})"
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        lines = []
        
        # GPU status
        if self.status['cuda']['ok']:
            lines.append(f"✅ GPU Acceleration: {self.status['gpu']['name']}")
        else:
            lines.append(f"❌ GPU Acceleration: Not available")
            if 'fix' in self.status['cuda']:
                lines.append(f"   Fix: {self.status['cuda']['fix']}")
        
        # FFmpeg status
        if self.status['ffmpeg']['ok']:
            nvenc_status = "✓" if self.status['nvenc']['ok'] else "✗"
            lines.append(f"✅ FFmpeg: {self.status['ffmpeg']['version']} (NVENC: {nvenc_status})")
        else:
            lines.append(f"❌ FFmpeg: Not found")
            if 'fix' in self.status['ffmpeg']:
                lines.append(f"   Fix: {self.status['ffmpeg']['fix']}")
        
        # Memory status
        mem_status = "✅" if self.status['memory']['ok'] else "⚠️"
        lines.append(f"{mem_status} Memory: {self.status['memory']['message']}")
        
        return "\n".join(lines)
    
    def get_issues(self) -> List[Tuple[str, str]]:
        """Get list of issues with fixes"""
        issues = []
        
        if not self.status['cuda']['ok'] and 'fix' in self.status['cuda']:
            issues.append(("CUDA not available", self.status['cuda']['fix']))
        
        if not self.status['ffmpeg']['ok'] and 'fix' in self.status['ffmpeg']:
            issues.append(("FFmpeg not found", self.status['ffmpeg']['fix']))
        
        if not self.status['nvenc']['ok'] and 'fix' in self.status['nvenc']:
            issues.append(("NVENC not available", self.status['nvenc']['fix']))
        
        if not self.status['memory']['ok']:
            issues.append(("Low memory", "Close other applications or add more RAM"))
        
        if not self.status['disk']['ok']:
            issues.append(("Low disk space", "Free up disk space (need 5GB+ for temp files)"))
        
        return issues


class ResourceMonitor:
    """Real-time resource monitoring during processing"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
    
    def start(self):
        """Start monitoring"""
        self.start_time = psutil.time.time()
        self.start_memory = self.process.memory_info().rss
    
    def get_stats(self) -> Dict:
        """Get current resource usage"""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            
            # Memory usage
            mem_info = self.process.memory_info()
            mem_mb = mem_info.rss / (1024**2)
            mem_percent = self.process.memory_percent()
            
            # System memory
            sys_mem = psutil.virtual_memory()
            sys_mem_available = sys_mem.available / (1024**3)
            
            # GPU usage (if nvidia-smi available)
            gpu_usage = self._get_gpu_usage()
            
            # Elapsed time
            elapsed = psutil.time.time() - self.start_time if self.start_time else 0
            
            return {
                'cpu_percent': cpu_percent,
                'memory_mb': mem_mb,
                'memory_percent': mem_percent,
                'system_memory_available_gb': sys_mem_available,
                'gpu_usage': gpu_usage,
                'elapsed_seconds': elapsed,
                'ok': mem_percent < 80 and sys_mem_available > 1.0  # Health check
            }
        except:
            return {
                'cpu_percent': 0,
                'memory_mb': 0,
                'memory_percent': 0,
                'system_memory_available_gb': 0,
                'gpu_usage': {},
                'elapsed_seconds': 0,
                'ok': True
            }
    
    def _get_gpu_usage(self) -> Dict:
        """Get GPU usage stats"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu', 
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                return {
                    'utilization': int(parts[0].strip()),
                    'memory_used_mb': int(parts[1].strip()),
                    'memory_total_mb': int(parts[2].strip()),
                    'temperature': int(parts[3].strip())
                }
        except:
            pass
        return {}
    
    def format_stats(self, stats: Dict) -> str:
        """Format stats as readable string"""
        lines = []
        lines.append(f"CPU: {stats['cpu_percent']:.1f}%")
        lines.append(f"Memory: {stats['memory_mb']:.0f}MB ({stats['memory_percent']:.1f}%)")
        
        if stats['gpu_usage']:
            gpu = stats['gpu_usage']
            lines.append(f"GPU: {gpu['utilization']}% | {gpu['memory_used_mb']}MB / {gpu['memory_total_mb']}MB | {gpu['temperature']}°C")
        
        return " | ".join(lines)
