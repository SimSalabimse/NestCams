"""
Bird Detection Module
Local AI-based bird species detection using pre-trained models
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
import os

logger = logging.getLogger(__name__)


class BirdDetector:
    """Local bird detection and classification"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_loaded = False
        self.detector = None
        self.classifier = None
        
        # Try to load models
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained detection models"""
        try:
            # Try to use YOLO for bird detection
            model_path = self.config.get('bird_model_path', 'models/yolov5s.pt')
            
            if not os.path.exists(model_path):
                logger.warning("Bird detection model not found. Install with:")
                logger.warning("  pip install torch torchvision")
                logger.warning("  Download YOLOv5 model to models/")
                return
            
            # Try to import torch/yolo
            try:
                import torch
                from models.yolo import YOLOv5
                
                self.detector = YOLOv5(model_path)
                self.model_loaded = True
                logger.info("Bird detection model loaded successfully")
                
            except ImportError:
                logger.warning("PyTorch not installed. AI detection disabled.")
                logger.warning("Install with: pip install torch torchvision ultralytics")
                
        except Exception as e:
            logger.warning(f"Could not load bird detection model: {e}")
    
    def detect_birds(self, frame: np.ndarray, confidence: float = 0.5) -> List[Dict]:
        """
        Detect birds in frame
        
        Args:
            frame: OpenCV frame (BGR)
            confidence: Minimum confidence threshold
        
        Returns:
            List of detections: [{'bbox': (x1,y1,x2,y2), 'confidence': float, 'class': str}, ...]
        """
        if not self.model_loaded:
            return []
        
        try:
            # Run detection
            results = self.detector(frame)
            
            birds = []
            for det in results:
                # Filter for bird class (class 14 in COCO is 'bird')
                if det['class'] == 14 and det['confidence'] > confidence:
                    birds.append({
                        'bbox': det['bbox'],
                        'confidence': det['confidence'],
                        'class': 'bird',
                        'species': 'unknown'  # Would need species classifier
                    })
            
            return birds
            
        except Exception as e:
            logger.error(f"Error during bird detection: {e}")
            return []
    
    def is_bird_present(self, frame: np.ndarray, confidence: float = 0.5) -> bool:
        """
        Quick check if bird is present in frame
        
        Args:
            frame: OpenCV frame
            confidence: Minimum confidence
        
        Returns:
            True if bird detected
        """
        detections = self.detect_birds(frame, confidence)
        return len(detections) > 0
    
    def annotate_frame(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on frame
        
        Args:
            frame: Input frame
            detections: List of detections from detect_birds()
        
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            confidence = det['confidence']
            species = det.get('species', 'bird')
            
            # Draw box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{species}: {confidence:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return annotated


class SimpleBirdDetector:
    """
    Simple bird detector using color/motion without AI
    Fallback when AI models not available
    """
    
    def __init__(self, config: dict):
        self.config = config
        logger.info("Using simple (non-AI) bird detection")
    
    def detect_birds(self, frame: np.ndarray) -> List[Dict]:
        """
        Simple detection based on color and size
        """
        # This is a very basic approach
        # Real implementation would use ML models
        
        # Convert to HSV for color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Common bird colors (brown, gray, white)
        # This is very approximate
        lower_brown = np.array([10, 50, 50])
        upper_brown = np.array([30, 255, 200])
        
        mask = cv2.inRange(hsv, lower_brown, upper_brown)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        birds = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Approximate bird size (adjust based on your camera)
            if 500 < area < 50000:
                x, y, w, h = cv2.boundingRect(contour)
                birds.append({
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.7,  # Arbitrary for simple detector
                    'class': 'bird',
                    'species': 'unknown'
                })
        
        return birds
    
    def is_bird_present(self, frame: np.ndarray) -> bool:
        """Check if bird-like object present"""
        detections = self.detect_birds(frame)
        return len(detections) > 0


def get_bird_detector(config: dict):
    """
    Factory function to get appropriate bird detector
    
    Returns:
        BirdDetector if AI available, SimpleBirdDetector otherwise
    """
    try:
        detector = BirdDetector(config)
        if detector.model_loaded:
            return detector
    except:
        pass
    
    # Fallback to simple detector
    logger.info("Using simple bird detector (no AI)")
    return SimpleBirdDetector(config)
