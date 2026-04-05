# Contributing to Bird Motion Video Processor

Thank you for your interest in contributing! This guide will help you get started.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Feature Requests](#feature-requests)

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to make better software for bird enthusiasts!

## Getting Started

### Prerequisites
- Python 3.8+
- FFmpeg
- Git
- Code editor (VS Code, PyCharm, etc.)

### Fork and Clone

```bash
# Fork the repository on GitHub first, then:
git clone https://github.com/YOUR_USERNAME/bird-motion-processor.git
cd bird-motion-processor

# Add upstream remote
git remote add upstream https://github.com/ORIGINAL_OWNER/bird-motion-processor.git
```

## Development Setup

### 1. Create Virtual Environment
```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt

# Optional: Install development dependencies
pip install pytest black pylint mypy
```

### 3. Run Tests
```bash
python test_installation.py
```

### 4. Start Application
```bash
python main.py
```

## Project Structure

```
bird-motion-processor/
├── main.py                  # GUI application entry point
├── motion_detector.py       # Motion detection algorithms
├── video_processor.py       # Video processing pipeline
├── config_manager.py        # Settings management
├── youtube_uploader.py      # YouTube API integration
├── update_checker.py        # Update checking
└── tests/                   # Test files (add as needed)
```

### Module Responsibilities

**main.py**
- User interface (PyQt5)
- Event handling
- Progress display
- Settings UI

**motion_detector.py**
- Frame-by-frame motion analysis
- Segment identification
- Noise filtering

**video_processor.py**
- Video extraction
- Speed adjustment
- Concatenation
- Encoding

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed design.

## Coding Standards

### Python Style Guide

We follow PEP 8 with some modifications:

```python
# Good: Clear, descriptive names
def detect_motion_in_video(video_path, sensitivity=5):
    """
    Detect motion in a video file.
    
    Args:
        video_path: Path to input video
        sensitivity: Motion sensitivity (1-10)
    
    Returns:
        List of (start_time, end_time) tuples
    """
    pass

# Bad: Unclear, abbreviated names
def dm(vp, s=5):
    pass
```

### Key Guidelines

1. **Docstrings**: All public functions need docstrings
2. **Type Hints**: Use where it improves clarity
3. **Comments**: Explain why, not what
4. **Naming**:
   - Classes: `PascalCase`
   - Functions/methods: `snake_case`
   - Constants: `UPPER_SNAKE_CASE`
5. **Line Length**: Max 100 characters (not strict 80)

### Code Formatting

```bash
# Format code
black main.py motion_detector.py

# Check style
pylint main.py
```

### Example Code Style

```python
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class MotionDetector:
    """Detects motion in video files."""
    
    def __init__(self, sensitivity: int = 5):
        """
        Initialize motion detector.
        
        Args:
            sensitivity: Detection sensitivity (1-10)
        """
        self.sensitivity = sensitivity
        logger.info(f"Initialized with sensitivity={sensitivity}")
    
    def detect_motion(self, 
                     video_path: str,
                     progress_callback: Optional[Callable] = None
                     ) -> List[Tuple[float, float]]:
        """
        Detect motion segments in video.
        
        Args:
            video_path: Path to video file
            progress_callback: Optional progress update function
        
        Returns:
            List of (start_time, end_time) tuples
        
        Raises:
            FileNotFoundError: If video doesn't exist
            ValueError: If video can't be opened
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video not found: {video_path}")
        
        # Implementation here
        segments = []
        return segments
```

## Testing

### Writing Tests

Create test files in `tests/` directory:

```python
# tests/test_motion_detector.py
import unittest
from motion_detector import MotionDetector

class TestMotionDetector(unittest.TestCase):
    def setUp(self):
        self.detector = MotionDetector(sensitivity=5)
    
    def test_initialization(self):
        self.assertEqual(self.detector.sensitivity, 5)
    
    def test_invalid_sensitivity(self):
        with self.assertRaises(ValueError):
            MotionDetector(sensitivity=15)  # Out of range

if __name__ == '__main__':
    unittest.main()
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_motion_detector.py

# Run with coverage
python -m pytest --cov=. tests/
```

## Common Contribution Patterns

### Adding a New Feature

1. **Create Feature Branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Implement Feature**
- Add code in appropriate module
- Add docstrings and comments
- Update documentation

3. **Test**
- Write unit tests
- Test manually with real videos
- Run `test_installation.py`

4. **Document**
- Update README.md if user-facing
- Update CHANGELOG.md
- Add code comments

5. **Commit**
```bash
git add .
git commit -m "Add: Brief description of feature

Detailed explanation of what was added and why.
Resolves #issue_number"
```

### Bug Fix Workflow

1. **Create Issue** (if not exists)
   - Describe the bug
   - Steps to reproduce
   - Expected vs actual behavior

2. **Create Branch**
```bash
git checkout -b fix/bug-description
```

3. **Fix Bug**
   - Write test that reproduces bug
   - Fix the bug
   - Verify test passes

4. **Commit**
```bash
git commit -m "Fix: Brief description

Explanation of fix.
Fixes #issue_number"
```

## Submitting Changes

### Pull Request Process

1. **Update from upstream**
```bash
git fetch upstream
git rebase upstream/main
```

2. **Push to your fork**
```bash
git push origin feature/your-feature
```

3. **Create Pull Request**
   - Go to GitHub
   - Click "New Pull Request"
   - Fill in template (if exists)
   - Link related issues

4. **PR Description Should Include:**
   - What changed and why
   - Screenshots (if UI change)
   - Testing performed
   - Related issue numbers

### Code Review

- Respond to feedback constructively
- Make requested changes in new commits
- Push updates to same branch
- Request re-review when ready

## Feature Requests

### Proposing New Features

1. **Check Existing Issues**: Avoid duplicates
2. **Create Issue**: Use "Feature Request" template
3. **Describe**:
   - Problem it solves
   - Proposed solution
   - Alternative approaches
   - Example use cases

4. **Discuss**: Wait for maintainer feedback before implementing

## Development Tips

### Debugging

```python
# Add detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# In your code
logger.debug(f"Processing frame {frame_num}")
logger.info(f"Detected motion at {timestamp}s")
logger.warning(f"Low confidence: {confidence}")
logger.error(f"Failed to process: {error}")
```

### Testing with Videos

Create a `test_videos/` directory (git-ignored) with:
- Short test video (1-2 min)
- Medium video (10-30 min)
- Various resolutions
- Different lighting conditions

### Performance Profiling

```python
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(20)  # Top 20 slowest functions
```

## Areas Looking for Contributions

### High Priority
- [ ] Unit test coverage
- [ ] Performance optimization for 4K videos
- [ ] Better GPU memory management
- [ ] macOS Apple Silicon optimization

### Medium Priority
- [ ] Additional export formats (GIF, WebM)
- [ ] Batch processing mode
- [ ] Video preview before processing
- [ ] Progress estimation improvements

### Low Priority / Fun Projects
- [ ] AI bird species detection
- [ ] Motion heatmap visualization
- [ ] Sound detection (bird calls)
- [ ] Multi-camera sync

## Release Process (Maintainers)

1. Update version in:
   - `main.py`
   - `update_checker.py`
   - `CHANGELOG.md`

2. Create release notes
3. Tag release: `git tag -a v1.x.x -m "Version 1.x.x"`
4. Push: `git push origin v1.x.x`
5. Create GitHub release with binaries (if applicable)

## Getting Help

- **Documentation**: Check README.md and ARCHITECTURE.md
- **Issues**: Search existing issues first
- **Discussions**: For questions and ideas
- **Code**: Read existing implementations for patterns

## Recognition

Contributors will be:
- Listed in CHANGELOG.md
- Mentioned in release notes
- Added to CONTRIBUTORS.md (if we create one)

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing!** 🙏

Every contribution, whether it's a bug report, feature request, documentation improvement, or code change, helps make this tool better for bird watchers everywhere! 🐦
