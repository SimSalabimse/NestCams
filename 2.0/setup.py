from setuptools import setup, find_packages

setup(
    name="NestCams",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "opencv-contrib-python>=4.5.0",
        "ffmpeg-python>=0.2.0",
        "PyQt5>=5.15.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "requests>=2.25.0",
        "GPUtil>=1.4.0",
        "google-api-python-client>=2.0.0",
        "google-auth-oauthlib>=0.4.0",
    ],
    entry_points={
        "console_scripts": [
            "nestcams=main:main",
        ],
    },
    author="Your Name",
    description="Video processor for motion detection and time-lapse creation",
    python_requires=">=3.8",
)
