FROM python:3.9
RUN apt-get update
RUN apt-get install -y ffmpeg libsm6 libxext6 cmake libgl1-mesa-glx
RUN apt-get install -y cmake
RUN pip install dlib
