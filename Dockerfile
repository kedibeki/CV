FROM python:3.9
RUN apt-get update
RUN apt-get install -y cmake
RUN pip install dlib
