# -*- coding: utf-8 -*-

"""## Emotion Streamlit Webapp"""

# %%writefile Emotion.py

# Import necessary libraries

import io
import os
import base64
from io import BytesIO
from github import Github
import tempfile

# scrapping library
import requests
import pathlib
import imghdr

# for regular expression operations
import re

# data processing
import pandas as pd
import numpy as np
import tensorflow as tf
import urllib
from urllib.parse import urlparse

# visualization modules
from PIL import Image
from PIL import UnidentifiedImageError
import plotly.graph_objects as go

# Import Streamlit library
import streamlit as st
from streamlit_webrtc import VideoProcessorBase, VideoTransformerBase, webrtc_streamer

# time library
import time
from datetime import datetime
import pytz
import random
from collections import Counter

# Image and Video processor libraries
import cv2
import torch
from deepface import DeepFace
from retinaface import RetinaFace
from retinaface.commons import postprocess
import face_alignment
import dlib
from imutils import face_utils
import av



# make the wide mode to occupy the entire webapp page
st.set_page_config(layout="wide")

# hide the streamlit menu
hide_menu_style = """
<style>
#MainMenu {visibility: hidden;}
button {
  background-color: red !important;
  color: white !important;
}
button:hover, button:focus {
  background-color: green !important;
  color: white !important;
}
footer {
    visibility: hidden !important;
}
</style>
"""
st.markdown(hide_menu_style, unsafe_allow_html=True)


# Define custom styles
font_family = "Times New Roman"
font_size = "20px"


def time_consuming_function():
    # Perform some time-consuming task here
    for i in range(10):
        # Sleep for 1 second to simulate a time-consuming task
        time.sleep(1)

    # Return the result of the time-consuming task
    return "result"

# spinner text
spinner_text = "Hello, I'm Kedir, please wait this won't take long. Thank you for your patience!"

# Display a spinner with the translated text
with st.spinner(spinner_text):
    # Run a time-consuming function
    result = time_consuming_function()


# Create a single column using st.columns function with 1 as argument and align parameter
col = st.columns(1)[0]

# Get the current date and time in Washington DC timezone using pytz module
tz = pytz.timezone("America/New_York")
now = datetime.now(tz)

# Format the date and time as strings
time_str = now.strftime("%H:%M:%S")
date_str = now.strftime("%a, %b. %d, %Y")  # Changed to abbreviated format
place_str = "Washington, DC"

# Display the clock in the column using st.write with markdown=True parameter and custom CSS styles
clock_html = f"""
    <div id="clock" style="background-color: yellow; color: black; padding: 0px; padding-right: 10px; margin-top: -40px; margin-right: 0px; align: top; width: 160px; float: right;">
        <h2 id="time" style="color: red; margin: 0; margin-top: -10px; text-align: right; font-size: 36px; font-weight: bold;">{time_str}</h2>
        <p id="date" style="margin: 0; margin-top: -10px; text-align: right;">{date_str}</p>
        <p style="margin-top: 0px; text-align: right;">{place_str}</p>
    </div>
    <style>
    .time {{
        color: red !important;
        animation: none !important;
    }}
    </style>
    <script>
        function updateTime() {{
            // Get the current date and time
            var now = new Date();

            // Format the time as a string
            var timeStr = now.toLocaleTimeString("en-US", {{ timeZone: "America/New_York" }});

            // Update the time element
            document.getElementById("time").textContent = timeStr;
        }}

        // Update the clock every second
        setInterval(updateTime, 1000);
    </script>
"""
col.write(clock_html, unsafe_allow_html=True)


# Create a single column using st.columns function with 1 as argument and align parameter
col = st.columns(1)[0]

# Set the URL of the Logo image
image_url = 'https://raw.githubusercontent.com/kedibeki/Emotion-Recognition-and-Tracker/main/Emotion%20Recognition%20and%20Tracker_Logo.jpg'

@st.cache_resource(ttl=3600, max_entries=10)
def get_image(image_url):
    # Try to get the content of the file from the URL
    try:
        response = requests.get(image_url)

        # Get the base64 encoding of the image
        image = base64.b64encode(response.content).decode()

        return image

    # Handle any exceptions that may occur
    except Exception as e:
        # Display the exception message
        st.error(f"Could not load the image. Exception: {e}")

image = get_image(image_url)

# Display the image in the column using st.markdown with markdown=True parameter and custom CSS styles
image_html = f"""
    <img src="data:image/png;base64,{image}" style="width: 150px; float: left; margin-top: -122px; margin-left: 0px;">
"""
col.markdown(image_html, unsafe_allow_html=True)


# Set the URL of the Background image
image_url = 'https://raw.githubusercontent.com/kedibeki/Emotion-Recognition-and-Tracker/main/Emotion%20Recognition%20and%20Tracker_Background.jpg'

@st.cache_resource(ttl=3600, max_entries=10)
def get_background_image(url):
    # Try to get the content of the file from the URL
    try:
        response = requests.get(url)

        # Get the base64 encoding of the image
        image = base64.b64encode(response.content).decode()

        return image

    # Handle any exceptions that may occur
    except Exception as e:
        # Display the exception message
        st.error(f"Could not load the background image. Exception: {e}")

image = get_background_image(image_url)

# Set the background image of the body and sidebar dark with element using CSS
st.markdown(f"""
<style>
.stApp {{
    background-image: url("data:image/png;base64,{image}");
    background-size: cover;
}}
div[data-testid="stSidebar"] {{
    background-color: black;
}}
</style>
""", unsafe_allow_html=True)


# Load a pre-trained model for attribute recognition using DeepFace
@st.cache_resource(ttl=3600, max_entries=10)
def load_model():
    model = DeepFace.build_model("VGG-Face")
    return model

model = load_model()

# Check if CUDA is available, else use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the face aligner
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False)

# Define a function that converts an image to base64
def img_to_base64(img):
    # Convert the image to a base64 encoded image as DeepFace.analyze() expects an image file or a base64 encoded image
    is_success, im_buf_arr = cv2.imencode(".jpg", img)

    # Access the second element of the tuple and convert it to base64
    byte_im = im_buf_arr.tobytes()
    base64_im = base64.b64encode(byte_im)

    # Convert the base64_im variable to a string
    base64_im = base64_im.decode('utf-8')

    return base64_im

# Modify process_frame_recognition function to use RetinaFace for face detection
def process_frame_recognition(frame):
    # Detect faces in the frame using RetinaFace
    faces = RetinaFace.detect_faces(frame)

    # Initialize an empty list to store info for all faces
    all_faces_info = []

    for i, face in enumerate(faces):  # Iterate over the tuple directly
        # Extract each face from the frame
        # Check if face has exactly four items
        if len(face) == 4:
            x, y, w, h = face  # Unpack the tuple directly
            extracted_face = frame[y:y+h, x:x+w]  # Define 'extracted_face' here

        # Analyze facial attributes using DeepFace
        results = DeepFace.analyze(img_path=extracted_face,
                                  actions=['age', 'gender', 'emotion', 'race'],
                                  enforce_detection=False)

        for result in results:
            age = result['age']
            gender = result['gender']
            emotion = result['dominant_emotion']
            race = result['dominant_race']

            # Add the info for this face to the list
            all_faces_info.append({
                'index': i+1,
                'age': age,
                'gender': gender,
                'emotion': emotion,
                'race': race
            })

        # Draw a rectangle around the face and label it with the index and dominant emotion
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"{i+1}: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return frame, all_faces_info

# Define a class that inherits from VideoProcessorBase instead of VideoTransformerBase as it is deprecated
class VideoProcessor(VideoProcessorBase):
    frame_lock: bool = False

    def recv(self):
        # Get the original frame from the video stream
        frame = self.video_stream.recv()

        if not self.frame_lock:
            # Convert the frame to an ndarray format
            img = frame.to_ndarray(format="bgr24")

            # Process the frame using the function defined above
            img_processed, all_faces_info = process_frame_recognition(img)

            # Save the info for all faces as an attribute of the class instance
            self.all_faces_info = all_faces_info

            self.frame_lock = True  # Lock after processing the first frame

        # Return a new frame with the processed image
        return av.VideoFrame.from_ndarray(img_processed, format="bgr24")


st.markdown("<h1 style='text-align: center;'>Emotion Recognizer and Tracker</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>Recognize and track emotions with state-of-art AI/ML</h2>", unsafe_allow_html=True)

# Add a line break
st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<b>Computer vision</b> is a field of artificial intelligence that enables computers and systems to derive meaningful information from digital images, videos and other visual inputs, and take actions or make recommendations based on that information. <b>Emotion recognition</b> and tracking is a subfield of computer vision that focuses on identifying and measuring human emotions from facial expressions, eye movements, voice tones, body gestures and other cues. Some of real world applications of Computer Vision for Emotion Recognition and tracking:

<ul style="margin-left: 5px; list-style-type: none;">
<li>✔ <b>Healthcare</b>: Useful for patient monitoring and mental health analysis.</li>
<li>✔ <b>Entertainment Industry</b>: Can evaluate audience reactions to movies or advertisements.</li>
<li>✔ <b>Retail</b>: Helps improve customer experience by analyzing customer reactions.</li>
<li>✔ <b>Automotive Industry</b>: Can enhance safety by monitoring driver's alertness.</li>
<li>✔ <b>Education</b>: Assists in understanding student engagement during online classes.</li>
</ul>

Welcome to this advanced emotion detection application! Here, you can upload an image, provide an image URL up to 200MB or capture a selfie (for now streamlit doesn't support this feature) for real-time emotion analysis. The app identifies emotions, along with additional attributes like age, gender, and race, and presents the results in an interactive and user-friendly format. You can zoom into the visualizations or download them for further exploration. Enjoy your journey of understanding emotional expressions!

The pie chart represents the distribution of emotions detected in the image. Each slice of the pie corresponds to a unique emotion, and the size of the slice indicates the proportion of faces with that emotion. You can hover over each slice to see more details. The plot also provides interactive tools at the top right corner. You can zoom in on specific parts of the plot, save it as a PNG image, and more.

The table presents detailed information about each face detected in the image. Each row corresponds to a unique emotion, and the columns provide additional attributes such as age, gender, and race. The values in each cell are comma-separated lists that correspond to multiple faces with the same emotion. The table also provides interactive tools at the top right corner. You can zoom in on specific parts of the table, save it as a PNG image, and more.

<b>⚠️Caution</b>: This application is primarily my personal project designed to demonstrate the capabilities of computer vision. It is not intended for real-world application. While it strives to be accurate, it is not 100% infallible and has certain limitations. Please exercise caution and use it at your own risk.
""", unsafe_allow_html=True)


st.sidebar.title("Navigation Menu")
source = st.sidebar.selectbox("Choose a source", ["Upload File", "URL", "Selfie"])
all_faces_info = None

# Add a line break
st.markdown("<br><br>", unsafe_allow_html=True)

# Add code for handling upload file source
if source == "Upload File":
    uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png"])

    if uploaded_file is not None:
        # Read the uploaded file as an image
        img_uploaded = Image.open(uploaded_file)

        # Convert the image to an ndarray format
        img_uploaded = np.array(img_uploaded)

        # Convert the image to base64
        base64_img = img_to_base64(img_uploaded)

        # Process the image using the function defined above
        img_processed, all_faces_info = process_frame_recognition(img_uploaded)

        # Display the processed image with a caption
        st.image(img_processed, caption="Processed Image")

# Add code for handling url source
if source == "URL":
    with st.form(key='url_form'):
        url_input = st.text_input("Enter an image URL")
        submit_button = st.form_submit_button(label='Submit')

        if submit_button and url_input:
            # Get the response from the url and check if it is valid
            response = requests.get(url_input)
            if response.status_code == 200:
                # Read the response content as an image
                img_url = Image.open(io.BytesIO(response.content))

                # Convert the image to an ndarray format
                img_url = np.array(img_url)

                # Convert the image to base64
                base64_img = img_to_base64(img_url)

                # Process the image using the function defined above
                img_processed, all_faces_info = process_frame_recognition(img_url)

                # Display the processed image with a caption
                st.image(img_processed, caption="Processed Image")
            else:
                # Display an error message if the url is invalid
                st.error("Invalid URL")

if source == "Selfie":
    webrtc_ctx = webrtc_streamer(key="selfie", video_processor_factory=VideoProcessor)

    if st.button('Capture'):
        if webrtc_ctx.video_processor is not None:
            if webrtc_ctx.video_processor.frame_lock:
                all_faces_info = webrtc_ctx.video_processor.all_faces_info
                # Convert the processed image to PIL format
                img_processed_pil = Image.fromarray(img_processed)
                # Display the processed image
                st.image(img_processed_pil, caption="Processed Image")


if all_faces_info is not None:
    # Count emotions and draw a pie chart
    emotions_count = Counter([info['emotion'] for info in all_faces_info])

    # Use a fixed color palette based on the emotion labels
    colors = {
        "angry": "red",
        "disgust": "green",
        "fear": "purple",
        "happy": "yellow",
        "sad": "blue",
        "surprise": "orange",
        "neutral": "gray"
    }

    fig = go.Figure(data=[go.Pie(labels=list(emotions_count.keys()),
                                 values=list(emotions_count.values()),
                                 marker_colors=[colors[emotion] for emotion in emotions_count.keys()])])  # Pass the colors to marker_colors based on the labels

    fig.update_layout(
        title_text="Emotion Distribution",
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
        paper_bgcolor="rgba(0,0,0,0)",  # Makes background transparent
        plot_bgcolor="rgba(0,0,0,0)",  # Makes plot background transparent
        legend=dict(
            orientation="h",  # Horizontal legend
            yanchor="bottom",
            y=-0.2,  # Puts legend below plot
            xanchor="center",
            x=0.5
        )
    )
    st.plotly_chart(fig, use_container_width=True)


    # Create a DataFrame from the list of dictionaries
    df = pd.DataFrame(all_faces_info)

    # Group the DataFrame by 'emotion' and aggregate other columns
    df_grouped = df.groupby('emotion').agg({
        'index': lambda x: ', '.join(map(str, x)),
        'age': lambda x: ', '.join(map(str, x)),
        'gender': lambda x: ', '.join(map(str, x)),
        'race': lambda x: ', '.join(map(str, x))
    })

    # Reset the index of the DataFrame to get a row number
    df_grouped.reset_index(inplace=True)

    # Add a new column at the beginning for row number
    df_grouped.insert(0, '', range(1, 1 + len(df_grouped)))

    # Create a table using Plotly
    fig = go.Figure(data=[go.Table(
        header=dict(values=list(df_grouped.columns),
                    fill_color='blue',
                    align='left',
                    font=dict(color='white', size=12)),
        cells=dict(values=[df_grouped[col].tolist() for col in df_grouped.columns],
                  fill_color=['lightgreen']+['white']*(len(df_grouped.columns)-1),
                  align='left',
                  font=dict(color='darkslategray', size=11),
                  line_color='darkslategray'))
    ])

    # Set the layout to have a transparent background
    fig.update_layout({
        'plot_bgcolor': 'rgba(0, 0, 0, 0)',
        'paper_bgcolor': 'rgba(0, 0, 0, 0)',
    })

    # Display the table
    st.plotly_chart(fig, use_container_width=True)



# Add a line break
st.markdown("<br>", unsafe_allow_html=True)

quote = urllib.parse.quote("Discover the power of smart Computer Vision for Emotion detection and insights with this innovative web app. Unleash the potential of data-driven insights for emotion detection, age, sex, and race, crafted meticulously by Kedir Nasir Omer. Experience a new dimension of personal emotion detection and analysis, all within the comfort of your smart device.")

SOCIAL_MEDIA = {
    "gmail": {
        "url": f"https://mail.google.com/mail/u/0/?view=cm&fs=1&to=&su=&body={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/512/281/281769.png"
    },
    "whatsapp": {
        "url": f"https://api.whatsapp.com/send?text={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/128/174/174879.png"
    },
    "telegram": {
        "url": f"https://t.me/share/url?url={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/128/2111/2111646.png"
    },
    "linkedin": {
        "url": f"https://www.linkedin.com/shareArticle?url={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/128/174/174857.png"
    },
    "twitter": {
        "url": f"https://twitter.com/intent/tweet?url={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/512/124/124021.png"
    },
    "facebook": {
        "url": f"https://www.facebook.com/sharer.php?u={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/128/124/124010.png"
    },
    "reddit": {
        "url": f"https://reddit.com/submit?url={quote}",
        "icon": "https://cdn-icons-png.flaticon.com/512/2111/2111589.png"
    }
}

def create_icon(icon_name: str):
    link = SOCIAL_MEDIA[icon_name]["url"]
    file_url = SOCIAL_MEDIA[icon_name]["icon"]
    return f'<a href="{link}" target="_blank"><img src="{file_url}" alt="{icon_name}" width="30" height="30" style="display:inline-block; margin-right:10px;"/></a>'

st.markdown(f'<p style="font-family: {font_family}; font-size: {font_size};">Share this app with your loved ones and friends, and help them to play with it:</p>', unsafe_allow_html=True)

icons_html = "".join([create_icon(icon) for icon in SOCIAL_MEDIA.keys()])
st.markdown(icons_html, unsafe_allow_html=True)


# Add a line break
st.markdown("<br><br>", unsafe_allow_html=True)

# "my_info" - my name and profession at the bottom left corner as footer
my_info = """
<div style="bottom: 0; left: 0; text-align: left;">Kedir Nasir Omer<br>(Data, AI, ML, Software and Cloud Practitioner)<br>©2023-2024. All rights reserved.</div>
"""
st.markdown(f'<p style="font-family: {font_family}; font-size: 20;">{my_info}</p>', unsafe_allow_html=True)

# Display social media icons side by side with a smaller font size
social_icons = """
<a href="https://kedibeki.my.canva.site"><img src="https://cdn-icons-png.flaticon.com/128/841/841364.png" alt="Website" width="25" height="25"></a>
<a href="https://www.linkedin.com/in/kediromer/"><img src="https://cdn-icons-png.flaticon.com/128/2504/2504923.png" alt="LinkedIn" width="25" height="25"></a>
<a href="https://github.com/kedibeki"><img src="https://cdn-icons-png.flaticon.com/128/733/733553.png" alt="GitHub" width="25" height="25"></a>
<a href="mailto:kedirnasir10@gmail.com"><img src="https://cdn-icons-png.flaticon.com/128/5968/5968534.png" alt="Gmail" width="25" height="25"></a>
"""

# write the texts and social media icons
st.markdown(f'<p style="font-family: {font_family}; font-size: 16;">{social_icons}</p>', unsafe_allow_html=True)
