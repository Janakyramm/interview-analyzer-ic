import streamlit as st
import os


def save_uploaded_file(uploaded_file, path):
    if uploaded_file is not None:
        # if not os.path.exists(path):
        #     os.makedirs(path)
        with st.spinner("Uploading ...."):
            with open(path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            return st.success(f"Saved file: {uploaded_file.name} in {path}")



st.title('Video Uploader App')


uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "mov", "avi", "mkv"])
file_name = st.text_input(label="File Name (Enter Drive ID here)")
save_path = "./DownloadedVideos/"+file_name+".mp4"
if st.button("Upload Video"):
    if uploaded_file is not None:
        save_uploaded_file(uploaded_file, save_path)