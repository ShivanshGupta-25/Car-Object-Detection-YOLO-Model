import streamlit as st
import time
import ultralytics
from ultralytics import YOLO
from PIL import Image
import glob
import os
import cv2

st.set_page_config(page_title="Car segmentation", page_icon="🚗", layout="wide")
st.title("🚗 Car Segmentation object detection  🚗")

# 2. File Upload UI
uploaded_file = st.file_uploader("Upload an image or video...", type=["jpg", "jpeg", "png", "mp4", "mov", "avi"])

if uploaded_file is not None:
    # Display the uploaded content
    if uploaded_file.type.startswith('image'):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        file_mime_type = uploaded_file.type
    else:
        st.video(uploaded_file)
        file_mime_type = uploaded_file.type

    # 3. Process on Button Click
    if st.button("Detect Cars"):
        with st.spinner("Processing..."):
            try:
                # Save uploaded file temporarily to disk (required for the File API)
                temp_file_path = f"{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Upload the file to the Yolo Model
                MODEL_PATH = "C:/Users/shiva/Documents/HartalkerModel/CarYoloModel/runs30epoch/car_parts_seg2/weights/best.pt"
                model = YOLO(MODEL_PATH)

                # 4. Generate Prediction
                results = model.predict(
                    source=temp_file_path,
                    conf=0.25,
                    iou=0.5,
                    save=True
                )
                time.sleep(4)

                # Directory where YOLO saved output
                save_dir = results[0].save_dir

                # Display result
                is_image = uploaded_file.type.startswith("image")
                if is_image:
                    output_images = glob.glob(os.path.join(save_dir, "*.jpg"))
                    if output_images:
                        st.image(output_images[0], caption="Prediction Result", use_container_width=True)
                else:
                    # Convert to MP4 
                    path = os.listdir(save_dir)
                    path = path[0]
                    save_dir = os.path.join(save_dir, path)
                    avi_path = save_dir
                    mp4_path = avi_path.replace(".avi", ".mp4")

                    cap = cv2.VideoCapture(avi_path)

                    fps = cap.get(cv2.CAP_PROP_FPS)
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    out = cv2.VideoWriter(mp4_path, fourcc, fps, (width, height))

                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        out.write(frame)

                    cap.release()
                    out.release()   

                    print("✅ Converted to MP4:", mp4_path)

                    time.sleep(2)

                    # Display the output video
                    st.video(mp4_path)
                    # with open(mp4_path, "rb") as f:
                    #     st.video(f.read())



                    # st.video(video_bytes)
  
            except Exception as e:
                st.error(f"An error occurred: {e}")