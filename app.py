import streamlit as st
from ultralytics import YOLO
import tensorflow as tf  # Change this to import TensorFlow
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import pandas as pd
import time
from paddleocr import PaddleOCR, draw_ocr
import re
import dateparser
import os
import matplotlib.pyplot as plt

# Initialize PaddleOCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')
# Define the class names based on your dataset
class_names = [
    'fresh_apple', 'fresh_banana', 'fresh_bitter_gourd', 'fresh_capsicum',
    'fresh_orange', 'fresh_tomato', 'stale_apple', 'stale_banana',
    'stale_bitter_gourd', 'stale_capsicum', 'stale_orange', 'stale_tomato'
]

# Team details
team_members = [
    {"name": "Aman Deep", "image": "aman.jpg"},  # Replace with actual paths to images
    {"name": "Abhishek Kumar Choudhary", "image": "myimage.jpg"},
    {"name": "Gaurav Lodhi", "image": "gaurav.jpg"},
    {"name": "Anand Jha", "image": "anandimg.jpg"}
]

# Function to preprocess the images for the model
from PIL import Image
import numpy as np

def preprocess_image(image):
    """
    Preprocess the input image for model prediction.

    Args:
        image (PIL.Image): Input image in PIL format.

    Returns:
        np.ndarray: Preprocessed image array ready for prediction.
    """
    try:
        # Resize image to match model input size
        img = image.resize((128, 128), Image.LANCZOS)  # Using LANCZOS filter for high-quality resizing

        # Convert image to NumPy array
        img_array = np.array(img)

        # Check if the image is grayscale and convert to RGB if needed
        if img_array.ndim == 2:  # Grayscale image
            img_array = np.stack([img_array] * 3, axis=-1)  # Convert to 3-channel RGB
        elif img_array.shape[2] == 1:  # Single-channel image
            img_array = np.concatenate([img_array, img_array, img_array], axis=-1)  # Convert to RGB

        # Normalize pixel values to [0, 1] range
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)  # Shape: (1, 128, 128, 3)

        return img_array

    except Exception as e:
        print(f"Error processing image: {e}")
        return None  # Return None if there's an error


# Function to create a high-quality circular mask for an image
def make_image_circular1(img, size=(256, 256)):
    img = img.resize(size, Image.LANCZOS)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    output = ImageOps.fit(img, mask.size, centering=(0.5, 0.5))
    output.putalpha(mask)  # Apply the mask as transparency
    return output
# Function to check if a file exists
def file_exists(file_path):
    return os.path.isfile(file_path)

def make_image_circular(image):
    # Create a circular mask
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)

    # Apply the mask to the image
    circular_image = Image.new("RGB", image.size)
    circular_image.paste(image.convert("RGBA"), (0, 0), mask)

    return circular_image

# Function to extract dates from recognized text using regex
def extract_dates_with_dateparser(texts, result):
    date_texts = []
    date_boxes = []
    date_scores = []
    
    def is_potential_date(text):
        valid_date_pattern = r'^(0[1-9]|[12][0-9]|3[01])[-/.]?(0[1-9]|1[0-2])[-/.]?(\d{2}|\d{4})$|' \
                             r'^(0[1-9]|[12][0-9]|3[01])[-/.]?[A-Za-z]{3}[-/.]?(\d{2}|\d{4})$|' \
                             r'^(0[1-9]|1[0-2])[-/.]?(\d{2}|\d{4})$|' \
                             r'^[A-Za-z]{3}[-/.]?(\d{2}|\d{4})$'
        return bool(re.match(valid_date_pattern, text))

    dates_found = []
    for i, text in enumerate(texts):
        if is_potential_date(text):  # Only process texts that are potential dates
            parsed_date = dateparser.parse(text, settings={'DATE_ORDER': 'DMY'})
            if parsed_date:
                dates_found.append(parsed_date.strftime('%Y-%m-%d'))  # Store as 'YYYY-MM-DD'
                date_texts.append(text)  # Store the original text
                date_boxes.append(result[0][i][0])  # Store the bounding box
                date_scores.append(result[0][i][1][1])  # Store confidence score
    return dates_found, date_texts, date_boxes, date_scores

# Function to display circular images in a matrix format
def display_images_in_grid(images, max_images_per_row=4):
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Calculate number of rows

    for i in range(num_rows):
        cols = st.columns(min(max_images_per_row, num_images - i * max_images_per_row))
        for j, img in enumerate(images[i * max_images_per_row:(i + 1) * max_images_per_row]):
            with cols[j]:
                st.image(img, use_column_width=True)

# Function to display team members in circular format
def display_team_members(members, max_members_per_row=4):
    num_members = len(members)
    num_rows = (num_members + max_members_per_row - 1) // max_members_per_row  # Calculate number of rows

    for i in range(num_rows):
        cols = st.columns(min(max_members_per_row, num_members - i * max_members_per_row))
        for j, member in enumerate(members[i * max_members_per_row:(i + 1) * max_members_per_row]):
            with cols[j]:
                img = Image.open(member["image"])  # Load the image
                circular_img = make_image_circular(img)  # Convert to circular format
                st.image(circular_img, use_column_width=True)  # Display the circular image
                st.write(member["name"])  # Display the name below the image

# Title and description
st.title("Flipkart GRID 6.0")
# Team Details with links
st.sidebar.title("Flipkart Grid 6.0")
st.sidebar.write("DELHI TECHNOLOGICAL UNIVERSITY")

# Navbar with task tabs
st.sidebar.title("Navigation")
st.sidebar.write("Team Name: aman.dp121")
app_mode = st.sidebar.selectbox("Choose the task", ["Welcome","Project Details", "Task 1", "Task 2", "Task 3", "Task 4", "Team Details"])
if app_mode == "Welcome":
    # Navigation Menu
    st.write("# Welcome to Flipkart GrID 6.0! 🎉")

    # Example for adding a local video
    video_file = open('Finalist.mp4', 'rb')  # Replace with the path to your video file
    video_bytes = video_file.read()
    # Embed the video using st.video()
    st.video(video_bytes) 

    # Add a welcome image
    welcome_image = Image.open("grid_banner.jpg")  # Replace with the path to your welcome image
    st.image(welcome_image, use_column_width=True)  # Display the welcome image
    
elif app_mode=="Project Details":
    st.markdown("""
    ## Navigation
    - [Project Overview](#project-overview)
    - [Proposal Round](#proposal-round)
    - [Problem Statement](#problem-statement)
    - [Proposed Solution](#proposed-solution)
    """)
    # Project Overview
    st.write("## Project Overview:")
    st.write(""" 
    1. **OCR to Extract Details** (20%):
        - Use OCR to read brand details, pack size, brand name, etc.
        - Train the model to read details from various products, including FMCG, OTC items, health supplements, personal care, and household items.
    
    2. **Using OCR for Expiry Date Details** (10%):
        - Validate expiry dates using OCR to read expiry and MRP details printed on items.
    
    3. **Image Recognition for Brand Recognition and Counting** (30%):
        - Use machine learning to recognize brands and count product quantities from images.
    
    4. **Detecting Freshness of Fresh Produce** (40%):
        - Assess the freshness of fruits and vegetables by analyzing various visual cues and patterns.
    """)
    
    st.write(""" 
    Our project aims to leverage OCR and image recognition to enhance product packaging analysis and freshness detection. 
    """)

    # Proposal Round
    st.write("## Proposal Round:")
    st.write(""" 
    **Format:** Use Case Submission & Code Review
    - Selected teams will submit detailed use case scenarios they plan to solve.
    - The submission should include a proposal outlining their approach and the code developed so far.
    - The GRID team will provide a set of images for testing the model.
    - Since this is an elimination stage, participants are encouraged to submit a video simulation of their solution on the image set provided to them, ensuring they can clearly articulate what they have solved. 
    - Teams working on detecting the freshness of produce may choose any fresh fruit/vegetable/bread, etc., and submit the freshness index based on the model.
    - The video will help demonstrate the effectiveness of their approach and provide a visual representation of their solution.
    
    Teams with the most comprehensive and innovative proposals will proceed to the final stage.
    """)

    # Problem Statement
    st.write("## Problem Statement:")
    st.write(""" 
    In today’s fast-paced retail environment, ensuring product quality and freshness is crucial for customer satisfaction. The Flipkart GrID 6.0 Challenge aims to address this issue by leveraging technology to enhance product packaging analysis and freshness detection. 

    Traditional methods of checking freshness often involve manual inspection, which can be time-consuming and prone to human error. Furthermore, with the increasing variety of products available, a more automated and reliable solution is needed to streamline this process.

    Our project focuses on developing an advanced system that utilizes Optical Character Recognition (OCR) and image recognition techniques to automate the extraction of product details from packaging. This will not only improve accuracy but also increase efficiency in assessing product freshness.
    """)

    # Proposed Solution
    st.write("## Proposed Solution:")
    st.write(""" 
    Our solution is designed to tackle the problem by implementing the following key components:

    ### 1. OCR for Product Detail Extraction
    We will use OCR technology to accurately extract critical information from product packaging, including:
    - Brand name
    - Pack size
    - Expiry date
    - MRP details

    This will allow for real-time analysis of product information, ensuring that customers receive accurate data about their purchases.

    ### 2. Freshness Detection using Image Recognition
    In conjunction with OCR, our model will utilize image recognition to assess the freshness of fruits, vegetables, and other perishable items. The model will be trained to classify products based on their appearance, detecting signs of spoilage and degradation. 

    ### 3. Data Validation and Reporting
    Our system will not only extract data but also validate expiry dates against the current date to ensure product safety. The results will be compiled into a user-friendly report that can be easily interpreted by retail staff.

    ### 4. Video Simulation
    To effectively demonstrate our solution, we will create a video simulation showcasing the functionality of our system. This will include real-time examples of how our model processes images and extracts relevant information.

    ### 5. Proposal Submission
    As part of the proposal round, we will provide a comprehensive submission outlining our approach, methodology, and the code developed thus far. This submission will highlight the effectiveness of our solution and our readiness to proceed to the final stage of the challenge.
    
    Our team is committed to delivering a robust solution that not only meets but exceeds the expectations of the Flipkart GrID 6.0 Challenge.
    """)

elif app_mode == "Team Details":
    st.write("## Meet Our Team:")
    display_team_members(team_members)
    st.write("Delhi Technological University")

elif app_mode == "Task 1":
    st.write("## Task 1: 🖼️ OCR to Extract Details 📄")
    st.write("Using OCR to extract details from product packaging material, including brand name and pack size.")
    
    # File uploader for images (supports multiple files)
    uploaded_files = st.file_uploader("Upload images of products", type=["jpeg", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### Uploaded Images in Circular Format:")
        circular_images = []

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            circular_img = make_image_circular(img)  # Create circular images
            circular_images.append(circular_img)

        # Display the circular images in a matrix/grid format
        display_images_in_grid(circular_images, max_images_per_row=4)

        # Function to simulate loading process with a progress bar
        def simulate_progress():
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)
        # Function to remove gibberish using regex (removes non-alphanumeric chars, filters out very short text)
        def clean_text(text):
            # Keep text with letters, digits, and spaces, and remove short/irrelevant text
            return re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

        # Function to extract the most prominent text (product name) and other details
        def extract_product_info(results):
            product_name = ""
            product_details = ""
            largest_text_size = 0

            for line in results:
                for box in line:
                    text, confidence = box[1][0], box[1][1]
                    text_size = box[0][2][1] - box[0][0][1]  # Calculate height of the text box

                    # Clean the text to avoid gibberish
                    clean_text_line = clean_text(text)

                    if confidence > 0.7 and len(clean_text_line) > 2:  # Only consider confident, meaningful text
                        if text_size > largest_text_size:  # Assume the largest text is the product name
                            largest_text_size = text_size
                            product_name = clean_text_line
                        else:
                            product_details += clean_text_line + " "

            return product_name, product_details.strip()
        if st.button("Start Analysis"):
            simulate_progress()
            # Loop through each uploaded image and process them
            for uploaded_image in uploaded_files:
                # Load the uploaded image
                image = Image.open(uploaded_image)
                # st.image(image, caption=f'Uploaded Image: {uploaded_image.name}', use_column_width=True)

                # Convert image to numpy array for OCR processing
                img_array = np.array(image)

                # Perform OCR on the image
                st.write(f"Extracting details from {uploaded_image.name}...")
                result = ocr.ocr(img_array, cls=True)

                # Process the OCR result to extract product name and properties
                product_name, product_details = extract_product_info(result)

                # UI display for single image product details
                st.markdown("---")
                st.markdown(f"### **Product Name:** `{product_name}`")
                st.write(f"**Product Properties:** {product_details}")
                st.markdown("---")
        
    else:
        st.write("Please upload images to extract product details.")

elif app_mode == "Task 2":
    st.write("## Task 2:📅 Expiry Date Validation ✅")
    st.write("Use OCR to get expiry and MRP details printed on items.")
    # File uploader for images (supports multiple files)
    uploaded_files = st.file_uploader("Upload images of products containing expiry date", type=["jpeg", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### Uploaded Images in Circular Format:")
        circular_images = []

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            circular_img = make_image_circular(img)  # Create circular images
            circular_images.append(circular_img)

        # Display the circular images in a matrix/grid format
        display_images_in_grid(circular_images, max_images_per_row=4)

        # Function to simulate loading process with a progress bar
        def simulate_progress():
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)

        for idx, uploaded_file in enumerate(uploaded_files):
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            result = ocr.ocr(img_array, cls=True)

            if result and result[0]:
                # Extract recognized texts
                recognized_texts = [line[1][0] for line in result[0]]

                # Clean up recognized texts by removing extra spaces and standardizing formats
                cleaned_texts = []
                for text in recognized_texts:
                    cleaned_text = re.sub(r'\s+', ' ', text.strip())  # Replace multiple spaces with a single space
                    cleaned_text = cleaned_text.replace('.', '').replace(',', '')  # Remove dots and commas for date detection
                    cleaned_texts.append(cleaned_text)

                # Extract dates from recognized texts
                extracted_dates, date_texts, date_boxes, date_scores = extract_dates_with_dateparser(cleaned_texts, result)

                if extracted_dates:
                    # Display extracted dates
                    st.write("**Extracted Dates**:")
                    for date, text in zip(extracted_dates, date_texts):
                        st.write(f"Detected Date: **{date}**, Original Text: *{text}*")
                else:
                    st.write("No valid dates found in the image.")

                # Option to visualize the bounding boxes on the image
                if st.checkbox(f"Show image with highlighted dates for {uploaded_file.name}", key=f"highlight_{idx}"):
                    # Draw the OCR results on the image
                    image_with_boxes = draw_ocr(image, date_boxes, date_texts, date_scores,font_path='CedarvilleCursive-Regular.ttf')  # Removed font path

                    # Display the image with highlighted boxes
                    plt.figure(figsize=(10, 10))
                    plt.imshow(image_with_boxes)
                    plt.axis('off')  # Hide axes
                    st.pyplot(plt)
            else:
                st.write("No text detected in the image.")


def make_image_circular1(image):
    # Create a circular mask
    mask = Image.new("L", image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, image.size[0], image.size[1]), fill=255)

    # Apply the mask to the image
    circular_image = Image.new("RGB", image.size)
    circular_image.paste(image.convert("RGBA"), (0, 0), mask)

    return circular_image

def display_images_in_grid1(images, max_images_per_row=4):
    rows = (len(images) + max_images_per_row - 1) // max_images_per_row  # Calculate number of rows needed

    for i in range(0, len(images), max_images_per_row):
        cols_to_show = images[i:i + max_images_per_row]
        
        # Prepare to display in a grid format
        cols = st.columns(max_images_per_row)  # Create columns dynamically
        
        for idx, img in enumerate(cols_to_show):
            img = img.convert("RGB")  # Ensure the image is in RGB mode
            
            if idx < len(cols):
                cols[idx].image(img, use_column_width=True)

# Initialize your Streamlit app
if app_mode == "Task 3":
    st.write("## Task 3: Image Recognition 📸 and IR-Based Counting 📊")
    
    # File uploader for images (supports multiple files)
    uploaded_files = st.file_uploader("Upload images of fruits, vegetables, or products for brand recognition and freshness detection", 
                                      type=["jpeg", "png", "jpg"], accept_multiple_files=True)
    if uploaded_files:
        st.write("### Uploaded Images:")
        # Load the pre-trained YOLOv8 model
        model = YOLO('yolov9c.pt')  # Adjust path to your YOLO model if needed

        # Initialize a dictionary to store counts of detected products
        product_count_dict = {}
        circular_images = []
        images=[]

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            circular_img = make_image_circular(img)  # Create circular images
            circular_images.append(circular_img)
            images.append(img)

        # Display the circular images in a matrix/grid format
        display_images_in_grid(circular_images, max_images_per_row=4)

        detected_images = []
        
        for idx, image in enumerate(images):
            # Run object detection
            results = model(image)

            # Initialize counts for this image
            image_counts = {}

            # Display results with bounding boxes
            for result in results:
                img_with_boxes = result.plot()  # Get image with bounding boxes
                detected_images.append(make_image_circular(image.resize((150, 150))))  # Resize and make circular
                
                # Display detected object counts per class
                counts = result.boxes.cls.tolist()  # Extract class IDs
                class_counts = {int(cls): counts.count(cls) for cls in set(counts)}
                
                # Update the image counts for this image
                for cls_id, count in class_counts.items():
                    product_name = result.names[cls_id]  # Get the product name from class ID
                    image_counts[product_name] = count
                
                # Aggregate counts into the main product count dictionary
                for product, count in image_counts.items():
                    if product in product_count_dict:
                        product_count_dict[product] += count
                    else:
                        product_count_dict[product] = count

            # Option to visualize the bounding boxes on the image
            if st.checkbox(f"Show image with highlighted boxes for image {idx + 1}", key=f"checkbox_{idx}"):
                st.image(img_with_boxes, caption="Image with Highlighted Boxes", use_column_width=True)

        # Display the total counts as a bar chart
        st.write("### Total Product Counts Across All Images:")
        if product_count_dict:
            product_count_df = pd.DataFrame(product_count_dict.items(), columns=["Product", "Count"])
            st.bar_chart(product_count_df.set_index("Product"))
        else:
            st.write("No products detected.")

elif app_mode == "Task 4":
    st.write("## Task 4: 🍏 Fruit and Vegetable Freshness Detector 🍅")
    # Load the trained model
    try:
        model = tf.keras.models.load_model('fruit_freshness_model.h5')  # Using TensorFlow to load the model
        st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
    
    # File uploader for images (supports multiple files)
    uploaded_files = st.file_uploader("Upload images of fruits/vegetables", type=["jpeg", "png", "jpg"], accept_multiple_files=True)

    if uploaded_files:
        st.write("### Uploaded Images in Circular Format:")
        circular_images = []
        images=[]

        for uploaded_file in uploaded_files:
            img = Image.open(uploaded_file)
            circular_img = make_image_circular(img)  # Create circular images
            circular_images.append(circular_img)
            images.append(img)

        # Display the circular images in a matrix/grid format
        display_images_in_grid(circular_images, max_images_per_row=4)

        # Function to simulate loading process with a progress bar
        def simulate_progress():
            progress_bar = st.progress(0)
            for percent_complete in range(100):
                time.sleep(0.02)
                progress_bar.progress(percent_complete + 1)

        # Create an empty DataFrame to hold the image name and prediction results
        results_df = pd.DataFrame(columns=["Image", "Prediction"])

        # Create a dictionary to count the occurrences of each class
        class_counts = {class_name: 0 for class_name in class_names}

        # Button to initiate predictions
        if st.button("Run Prediction"):
            # Display progress bar
            simulate_progress()

            for idx, img in enumerate(images):  # Use circular images for predictions
                img_array = preprocess_image(img.convert('RGB'))  # Convert to RGB

                try:
                    # Perform the prediction
                    prediction = model.predict(img_array)

                    # Get the class with the highest probability
                    result = class_names[np.argmax(prediction)]
                    st.success(f'Prediction for Image {idx + 1}: **{result}**')

                    # Increment the class count
                    class_counts[result] += 1

                    # Add the result to the DataFrame
                    result_data = pd.DataFrame({"Image": [uploaded_files[idx].name], "Prediction": [result]})
                    results_df = pd.concat([results_df, result_data], ignore_index=True)

                except Exception as e:
                    st.error(f"Error occurred during prediction: {e}")

            # Display class distribution as a bar chart
            st.write("### Class Distribution:")
            class_counts_df = pd.DataFrame(list(class_counts.items()), columns=['Class', 'Count'])
            st.bar_chart(class_counts_df.set_index('Class'))

            # Option to download the prediction results as a CSV file
            st.write("### Download Results:")
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download prediction results as CSV",
                data=csv,
                file_name='prediction_results.csv',
                mime='text/csv',
            )

            # Display the dataframe after the graph
            st.write("### Prediction Data:")
            st.dataframe(results_df)

# Footer with animation
st.markdown(""" 
    <style>
        @keyframes fade-in {
            from { opacity: 0; }
            to { opacity: 1;}
        }
        .footer {
            text-align: center;
            font-size: 1.1em;
            animation: fade-in 2s;
            padding-top: 2rem;
        }
    </style>
    <div class="footer">
        <p>© 2024 Flipkart GRiD 6.0 Challenge. All rights reserved.</p>
    </div>
""", unsafe_allow_html=True)
