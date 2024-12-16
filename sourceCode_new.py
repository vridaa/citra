import streamlit as st
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import scipy.ndimage as ndi
import math
import time
import uuid  # Sudah diimport dengan benar

# Contoh parameter awal 
parameter = {
    'brightness_beta': 0,
    'batas': 127,
    'contrast_alpha': 1.0,
    'smoothing_kernel': 3,
    'edge_detection_low': 50,
    'edge_detection_high': 150,
    'mirroring_type': "Horizontal",
    'rotation_angle': 0
}

# Dynamically update parameters based on selected method
def update_parameters(method):
    """Dynamically updates parameters based on selected processing method."""
    if method == "Brightness Adjustment":
        parameter['brightness_beta'] = st.slider("Intensitas Kecerahan", -50, 50, parameter.get('brightness_beta', 0), key="brightness_beta_slider")
    elif method == "Treshold":
        parameter['batas'] = st.slider("Ambang Batas", 0, 255, parameter.get('batas', 127), key="treshold_slider")
    elif method == "RGB":
        parameter["Red"] = st.slider("Red", 0, 255, parameter.get('Red', 127), key="red_slider")
        parameter["Green"] = st.slider("Green", 0, 255, parameter.get('Green', 127), key="green_slider")
        parameter["Blue"] = st.slider("Blue", 0, 255, parameter.get('Blue', 127), key="blue_slider")
    elif method == "Contrast Adjustment":
        parameter['contrast_threshold'] = st.slider("Nilai Ambang Kontras (m)", 0, 255, parameter.get('contrast_threshold', 127), key="contrast_threshold_slider")
        parameter['contrast_type'] = st.selectbox("Tipe Perbaikan Kontras", ["stretching", "thresholding"], index=["stretching", "thresholding"].index(parameter.get('contrast_type', "stretching")), key="contrast_type_selectbox")
    elif method == "Edge Detection":
        parameter['edge_detection_low'] = st.slider("Ambang Batas Rendah", 0, 255, parameter.get('edge_detection_low', 50), key="edge_detection_low_slider")
        parameter['edge_detection_high'] = st.slider("Ambang Batas Tinggi", 0, 255, parameter.get('edge_detection_high', 150), key="edge_detection_high_slider")
    elif method == "Rotation":
        # Slider untuk mengatur sudut rotasi
        parameter['rotation_angle'] = st.slider("Sudut Rotasi", -180, 180, parameter.get('rotation_angle', 45), key="rotation_angle_slider")
    elif method == "Translasi":
        parameter['translasi_m'] = st.slider("Translasi Horizontal (m)", -100, 100, 0, key="translasi_m_slider")
        parameter['translasi_n'] = st.slider("Translasi Vertikal (n)", -100, 100, 0, key="translasi_n_slider")
    elif method == "Mirroring":
        parameter['mirroring_type'] = st.selectbox("Jenis Mirroring", ["Horizontal", "Vertical", "Both"], index=["Horizontal", "Vertical", "Both"].index(parameter.get('mirroring_type', "Horizontal")), key="mirroring_type_selectbox")
    elif method == "Gaussian Noise":
        parameter['prob_noise'] = st.slider("Probabilitas Noise Gaussian", 0.0, 1.0, 0.05, key="prob_noise_slider")
        
def adjust_brightness_per_pixel(image_array, brightness_beta):
    adjusted_image = image_array.astype(np.float32) + brightness_beta
    
    # Pastikan nilai berada dalam rentang 0-255
    adjusted_image = np.clip(adjusted_image, 0, 255)

    
    return adjusted_image.astype(np.uint8)

def manual_canny_edge_detection(image_array, low_threshold, high_threshold):
    # Step 1: Convert to Grayscale
    gray_image = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]

    # Step 2: Apply Gaussian Blur
    blurred_image = ndi.gaussian_filter(gray_image, sigma=1.4)

    # Step 3: Compute Sobel gradients
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Ix = ndi.convolve(blurred_image, Kx)
    Iy = ndi.convolve(blurred_image, Ky)

    # Step 4: Calculate gradient magnitude and direction
    magnitude = np.hypot(Ix, Iy)
    direction = np.arctan2(Iy, Ix)
    direction = direction * 180. / np.pi
    direction[direction < 0] += 180

    suppressed = np.zeros_like(magnitude)
    rows, cols = magnitude.shape

    # Non-maximum Suppression with bounds checking
    for i in range(1, rows-1):
        for j in range(1, cols-1):
            angle = direction[i, j]
            q, r = 255, 255

            # Determine neighboring pixels to interpolate
            if (0 <= angle < 22.5) or (157.5 <= angle <= 180):
                q = magnitude[i, j+1] if j+1 < cols else 0
                r = magnitude[i, j-1] if j-1 >= 0 else 0
            elif 22.5 <= angle < 67.5:
                q = magnitude[i+1, j-1] if (i+1 < rows and j-1 >= 0) else 0
                r = magnitude[i-1, j+1] if (i-1 >= 0 and j+1 < cols) else 0
            elif 67.5 <= angle < 112.5:
                q = magnitude[i+1, j] if i+1 < rows else 0
                r = magnitude[i-1, j] if i-1 >= 0 else 0
            elif 112.5 <= angle < 157.5:
                q = magnitude[i-1, j-1] if (i-1 >= 0 and j-1 >= 0) else 0
                r = magnitude[i+1, j+1] if (i+1 < rows and j+1 < cols) else 0

            # Suppress non-maximum pixels
            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                suppressed[i, j] = magnitude[i, j]
            else:
                suppressed[i, j] = 0

    # Step 6: Double Thresholding
    strong_edges = (suppressed >= high_threshold)
    weak_edges = ((suppressed >= low_threshold) & (suppressed < high_threshold))

    # Step 7: Edge Tracking by Hysteresis
    final_edges = np.zeros_like(suppressed)
    strong_i, strong_j = np.where(strong_edges)
    weak_i, weak_j = np.where(weak_edges)

    # Set strong edges
    final_edges[strong_i, strong_j] = 255

    # Connect weak edges to strong edges
    for i, j in zip(weak_i, weak_j):
        if ((final_edges[i+1, j] == 255 if i+1 < rows else 0) or 
            (final_edges[i-1, j] == 255 if i-1 >= 0 else 0) or 
            (final_edges[i, j+1] == 255 if j+1 < cols else 0) or 
            (final_edges[i, j-1] == 255 if j-1 >= 0 else 0) or 
            (final_edges[i+1, j+1] == 255 if (i+1 < rows and j+1 < cols) else 0) or 
            (final_edges[i-1, j-1] == 255 if (i-1 >= 0 and j-1 >= 0) else 0) or 
            (final_edges[i+1, j-1] == 255 if (i+1 < rows and j-1 >= 0) else 0) or 
            (final_edges[i-1, j+1] == 255 if (i-1 >= 0 and j+1 < cols) else 0)):
            final_edges[i, j] = 255

    return final_edges.astype(np.uint8)
def smoothing_with_average_filter(image_array, kernel_size=3):
    
    # Get the height, width, and number of channels
    height, width, channels = image_array.shape
    
    # Initialize output image
    smoothed_image = np.zeros_like(image_array)
    
    # Define the padding size
    pad_size = kernel_size // 2
    
    # Pad the image to handle borders
    padded_image = np.pad(image_array, ((pad_size, pad_size), (pad_size, pad_size), (0, 0)), mode='reflect')
    
    # Number of elements in the kernel
    K = kernel_size * kernel_size
    
    # Loop through each pixel in the original image
    for x in range(height):
        for y in range(width):
            # Extract the neighborhood for each channel
            R_neighborhood = padded_image[x:x + kernel_size, y:y + kernel_size, 0]
            G_neighborhood = padded_image[x:x + kernel_size, y:y + kernel_size, 1]
            B_neighborhood = padded_image[x:x + kernel_size, y:y + kernel_size, 2]
            
            # Apply the average formula for each channel
            smoothed_image[x, y, 0] = np.sum(R_neighborhood) / K
            smoothed_image[x, y, 1] = np.sum(G_neighborhood) / K
            smoothed_image[x, y, 2] = np.sum(B_neighborhood) / K
    
    return smoothed_image.astype(np.uint8)
def sharpening_with_laplacian(image_array):
    # Initialize an output image with the same shape as the input image
    sharpened_image = np.zeros_like(image_array)
    
    # Loop through each channel (R, G, B)
    for channel in range(3):
        # Apply Laplacian operator to the channel
        laplacian = cv2.Laplacian(image_array[:, :, channel], cv2.CV_64F)
        
        # Add the Laplacian result back to the original channel for sharpening
        sharpened_channel = image_array[:, :, channel] - laplacian
        
        # Clip the values to be in range [0, 255] and store in the output
        sharpened_image[:, :, channel] = np.clip(sharpened_channel, 0, 255)
    
    return sharpened_image.astype(np.uint8)

def rotate_image(image_array, degrees):
    theta = math.radians(degrees)
    height, width = image_array.shape[:2]
    x_center = width // 2
    y_center = height // 2
    rotated_image = np.zeros_like(image_array)

    for x in range(width):
        for y in range(height):
            x_orig = int(math.cos(theta) * (x - x_center) + math.sin(theta) * (y - y_center) + x_center)
            y_orig = int(-math.sin(theta) * (x - x_center) + math.cos(theta) * (y - y_center) + y_center)
            
            if 0 <= x_orig < width and 0 <= y_orig < height:
                rotated_image[y, x] = image_array[y_orig, x_orig]
            else:
                rotated_image[y, x] = (0, 0, 0)
    
    return rotated_image
def translate_image(image_array, m, n):
    height, width, channels = image_array.shape
    
    # Buat array baru dengan ukuran yang sama
    translated_image = np.zeros_like(image_array)
    
    # Hitung batas mulai translasi
    start_m = max(0, m)
    start_n = max(0, n)
    
    for x in range(start_m, width):
        for y in range(start_n, height):
            x_baru = x - m
            y_baru = y - n
            
            if 0 <= x_baru < width and 0 <= y_baru < height:
                translated_image[y, x] = image_array[y_baru, x_baru]
            else:
                translated_image[y, x] = (0, 0, 0)  # Set piksel di luar batas ke hitam
    
    return translated_image
# Function to plot histogram
def plot_histogram(image, title="Histogram"):
    """Plots the histogram of an image, handling both grayscale and RGB images."""
    fig, ax = plt.subplots()
    
    # Check if the image has multiple color channels (RGB) or is grayscale
    if len(image.shape) == 3:  # RGB image
        color = ('r', 'g', 'b')
        for i, col in enumerate(color):
            histr = cv2.calcHist([image], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
    else:  # Grayscale image
        histr = cv2.calcHist([image], [0], None, [256], [0, 256])
        ax.plot(histr, color='black')  # Use black for grayscale histogram
    
    ax.set_title(title)
    return fig

# Main categories and sub-methods
categories = {
    "Filtering": ["Grayscale", "Negative", "Treshold", "RGB"],
    "Noise": ["Gaussian Noise"],  # Added noise method here
    "Adjustment": ["Brightness Adjustment", "Contrast Adjustment", "Smoothing", "Sharpening"],
    "Edge Detection": ["Edge Detection"],
    "Mirroring & Rotate": ["Mirroring", "Rotation"],
    "Translasi": ["Translasi"]
}

# Processing function
def process_image(image_array, method):
    """Applies selected image processing method to the image array."""
    if method == "Grayscale":
        grayscale_image = 0.299 * image_array[:, :, 0] + 0.587 * image_array[:, :, 1] + 0.114 * image_array[:, :, 2]
        return grayscale_image.astype(np.uint8)
    elif method == "Negative":
        return 255 - image_array
    elif method == "Treshold":
        gray_image = np.mean(image_array, axis=2).astype(np.uint8)
        thresholded_image = np.where(gray_image > parameter['batas'], 255, 0).astype(np.uint8)
        return thresholded_image
    elif method == "RGB":
        red, green, blue = parameter["Red"], parameter["Green"], parameter["Blue"]
        # Apply each color adjustment to the corresponding channel
        adjusted_image = image_array.copy()
        adjusted_image[:, :, 0] = np.clip(adjusted_image[:, :, 0] * (red / 127), 0, 255)
        adjusted_image[:, :, 1] = np.clip(adjusted_image[:, :, 1] * (green / 127), 0, 255)
        adjusted_image[:, :, 2] = np.clip(adjusted_image[:, :, 2] * (blue / 127), 0, 255)
        return adjusted_image.astype(np.uint8)
    elif method == "Gaussian Noise":
        # Gunakan nilai dari parameter yang sudah diatur di update_parameters
        prob_noise = parameter.get('prob_noise', 0.05)
        noise = np.random.normal(0, 1, image_array.shape)
        noisy_image = np.clip(image_array + noise * prob_noise * 255, 0, 255)
        return noisy_image.astype(np.uint8)
    elif method =="Brightness Adjustment":
        brightness_beta = parameter.get('brightness_beta', 0)
        return adjust_brightness_per_pixel(image_array, brightness_beta)
    elif method == "Contrast Adjustment":
        m = parameter.get('contrast_threshold', 127)  # Nilai ambang untuk kontras
        contrast_type = parameter.get('contrast_type', 'stretching')  # 'stretching' atau 'thresholding'
        
        if contrast_type == 'stretching':
            # Operasi peregangan kontras
            contrast_adjusted = np.where(image_array < m, image_array * 0.5, np.clip(image_array * 1.5, 0, 255))
        elif contrast_type == 'thresholding':
            # Operasi pengambangan (thresholding)
            contrast_adjusted = np.where(image_array < m, 0, 255)
        
        return contrast_adjusted.astype(np.uint8)
    elif method == "Rotation":
        rotation_angle = parameter.get('rotation_angle', 45)  # Nilai default 45 derajat
        rotated_image = rotate_image(image_array, rotation_angle)
        return rotated_image
    elif method == "Smoothing":
        # Apply Gaussian blur for smoothing effect
        kernel_size = parameter['smoothing_kernel']  # Kernel size from parameter
        smoothed_image = smoothing_with_average_filter(image_array, kernel_size=kernel_size)
        return smoothed_image
    elif method == "Sharpening":
        # Call the Laplacian-based sharpening function
        sharpened_image = sharpening_with_laplacian(image_array)
        return sharpened_image
    elif method=="Edge Detection":
        low_threshold = parameter['edge_detection_low']
        high_threshold = parameter['edge_detection_high']
        return manual_canny_edge_detection(image_array, low_threshold, high_threshold)
    elif method == "Mirroring":
        flip_type = parameter.get('mirroring_type', 'Horizontal')
        if flip_type == "Horizontal":
            return image_array[:, ::-1]
        elif flip_type == "Vertical":
            return image_array[::-1]
        elif flip_type == "Both":
            return image_array[::-1, ::-1]
    if method == "Translasi":
        m = parameter.get('translasi_m', 0)
        n = parameter.get('translasi_n', 0)
        translated_image = translate_image(image_array, m, n)
        return translated_image
    return image_array



def upload_image():
    uploaded_file = st.file_uploader("Upload Gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)

        # Main category selection
        category = st.selectbox("Pilih Kategori", list(categories.keys()))
        
        # Sub-method selection based on the main category
        method = st.selectbox("Pilih Metode Pengolahan", categories[category])
        
        # Update parameters based on the selected method
        update_parameters(method)
        
        # Process the image based on the selected method
        processed_image = process_image(image_array, method)
        
        col1, col2 = st.columns(2)
        col1.image(image, caption="Gambar Asli")
        col2.image(processed_image, caption=f"Gambar Setelah Diolah ({method})")
        
        fig1 = plot_histogram(image_array, "Histogram Warna (Asli)")
        fig2 = plot_histogram(processed_image, "Histogram Warna (Setelah Diolah)")
        
        col1.pyplot(fig1)
        col2.pyplot(fig2)

def display_camera():
    """Displays real-time webcam feed with dynamic processing options and unique keys."""
    
    if 'run_camera' not in st.session_state:
        st.session_state['run_camera'] = False  # Initialize camera state

    # Attempt to access the camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.warning("Tidak dapat membuka kamera.")
        return  # Exit the function if camera cannot be accessed

    col1, col2 = st.columns(2)
    stframe_original = col1.empty()
    stframe_hist_original = col1.empty()
    stframe_processed = col2.empty()
    stframe_hist_processed = col2.empty()

    # Main category selection
    category = st.selectbox("Pilih Kategori", list(categories.keys()), key="category_select")

    # Sub-method selection based on the main category
    method = st.selectbox("Pilih Metode Pengolahan", categories[category], key="method_select")

    # Update parameters based on the selected method
    update_parameters(method)

    # Button to start, stop, and update the processing
    if st.button("Mulai Kamera" if not st.session_state['run_camera'] else "Stop Kamera"):
        st.session_state['run_camera'] = not st.session_state['run_camera']
    
    # Tombol untuk memperbarui gambar setelah mengubah parameter
    update_button = st.button("Update Frame")

    # Capture and display frames only if the camera is running
    while st.session_state['run_camera']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Gagal mendapatkan frame dari kamera.")
            break

        # Convert to RGB format for processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image based on the selected method only if update_button is pressed
        if update_button:
            processed_frame = process_image(frame_rgb, method)
        
        else:
            processed_frame = frame_rgb

        # Display the original frame and histogram in the first column
        stframe_original.image(frame_rgb, caption="Gambar Asli (Real-Time)", channels="RGB", use_column_width=True)
        stframe_hist_original.pyplot(plot_histogram(frame_rgb, "Histogram Warna (Asli)"))

        # Ensure processed frame has RGB channels for display consistency
        if len(processed_frame.shape) == 2:
            processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2RGB)

        # Display processed frame and histogram in the second column
        stframe_processed.image(processed_frame, caption=f"Gambar Setelah Diolah ({method})", channels="RGB", use_column_width=True)
        stframe_hist_processed.pyplot(plot_histogram(processed_frame, "Histogram Warna (Setelah Diolah)"))

        # Adding a delay to reduce processing load (adjust as needed)
        time.sleep(0.1)

    # Release the camera and close OpenCV windows when done
    cap.release()
    cv2.destroyAllWindows()


menu = st.sidebar.radio("Pilih Opsi", ("Upload Gambar", "Tampilan Kamera"))
if menu == "Upload Gambar":
    st.subheader("Upload Gambar untuk Pengolahan")
    upload_image()
elif menu == "Tampilan Kamera":
    st.subheader("Tampilan Kamera Real-Time")
    display_camera()

