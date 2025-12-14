import streamlit as st
import cv2
import os
import tempfile
import logging
from pathlib import Path
from ultralytics import YOLO
import time
import torch

# Set MPS fallback for operations not supported on MPS
# This allows MPS to fall back to CPU for unsupported ops like NMS
if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Configure page
st.set_page_config(
    page_title="Vehicle Detection & Tracking",
    page_icon="🚗",
    layout="wide"
)

# Silence Ultralytics logs
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Configuration
FONT_SCALE_LABEL = 0.45
BOX_THICKNESS = 2
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]  # bicycle, car, motorcycle, bus, train, truck

# Model path
MODEL_PATH = "Model/yolo11l.pt"

@st.cache_resource
def load_model():
    """Load YOLO model (cached for performance)"""
    try:
        model = YOLO(MODEL_PATH)
        # Use GPU if available
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if device == 'cpu':
            # Check for MPS (Metal Performance Shaders) on Apple Silicon
            # Note: MPS has limitations with some operations (like NMS)
            # Use CPU directly for better reliability and performance
            # MPS fallback is slower than native CPU, so we prefer CPU
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # MPS is available but has limitations - use CPU for reliability
                # The environment variable PYTORCH_ENABLE_MPS_FALLBACK is set
                # but we'll use CPU directly for better performance
                device = 'cpu'  # Use CPU for better compatibility
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file exists at: Model/yolo11l.pt")
        return None, None

def process_video(input_video_path, output_video_path, model, device, progress_bar, status_text, process_every_n=1):
    """
    Process video with vehicle detection and tracking
    
    Args:
        input_video_path: Path to input video
        output_video_path: Path to save output video
        model: YOLO model instance
        device: Device to use (cuda/mps/cpu)
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text container
        process_every_n: Process every Nth frame (1 = all frames, 2 = every other frame, etc.)
    """
    # Open input video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Could not open input video")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        cap.release()
        raise ValueError("Could not initialize video writer")
    
    # Get class names
    class_list = model.names
    
    # Process frames
    frame_count = 0
    processed_count = 0
    start_time = time.time()
    last_update_time = start_time
    update_interval = 0.5  # Update progress every 0.5 seconds
    
    # Determine optimal image size for processing (smaller = faster)
    # Use 640 for faster processing, or original size for better accuracy
    imgsz = 640 if width > 1280 or height > 720 else None
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame or copy previous
        if frame_count % process_every_n == 0 or frame_count == 1:
            # Run YOLO tracking with optimizations
            # Use half precision only on CUDA (not MPS or CPU)
            use_half = device == 'cuda'
            
            # Build track arguments - only include imgsz if it's not None
            track_args = {
                'persist': True,
                'classes': VEHICLE_CLASSES,
                'verbose': False,
                'device': device,
                'conf': 0.25,  # Confidence threshold
                'iou': 0.45,   # IoU threshold for NMS
                'half': use_half  # Use half precision on CUDA only
            }
            
            # Only add imgsz if we want to resize (not None)
            if imgsz is not None:
                track_args['imgsz'] = imgsz
            
            results = model.track(frame, **track_args)
            
            # Draw detections
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu()
                track_ids = results[0].boxes.id.int().cpu().tolist()
                class_indices = results[0].boxes.cls.int().cpu().tolist()
                
                for box, track_id, class_idx in zip(boxes, track_ids, class_indices):
                    x1, y1, x2, y2 = map(int, box)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    
                    class_name = class_list[class_idx]
                    
                    # Draw bounding box (green)
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 200, 0),
                        BOX_THICKNESS
                    )
                    
                    # Draw center point (red)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                    
                    # Draw label
                    cv2.putText(
                        frame,
                        f"{class_name} | ID {track_id}",
                        (x1, max(y1 - 6, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        FONT_SCALE_LABEL,
                        (255, 255, 255),
                        1
                    )
            
            processed_count += 1
        
        # Write frame
        out.write(frame)
        
        # Update progress less frequently to reduce overhead
        current_time = time.time()
        if current_time - last_update_time >= update_interval:
            if total_frames > 0:
                progress = frame_count / total_frames
                progress_bar.progress(progress)
                elapsed_time = current_time - start_time
                fps_processing = frame_count / elapsed_time if elapsed_time > 0 else 0
                device_display = device.upper() if device else "CPU"
                status_text.text(f"Processing frame {frame_count}/{total_frames} | Speed: {fps_processing:.1f} FPS | Device: {device_display}")
            last_update_time = current_time
    
    # Release resources
    cap.release()
    out.release()
    
    return frame_count

def cleanup_files(*file_paths):
    """Remove temporary files"""
    for file_path in file_paths:
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            st.warning(f"Could not delete {file_path}: {str(e)}")

def main():
    # Header
    st.title("🚗 Vehicle Detection & Tracking System")
    st.markdown("### Powered by YOLO11, BoT-SORT & ByteTrack")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading YOLO11 model..."):
        model, device = load_model()
    
    if model is None:
        st.stop()
    
    device_name = device.upper() if device else "CPU"
    st.success(f"✅ Model loaded successfully! Using device: {device_name}")
    
    # File upload section
    st.markdown("### 📤 Upload Video")
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video file to detect and track vehicles"
    )
    
    if uploaded_file is not None:
        # Create temporary directory for processing
        temp_dir = tempfile.mkdtemp()
        input_video_path = os.path.join(temp_dir, f"input_{uploaded_file.name}")
        output_video_path = os.path.join(temp_dir, f"output_{uploaded_file.name}")
        
        # Save uploaded file
        with open(input_video_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.success(f"✅ Video uploaded: {uploaded_file.name}")
        
        # Display video info
        cap = cv2.VideoCapture(input_video_path)
        if cap.isOpened():
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            cap.release()
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Resolution", f"{width}x{height}")
            with col2:
                st.metric("FPS", fps)
            with col3:
                st.metric("Frames", total_frames)
            with col4:
                st.metric("Duration", f"{duration:.1f}s")
        
        # Processing options
        with st.expander("⚙️ Processing Options", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                process_every = st.selectbox(
                    "Process every N frames",
                    options=[1, 2, 3, 4, 5],
                    index=0,
                    help="1 = all frames (best quality, slower), Higher = faster but may skip some detections"
                )
            with col2:
                st.info(f"💡 Current device: **{device_name}**\n\nFor faster processing, use GPU if available.")
        
        # Process button
        if st.button("🎬 Process Video", type="primary", use_container_width=True):
            # Processing section
            st.markdown("### ⚙️ Processing Video")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Process video
                frame_count = process_video(
                    input_video_path,
                    output_video_path,
                    model,
                    device,
                    progress_bar,
                    status_text,
                    process_every_n=process_every
                )
                
                progress_bar.progress(1.0)
                status_text.text(f"✅ Processing complete! Processed {frame_count} frames.")
                
                st.success("🎉 Video processed successfully!")
                
                # Download section
                st.markdown("### 📥 Download Processed Video")
                
                if os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                    
                    with open(output_video_path, "rb") as video_file:
                        st.download_button(
                            label=f"⬇️ Download Processed Video ({file_size:.2f} MB)",
                            data=video_file,
                            file_name=f"processed_{uploaded_file.name}",
                            mime="video/mp4",
                            use_container_width=True
                        )
                    
                    # Cleanup after download
                    if st.button("🧹 Clean Up Files", use_container_width=True):
                        cleanup_files(input_video_path, output_video_path)
                        try:
                            os.rmdir(temp_dir)
                        except:
                            pass
                        st.success("✅ Files cleaned up successfully!")
                        st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error processing video: {str(e)}")
                cleanup_files(input_video_path, output_video_path)
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload a video file to get started")
        
        # Example video info
        st.markdown("#### 📋 Supported Features:")
        st.markdown("""
        - ✅ **Multi-Vehicle Detection**: Cars, Buses, Trucks, Motorcycles, Bicycles, Trains
        - ✅ **Object Tracking**: Persistent tracking IDs across frames
        - ✅ **Real-time Processing**: Fast and efficient video processing
        - ✅ **Visual Annotations**: Bounding boxes, center points, and labels
        """)
    
    # Footer with credits - Always visible at bottom
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; padding: 20px; color: #666;'>
            <p>Developed with ❤️ by <strong>Fahad Pathan</strong></p>
            <p style='font-size: 0.9em;'>
                Powered by YOLO11 | BoT-SORT & ByteTrack | Python & OpenCV
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()

