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
        
        # Auto-detect best available device
        # Priority: CUDA (NVIDIA GPU) > CPU (MPS has limitations) > MPS (with fallback)
        device = 'cpu'
        device_name = 'CPU'
        
        # Check for CUDA (NVIDIA GPU) - best option
        if torch.cuda.is_available():
            device = 'cuda'
            device_name = f'CUDA ({torch.cuda.get_device_name(0)})'
        # Check for MPS (Metal Performance Shaders on Apple Silicon)
        # Note: MPS has limitations with torchvision operations (NMS, etc.)
        # We'll use CPU for better compatibility, but MPS is available as fallback
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            # MPS is available but has known issues with torchvision operations
            # Use CPU for reliability, but note that MPS is available
            device = 'cpu'
            device_name = 'CPU (MPS available but limited compatibility)'
            # MPS fallback is already enabled via environment variable
            # This allows operations to fall back to CPU when needed
        
        return model, device, device_name
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please ensure the model file exists at: Model/yolo11l.pt")
        return None, None, None

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
    
    # Store last detection results for skipped frames to maintain tracking
    last_detections = None
    
    # Determine optimal image size for processing (smaller = faster)
    # Optimize based on device type for maximum performance
    if device == 'cpu':
        # CPU: Use smaller image size for faster processing
        imgsz = 640 if width > 960 or height > 540 else None
    elif device == 'mps':
        # MPS: Use medium size
        imgsz = 640 if width > 1280 or height > 720 else None
    else:
        # CUDA: Can handle larger images, use original or 640 for very large
        imgsz = 640 if width > 1920 or height > 1080 else None
    
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
            # Improve tracking with better parameters for stable detection
            track_args = {
                'persist': True,
                'classes': VEHICLE_CLASSES,
                'verbose': False,
                'device': device,
                'conf': 0.15,  # Lower confidence threshold for better detection (was 0.20)
                'iou': 0.50,   # Slightly higher IoU threshold for better tracking stability
                'half': use_half,  # Use half precision on CUDA only
                'max_det': 500,  # Increase max detections to catch more vehicles
            }
            
            # Only add imgsz if we want to resize (not None)
            if imgsz is not None:
                track_args['imgsz'] = imgsz
            
            try:
                results = model.track(frame, **track_args)
                # Store results for skipped frames to maintain tracking
                last_detections = results
            except RuntimeError as e:
                # If MPS error occurs, fallback to CPU
                if 'mps' in str(e).lower() or 'torchvision' in str(e).lower():
                    if device == 'mps':
                        # Fallback to CPU for this operation
                        track_args['device'] = 'cpu'
                        results = model.track(frame, **track_args)
                        last_detections = results
                    else:
                        raise
                else:
                    raise
            
            processed_count += 1
        else:
            # For skipped frames, use last detection results to maintain tracking visibility
            results = last_detections
        
        # Draw detections (use current or last detections)
        if results is not None and results[0].boxes is not None and results[0].boxes.id is not None:
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

def render_footer():
    """Render footer with credits - always visible"""
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

def render_footer_sidebar():
    """Render footer in sidebar - always visible"""
    with st.sidebar:
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; padding: 10px; color: #666; font-size: 0.85em;'>
                <p>Developed with ❤️ by <strong>Fahad Pathan</strong></p>
                <p style='font-size: 0.8em;'>
                    YOLO11 | BoT-SORT & ByteTrack
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

def main():
    # Initialize session state for processing status and file paths
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'input_video_path' not in st.session_state:
        st.session_state.input_video_path = None
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None
    if 'temp_dir' not in st.session_state:
        st.session_state.temp_dir = None
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    
    # Render footer in sidebar (always visible)
    render_footer_sidebar()
    
    # Header
    st.title("🚗 Vehicle Detection & Tracking System")
    st.markdown("### Powered by YOLO11, BoT-SORT & ByteTrack")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading YOLO11 model..."):
        model, device, device_name = load_model()
    
    if model is None:
        st.stop()
    
    st.success(f"✅ Model loaded successfully! Using device: **{device_name}**")
    
    # File upload section
    st.markdown("### 📤 Upload Video")
    
    # Show warning if processing is in progress
    if st.session_state.processing:
        st.warning("⏳ Video processing in progress. Please wait...")
        st.info("Upload and processing are disabled while a video is being processed.")
    
    # Handle file upload
    uploaded_file = None
    if not st.session_state.processing:
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to detect and track vehicles",
            disabled=st.session_state.processing
        )
        
        # Save uploaded file and store paths in session state
        if uploaded_file is not None:
            # Create temporary directory for processing
            temp_dir = tempfile.mkdtemp()
            input_video_path = os.path.join(temp_dir, f"input_{uploaded_file.name}")
            output_video_path = os.path.join(temp_dir, f"output_{uploaded_file.name}")
            
            # Save uploaded file
            with open(input_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Store in session state
            st.session_state.input_video_path = input_video_path
            st.session_state.output_video_path = output_video_path
            st.session_state.temp_dir = temp_dir
            st.session_state.uploaded_file_name = uploaded_file.name
            
            st.success(f"✅ Video uploaded: {uploaded_file.name}")
    else:
        # Show placeholder when processing
        st.file_uploader(
            "Choose a video file (disabled during processing)",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload is disabled while processing",
            disabled=True
        )
        # Show uploaded file name if available
        if st.session_state.uploaded_file_name:
            st.info(f"📹 Processing: {st.session_state.uploaded_file_name}")
    
    # Use stored paths from session state
    if st.session_state.input_video_path and os.path.exists(st.session_state.input_video_path):
        input_video_path = st.session_state.input_video_path
        output_video_path = st.session_state.output_video_path
        temp_dir = st.session_state.temp_dir
        uploaded_file_name = st.session_state.uploaded_file_name
        
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
        
        # Processing options - default to 3 for better tracking accuracy
        # Frame options from 1 to 200
        frame_options = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 40, 50, 75, 100, 150, 200]
        
        # Set default to 3 (index 2) for better tracking - processing every 15th frame
        # causes tracking to lose objects. 3 provides good balance of speed and accuracy
        default_index = 2  # Process every 3rd frame by default
        
        with st.expander("⚙️ Processing Options", expanded=True):
            # Instruction about quality vs speed
            st.info("💡 **Quality Tip**: The **lower** the number you select, the **better** the processing quality and tracking accuracy. Lower values (1-5) provide the best results but are slower. Higher values (50+) are faster but may miss detections or lose tracking.")
            
            col1, col2 = st.columns(2)
            with col1:
                process_every = st.selectbox(
                    "Process every N frames",
                    options=frame_options,
                    index=default_index,
                    help=f"1 = all frames (best quality, slower), Higher = faster. Default: 3 (balanced). Lower values = better quality!"
                )
            with col2:
                device_info = f"💡 Current device: **{device_name}**"
                if device == 'cpu':
                    if 'MPS' in device_name:
                        device_info += "\n\n⚠️ Using CPU for compatibility. MPS (Apple Silicon GPU) has limitations with some operations."
                    else:
                        device_info += "\n\n⚠️ Using CPU. For faster processing, NVIDIA GPU (CUDA) is recommended."
                elif device == 'mps':
                    device_info += "\n\n✅ Using Apple Silicon GPU (MPS). Some operations may fallback to CPU."
                elif device == 'cuda':
                    device_info += "\n\n✅ Using NVIDIA GPU (CUDA). Maximum performance!"
                st.info(device_info)
        
        # Process button - disabled when processing
        process_button_disabled = st.session_state.processing
        
        if st.button(
            "🎬 Process Video" if not st.session_state.processing else "⏳ Processing...",
            type="primary",
            use_container_width=True,
            disabled=process_button_disabled
        ):
            # Set processing state
            st.session_state.processing = True
            
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
                
                # Reset processing state
                st.session_state.processing = False
                
                st.success("🎉 Video processed successfully!")
                
                # Download section
                st.markdown("### 📥 Download Processed Video")
                
                if os.path.exists(output_video_path):
                    file_size = os.path.getsize(output_video_path) / (1024 * 1024)  # MB
                    
                    with open(output_video_path, "rb") as video_file:
                        st.download_button(
                            label=f"⬇️ Download Processed Video ({file_size:.2f} MB)",
                            data=video_file,
                            file_name=f"processed_{uploaded_file_name}",
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
                        # Clear session state
                        st.session_state.input_video_path = None
                        st.session_state.output_video_path = None
                        st.session_state.temp_dir = None
                        st.session_state.uploaded_file_name = None
                        st.success("✅ Files cleaned up successfully!")
                        st.rerun()
                
            except Exception as e:
                # Reset processing state on error
                st.session_state.processing = False
                st.error(f"❌ Error processing video: {str(e)}")
                cleanup_files(input_video_path, output_video_path)
                st.rerun()
    
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

if __name__ == "__main__":
    main()

