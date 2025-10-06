import os
import time
import cv2

def wait_for_file_ready(file_path, min_size_mb=0.1, max_wait_time=60, check_interval=1):
    """
    Check if a file is ready to use.
    
    Args:
        file_path (str): Path to the file to check.
        min_size_mb (float): Minimum expected file size (in MB).
        max_wait_time (int): Maximum time to wait (in seconds).
        check_interval (int): Interval between checks (in seconds).
    
    Returns:
        bool: True if the file is ready, False if it times out.
    """
    print(f"🔍 Checking file: {file_path}")
    start_time = time.time()
    min_size_bytes = min_size_mb * 1024 * 1024
    last_size = 0
    stable_count = 0
    
    while time.time() - start_time < max_wait_time:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"⏳ File not found yet. Waiting {check_interval}s...")
            time.sleep(check_interval)
            continue
        
        try:
            # Check file size
            current_size = os.path.getsize(file_path)
            print(f"📏 Current file size: {current_size / (1024 * 1024):.2f} MB")
            
            # Check if file meets minimum size
            if current_size < min_size_bytes:
                print(f"⚠️ File size below minimum threshold ({min_size_mb} MB). Waiting...")
                time.sleep(check_interval)
                continue
            
            # Check if file size is stable (not being written)
            if current_size == last_size:
                stable_count += 1
                if stable_count >= 3:  # File stable for 3 consecutive checks
                    print("📊 File size stable. Verifying file integrity...")
                    
                    # Try opening file as a video to verify validity
                    try:
                        cap = cv2.VideoCapture(file_path)
                        if cap.isOpened():
                            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                            fps = cap.get(cv2.CAP_PROP_FPS)
                            cap.release()
                            
                            if frame_count > 0 and fps > 0:
                                print(f"✅ Valid video file - Frames: {frame_count}, FPS: {fps}")
                                return True
                            else:
                                print("❌ Invalid video file (no frames or fps)")
                        else:
                            print("❌ Failed to open video file")
                    except Exception as e:
                        print(f"❌ Error while verifying file: {e}")
                    
                    time.sleep(check_interval)
            else:
                stable_count = 0
                last_size = current_size
                print(f"🔄 File size still changing...")
                time.sleep(check_interval)
                
        except Exception as e:
            print(f"⚠️ Error while checking file: {e}")
            time.sleep(check_interval)
    
    print(f"❌ Timeout after {max_wait_time}s waiting for file readiness")
    return False


# ========== SIMPLE USAGE EXAMPLE ==========

def simple_file_check(file_path):
    """
    Simplified one-line file readiness check.
    """
    return wait_for_file_ready(file_path, min_size_mb=0.5, max_wait_time=30)


# ========== TEMPLATE FOR YOUR USE CASE ==========

def your_processing_function():
    """
    Example template for integrating file readiness check
    into your video processing workflow.
    """
    input_file = "input_video.mp4"
    output_file = "output_video.mp4"
    
    # Check input file readiness
    if not wait_for_file_ready(input_file):
        print("❌ Input file not ready.")
        return False
    
    # Perform your processing logic here
    # your_video_processing_code()
    
    # Check output file readiness
    if wait_for_file_ready(output_file):
        print("✅ Processing completed successfully!")
        return True
    else:
        print("❌ Output file was not created properly.")
        return False
