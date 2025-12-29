import os
import time
import librosa
import soundfile as sf
from mutagen import File
import wave
import subprocess

def wait_for_audio_ready(file_path, min_size_mb=0.01, max_wait_time=60, check_interval=1, min_duration=0.1):
    """
    Check if an audio file is ready for use.
    
    Args:
        file_path: path to the audio file to check
        min_size_mb: minimum file size (in MB)
        max_wait_time: maximum waiting time (in seconds)
        check_interval: interval between checks (in seconds)
        min_duration: minimum duration of audio (in seconds)
    
    Returns:
        bool: True if the file is ready, False if timeout
    """
    print(f"🎵 Checking audio file: {file_path}")
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
            print(f"📏 Current file size: {current_size / (1024*1024):.2f} MB")
            
            # Check minimum size requirement
            if current_size < min_size_bytes:
                print(f"⚠️ File size below minimum ({min_size_mb} MB). Waiting...")
                time.sleep(check_interval)
                continue
            
            # Check if file size is stable (not still being written)
            if current_size == last_size:
                stable_count += 1
                if stable_count >= 3:  # Stable for 3 consecutive checks
                    print("📊 File size stable, validating audio integrity...")
                    
                    # Validate audio readability
                    if _validate_audio_file(file_path, min_duration):
                        return True
                    
                    time.sleep(check_interval)
            else:
                stable_count = 0
                last_size = current_size
                print(f"🔄 File size still changing...")
                time.sleep(check_interval)
                
        except Exception as e:
            print(f"⚠️ Error while checking file: {e}")
            time.sleep(check_interval)
    
    print(f"❌ Timeout after {max_wait_time}s")
    return False


def _validate_audio_file(file_path, min_duration=0.1):
    """
    Validate the audio file using multiple methods.
    """
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # Method 1: Using librosa (best for most formats)
    try:
        duration = librosa.get_duration(path=file_path)
        if duration >= min_duration:
            print(f"✅ Valid audio (librosa) - Duration: {duration:.2f}s")
            return True
        else:
            print(f"⚠️ Audio too short: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Librosa error: {e}")
    
    # Method 2: Using soundfile
    try:
        with sf.SoundFile(file_path) as f:
            frames = len(f)
            samplerate = f.samplerate
            duration = frames / samplerate
            if duration >= min_duration:
                print(f"✅ Valid audio (soundfile) - Duration: {duration:.2f}s, SR: {samplerate}Hz")
                return True
            else:
                print(f"⚠️ Audio too short: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ SoundFile error: {e}")
    
    # Method 3: Using mutagen (good for metadata)
    try:
        audio_file = File(file_path)
        if audio_file is not None and hasattr(audio_file, 'info'):
            duration = audio_file.info.length
            if duration >= min_duration:
                print(f"✅ Valid audio (mutagen) - Duration: {duration:.2f}s")
                return True
            else:
                print(f"⚠️ Audio too short: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ Mutagen error: {e}")
    
    # Method 4: Using wave (for WAV files only)
    if file_ext == '.wav':
        try:
            with wave.open(file_path, 'rb') as wav_file:
                frames = wav_file.getnframes()
                sample_rate = wav_file.getframerate()
                duration = frames / sample_rate
                if duration >= min_duration:
                    print(f"✅ Valid WAV file - Duration: {duration:.2f}s, SR: {sample_rate}Hz")
                    return True
                else:
                    print(f"⚠️ Audio too short: {duration:.2f}s < {min_duration}s")
        except Exception as e:
            print(f"⚠️ Wave error: {e}")
    
    # Method 5: Using ffprobe (requires ffmpeg)
    try:
        result = subprocess.run([
            'ffprobe', '-v', 'quiet', '-show_entries', 
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', 
            file_path
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            duration = float(result.stdout.strip())
            if duration >= min_duration:
                print(f"✅ Valid audio (ffprobe) - Duration: {duration:.2f}s")
                return True
            else:
                print(f"⚠️ Audio too short: {duration:.2f}s < {min_duration}s")
    except Exception as e:
        print(f"⚠️ ffprobe error: {e}")
    
    print("❌ Unable to validate audio file with any method")
    return False


# ========== SIMPLE USAGE ==========

def simple_audio_check(file_path):
    """
    Simple one-line audio check.
    """
    return wait_for_audio_ready(file_path, min_size_mb=0.05, max_wait_time=30, min_duration=0.5)

def quick_audio_check(file_path, timeout=15):
    """
    Quick audio check with short timeout.
    """
    return wait_for_audio_ready(file_path, min_size_mb=0.01, max_wait_time=timeout, min_duration=0.1)


# ========== TEMPLATE EXAMPLES ==========

# Template 1: Basic audio processing
def process_audio_with_check():
    input_file = "input_audio.wav"
    output_file = "output_audio.wav"
    
    # Check input file
    if not wait_for_audio_ready(input_file):
        print("❌ Input file not ready")
        return False
    
    # Perform your audio processing here
    print("🎵 Processing audio...")
    # your_audio_processing_code()
    
    # Check output file
    if wait_for_audio_ready(output_file, min_duration=1.0):
        print("✅ Audio processing successful!")
        return True
    else:
        print("❌ Output file not created or invalid")
        return False


# Template 2: Batch processing with validation
def batch_audio_processing(input_files, output_dir):
    """
    Process multiple audio files with validation.
    """
    results = []
    
    for input_file in input_files:
        print(f"\n🎵 Processing: {input_file}")
        
        # Check input file
        if not wait_for_audio_ready(input_file):
            print(f"❌ Skipping file: {input_file}")
            results.append((input_file, False, "Input not ready"))
            continue
        
        # Create output filename
        filename = os.path.basename(input_file)
        name, ext = os.path.splitext(filename)
        output_file = os.path.join(output_dir, f"{name}_processed{ext}")
        
        # Perform processing
        try:
            # your_audio_processing_function(input_file, output_file)
            
            # Validate output
            if wait_for_audio_ready(output_file, min_duration=0.5):
                print(f"✅ Success: {output_file}")
                results.append((input_file, True, output_file))
            else:
                print(f"❌ Invalid output: {output_file}")
                results.append((input_file, False, "Invalid output"))
                
        except Exception as e:
            print(f"❌ Processing error: {e}")
            results.append((input_file, False, str(e)))
    
    return results


# Template 3: Monitor audio render/export progress
def monitor_audio_export(output_path, expected_duration=None, timeout=120):
    """
    Monitor the audio rendering/exporting process.
    """
    print(f"🎵 Monitoring audio export: {output_path}")
    
    # Adjust timeout for large files
    if expected_duration and expected_duration > 60:
        timeout = max(timeout, expected_duration * 2)
    
    min_duration = expected_duration * 0.8 if expected_duration else 0.5
    
    success = wait_for_audio_ready(
        output_path, 
        min_size_mb=0.1,
        max_wait_time=timeout,
        min_duration=min_duration
    )
    
    if success:
        print("🎉 Audio export complete!")
        return True
    else:
        print("💥 Audio export failed or timed out!")
        return False


# # ========== EXAMPLE USAGE ==========
# if __name__ == "__main__":
#     audio_file = "test_audio.wav"
    
#     print("=== Test 1: Simple check ===")
#     if simple_audio_check(audio_file):
#         print("Ready to use!")
    
#     print("\n=== Test 2: Quick check ===")
#     if quick_audio_check(audio_file):
#         print("Quick check passed!")
    
#     print("\n=== Test 3: Detailed check ===")
#     if wait_for_audio_ready(audio_file, min_size_mb=0.1, max_wait_time=60, min_duration=2.0):
#         print("Detailed check passed!")
