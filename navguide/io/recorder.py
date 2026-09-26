"""Record ESP32 camera frames and speaker audio on a shared timeline."""

import os
import cv2
import wave
import numpy as np
import threading
import time
from datetime import datetime
from navguide.runtime.config import asset_path

class SyncRecorder:
    """Synchronize audio recording with the video timeline."""

    def __init__(self, output_dir=None, fps=15.0):
        """Initialize the output directory and target video frame rate."""
        if output_dir is None:
            output_dir = asset_path("NAVGUIDE_RECORDINGS_DIR", "recordings")
        self.output_dir = output_dir
        self.fps = fps
        self.frame_duration = 1.0 / fps

        os.makedirs(output_dir, exist_ok=True)

        self.is_recording = False
        self.start_time = None

        self.video_writer = None
        self.video_path = None
        self.last_frame = None
        self.frame_count = 0

        self.audio_writer = None
        self.audio_path = None
        self.audio_buffer = bytearray()
        self.last_audio_time = 0.0

        # Recording format: mono PCM16 at 16 kHz.
        self.sample_rate = 16000
        self.sample_width = 2  # 16bit = 2 bytes
        self.channels = 1

        self.lock = threading.Lock()

        self.frames_written = 0
        self.audio_bytes_written = 0
        self.last_log_time = time.time()

        print(f'[RECORDER] Initialized, FPS={fps}, output directory={output_dir}')

    def start_recording(self):
        """Start a recording session."""
        if self.is_recording:
            print('[RECORDER] Warning: recording is already active')
            return False

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_path = os.path.join(self.output_dir, f"video_{timestamp}.avi")
        self.audio_path = os.path.join(self.output_dir, f"audio_{timestamp}.wav")

        self.start_time = time.time()
        self.last_audio_time = 0.0
        self.frame_count = 0
        self.frames_written = 0
        self.audio_bytes_written = 0
        self.audio_buffer.clear()
        self.last_frame = None

        try:
            self.audio_writer = wave.open(self.audio_path, 'wb')
            self.audio_writer.setnchannels(self.channels)
            self.audio_writer.setsampwidth(self.sample_width)
            self.audio_writer.setframerate(self.sample_rate)
        except Exception as e:
            print(f'[RECORDER] Failed to initialize audio file: {e}')
            return False

        self.is_recording = True
        print(f'[RECORDER] Recording started')
        print(f'  Video: {self.video_path}')
        print(f'  Audio: {self.audio_path}')
        return True

    def add_frame(self, jpeg_data: bytes):
        """Record a video frame from JPEG bytes."""
        if not self.is_recording:
            return

        try:
            with self.lock:
                arr = np.frombuffer(jpeg_data, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

                if frame is None:
                    print(f'[RECORDER] Warning: frame decode failed')
                    return

                if self.video_writer is None:
                    height, width = frame.shape[:2]
                    # Use XVID for broad AVI playback support.
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    self.video_writer = cv2.VideoWriter(
                        self.video_path,
                        fourcc,
                        self.fps,
                        (width, height)
                    )

                    if not self.video_writer.isOpened():
                        print(f'[RECORDER] Error: video writer initialization failed')
                        self.is_recording = False
                        return

                    print(f'[RECORDER] Video writer initialized: {width}x{height} @ {self.fps}fps')

                self.video_writer.write(frame)
                self.frame_count += 1
                self.frames_written += 1
                self.last_frame = frame

                current_video_time = self.frame_count * self.frame_duration

                # Pad audio to the video timeline.
                self._sync_audio_to_video(current_video_time)

                now = time.time()
                if now - self.last_log_time > 10.0:
                    elapsed = now - self.start_time
                    avg_fps = self.frames_written / elapsed if elapsed > 0 else 0
                    audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)
                    print(f'[RECORDER] Recording, frames={self.frames_written}, measured FPS={avg_fps:.1f}, video duration={current_video_time:.1f}s, audio duration={audio_duration:.1f}s')
                    self.last_log_time = now

        except Exception as e:
            print(f'[RECORDER] Failed to record frame: {e}')
            import traceback
            traceback.print_exc()

    def add_audio(self, pcm_data: bytes, text: str = ""):
        """Record PCM16 audio with an optional transcript for diagnostics."""
        if not self.is_recording:
            return

        try:
            with self.lock:
                current_video_time = self.frame_count * self.frame_duration

                # Pad silence to the current video timestamp before writing audio.
                self._sync_audio_to_video(current_video_time)

                self.audio_writer.writeframes(pcm_data)
                audio_duration = len(pcm_data) / (self.sample_rate * self.sample_width)
                self.last_audio_time = current_video_time + audio_duration
                self.audio_bytes_written += len(pcm_data)

                if text:
                    print(f'[RECORDER] Recording speech: {text[:30]}... (timestamp={current_video_time:.2f}s, duration={audio_duration:.2f}s)')

        except Exception as e:
            print(f'[RECORDER] Failed to record audio: {e}')

    def _sync_audio_to_video(self, video_time: float):
        """Pad audio with silence up to the video timestamp in seconds."""
        silence_duration = video_time - self.last_audio_time

        if silence_duration > 0.01:  # Ignore gaps shorter than 10 ms.
            silence_samples = int(silence_duration * self.sample_rate)
            silence_bytes = silence_samples * self.sample_width
            silence_data = b'\x00' * silence_bytes

            self.audio_writer.writeframes(silence_data)
            self.audio_bytes_written += len(silence_data)
            self.last_audio_time = video_time

    def stop_recording(self):
        """Stop recording and close the output files."""
        if not self.is_recording:
            return

        print('[RECORDER] Saving recording files...')
        self.is_recording = False

        with self.lock:
            try:
                if self.frame_count > 0:
                    final_video_time = self.frame_count * self.frame_duration
                    self._sync_audio_to_video(final_video_time)
            except Exception as e:
                print(f'[RECORDER] Final audio synchronization failed: {e}')

            if self.video_writer is not None:
                try:
                    print('[RECORDER] Closing video writer...')
                    self.video_writer.release()
                    print('[RECORDER] Video writer closed')
                except Exception as e:
                    print(f'[RECORDER] Failed to close video writer: {e}')
                finally:
                    self.video_writer = None

            if self.audio_writer is not None:
                try:
                    print('[RECORDER] Closing audio writer...')
                    self.audio_writer.close()
                    print('[RECORDER] Audio writer closed')
                except Exception as e:
                    print(f'[RECORDER] Failed to close audio writer: {e}')
                finally:
                    self.audio_writer = None

            try:
                elapsed = time.time() - self.start_time if self.start_time else 0
                video_duration = self.frame_count * self.frame_duration
                audio_duration = self.audio_bytes_written / (self.sample_rate * self.sample_width)

                print(f"\n{'='*60}")
                print(f'[RECORDER] Recording complete')
                print(f"{'='*60}")
                print(f'  Elapsed time: {elapsed:.1f} seconds')
                print(f'\n  Video: {self.video_path}')
                print(f'    Frames: {self.frames_written}')
                print(f'    Duration: {video_duration:.2f} seconds')
                if elapsed > 0:
                    print(f'    Average FPS: {self.frames_written / elapsed:.1f}')
                print(f'\n  Audio: {self.audio_path}')
                print(f'    Data size: {self.audio_bytes_written / 1024:.1f} KB')
                print(f'    Duration: {audio_duration:.2f} seconds')
                print(f'\n  Timeline difference: {abs(video_duration - audio_duration):.3f} seconds')

                if os.path.exists(self.video_path):
                    video_size = os.path.getsize(self.video_path) / 1024 / 1024
                    print(f'  Video file size: {video_size:.2f} MB')
                else:
                    print(f'  Warning: video file was not created')

                if os.path.exists(self.audio_path):
                    audio_size = os.path.getsize(self.audio_path) / 1024
                    print(f'  Audio file size: {audio_size:.2f} KB')
                else:
                    print(f'  Warning: audio file was not created')

                print(f"{'='*60}\n")
            except Exception as e:
                print(f'[RECORDER] Failed to report statistics: {e}')

_global_recorder = None
_recorder_lock = threading.Lock()

def get_recorder():
    """Return the shared recorder instance."""
    global _global_recorder
    with _recorder_lock:
        if _global_recorder is None:
            _global_recorder = SyncRecorder()
        return _global_recorder

def start_recording():
    """Start a recording session."""
    recorder = get_recorder()
    return recorder.start_recording()

def stop_recording():
    """Stop recording and close the output files."""
    recorder = get_recorder()
    recorder.stop_recording()

def record_frame(jpeg_data: bytes):
    """Record a JPEG frame when recording is active."""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_frame(jpeg_data)

def record_audio(pcm_data: bytes, text: str = ""):
    """Record PCM16 audio when recording is active."""
    recorder = get_recorder()
    if recorder.is_recording:
        recorder.add_audio(pcm_data, text)
