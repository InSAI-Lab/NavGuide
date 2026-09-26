"""PCM audio compression and resampling utilities."""
import os
import wave
import struct
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class AudioCompressor:
    """Provide mu-law and IMA ADPCM codecs and PCM resampling."""

    @staticmethod
    def pcm16_to_ulaw(pcm_data: bytes) -> bytes:
        """Encode 16-bit PCM as 8-bit mu-law, halving the sample payload."""
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        ulaw_data = bytearray()
        for sample in samples:
            ulaw_byte = AudioCompressor._linear_to_ulaw(sample)
            ulaw_data.append(ulaw_byte)

        return bytes(ulaw_data)

    @staticmethod
    def ulaw_to_pcm16(ulaw_data: bytes) -> bytes:
        """Decode 8-bit mu-law into 16-bit PCM."""
        pcm_samples = []
        for ulaw_byte in ulaw_data:
            pcm_sample = AudioCompressor._ulaw_to_linear(ulaw_byte)
            pcm_samples.append(pcm_sample)

        return np.array(pcm_samples, dtype=np.int16).tobytes()

    @staticmethod
    def _linear_to_ulaw(sample: int) -> int:
        """Encode a signed 16-bit PCM sample as mu-law."""
        ULAW_MAX = 0x1FFF
        ULAW_BIAS = 0x84

        sample = max(-32768, min(32767, sample))

        sign = 0
        if sample < 0:
            sign = 0x80
            sample = -sample

        sample = sample + ULAW_BIAS

        if sample > ULAW_MAX:
            sample = ULAW_MAX

        exponent = 7
        for exp in range(7, -1, -1):
            if sample & (0x4000 >> exp):
                exponent = exp
                break

        mantissa = (sample >> (exponent + 3)) & 0x0F
        ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF

        return ulawbyte

    @staticmethod
    def _ulaw_to_linear(ulawbyte: int) -> int:
        """Decode a mu-law sample into signed 16-bit PCM."""
        ULAW_BIAS = 0x84

        ulawbyte = ~ulawbyte & 0xFF
        sign = ulawbyte & 0x80
        exponent = (ulawbyte >> 4) & 0x07
        mantissa = ulawbyte & 0x0F

        sample = ((mantissa << 3) + ULAW_BIAS) << exponent

        if sign:
            sample = -sample

        return sample

    @staticmethod
    def pcm16_to_adpcm(pcm_data: bytes) -> bytes:
        """Encode 16-bit PCM as 4-bit IMA ADPCM."""
        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # IMA ADPCM step table.
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]

        # IMA ADPCM index adjustment table.
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

        adpcm_data = bytearray()
        predicted = 0
        step_index = 0

        # Pack two 4-bit samples into each byte.
        for i in range(0, len(samples), 2):
            byte = 0

            for j in range(2):
                if i + j < len(samples):
                    sample = samples[i + j]

                    diff = sample - predicted

                    step = step_table[step_index]
                    adpcm_sample = 0

                    if diff < 0:
                        adpcm_sample = 8
                        diff = -diff

                    if diff >= step:
                        adpcm_sample |= 4
                        diff -= step

                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 2
                        diff -= step

                    step >>= 1
                    if diff >= step:
                        adpcm_sample |= 1

                    step = step_table[step_index]
                    diff = 0
                    if adpcm_sample & 4:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 2:
                        diff += step
                    step >>= 1
                    if adpcm_sample & 1:
                        diff += step
                    step >>= 1
                    diff += step

                    if adpcm_sample & 8:
                        predicted -= diff
                    else:
                        predicted += diff

                    if predicted > 32767:
                        predicted = 32767
                    elif predicted < -32768:
                        predicted = -32768

                    step_index += index_table[adpcm_sample & 7]
                    if step_index < 0:
                        step_index = 0
                    elif step_index > 88:
                        step_index = 88

                    if j == 0:
                        byte = adpcm_sample
                    else:
                        byte |= (adpcm_sample << 4)

            adpcm_data.append(byte)

        # Store the initial predictor and step index in the header.
        header = struct.pack('<hB', predicted, step_index)
        return header + bytes(adpcm_data)

    @staticmethod
    def adpcm_to_pcm16(adpcm_data: bytes) -> bytes:
        """Decode 4-bit IMA ADPCM into 16-bit PCM."""
        if len(adpcm_data) < 3:
            return b''

        predicted, step_index = struct.unpack('<hB', adpcm_data[:3])
        adpcm_bytes = adpcm_data[3:]

        # IMA ADPCM step table.
        step_table = [
            7, 8, 9, 10, 11, 12, 13, 14, 16, 17,
            19, 21, 23, 25, 28, 31, 34, 37, 41, 45,
            50, 55, 60, 66, 73, 80, 88, 97, 107, 118,
            130, 143, 157, 173, 190, 209, 230, 253, 279, 307,
            337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
            876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066,
            2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358,
            5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899,
            15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767
        ]

        # IMA ADPCM index adjustment table.
        index_table = [-1, -1, -1, -1, 2, 4, 6, 8]

        pcm_samples = []

        for byte in adpcm_bytes:
            # Decode both 4-bit samples in the byte.
            for shift in [0, 4]:
                adpcm_sample = (byte >> shift) & 0x0F

                step = step_table[step_index]
                diff = 0

                if adpcm_sample & 4:
                    diff += step
                step >>= 1
                if adpcm_sample & 2:
                    diff += step
                step >>= 1
                if adpcm_sample & 1:
                    diff += step
                step >>= 1
                diff += step

                if adpcm_sample & 8:
                    predicted -= diff
                else:
                    predicted += diff

                if predicted > 32767:
                    predicted = 32767
                elif predicted < -32768:
                    predicted = -32768

                pcm_samples.append(predicted)

                step_index += index_table[adpcm_sample & 7]
                if step_index < 0:
                    step_index = 0
                elif step_index > 88:
                    step_index = 88

        return np.array(pcm_samples, dtype=np.int16).tobytes()

    @staticmethod
    def downsample_pcm16(pcm_data: bytes, from_rate: int = 16000, to_rate: int = 8000) -> bytes:
        """Resample PCM16 to the requested rate."""
        if from_rate == to_rate:
            return pcm_data

        samples = np.frombuffer(pcm_data, dtype=np.int16)

        # Decimate by two for the 16 kHz to 8 kHz conversion.
        if from_rate == 16000 and to_rate == 8000:
            downsampled = samples[::2]
        else:
            # Use linear interpolation for other sample rates.
            ratio = to_rate / from_rate
            new_length = int(len(samples) * ratio)
            downsampled = np.interp(
                np.linspace(0, len(samples) - 1, new_length),
                np.arange(len(samples)),
                samples
            ).astype(np.int16)

        return downsampled.tobytes()

class CompressedAudioCache:
    """Cache compressed audio files."""

    def __init__(self, compression_type: str = "adpcm", use_downsample: bool = False):
        """
        compression_type: "none", "ulaw", "adpcm"
        """
        self.compression_type = compression_type
        self.use_downsample = use_downsample
        self._cache = {}  # {filepath: compressed_data}
        self._original_sizes = {}  # {filepath: original_size}

    def load_and_compress(self, filepath: str) -> Optional[bytes]:
        """Load audio, resample to 8 kHz, and cache the compressed payload."""
        if filepath in self._cache:
            return self._cache[filepath]

        try:
            with wave.open(filepath, 'rb') as wav:
                channels = wav.getnchannels()
                sampwidth = wav.getsampwidth()
                framerate = wav.getframerate()

                if channels != 1:
                    logger.warning(f'{filepath} is not mono')
                if sampwidth != 2:
                    logger.warning(f'{filepath} is not 16-bit audio')

                frames = wav.readframes(wav.getnframes())

                # Use the first channel for stereo input.
                if channels == 2:
                    import audioop
                    frames = audioop.tomono(frames, sampwidth, 1, 0)

                # Resample to 8 kHz while preserving pitch and duration.
                if framerate != 8000:
                    import audioop
                    frames, _ = audioop.ratecv(frames, sampwidth, 1, framerate, 8000, None)
                    framerate = 8000

                self._original_sizes[filepath] = len(frames)

                if self.compression_type == "ulaw":
                    compressed = AudioCompressor.pcm16_to_ulaw(frames)
                    # Header: 1-byte codec identifier and 4-byte original payload length.
                    header = struct.pack('!BI', 0x01, len(frames))  # mu-law codec identifier.
                    compressed = header + compressed
                elif self.compression_type == "adpcm":
                    compressed = AudioCompressor.pcm16_to_adpcm(frames)
                    # Header: 1-byte codec identifier and 4-byte original payload length.
                    header = struct.pack('!BI', 0x02, len(frames))  # ADPCM codec identifier.
                    compressed = header + compressed
                else:
                    compressed = frames

                self._cache[filepath] = compressed

                compression_ratio = len(compressed) / self._original_sizes[filepath]
                logger.info(f'[COMPRESSION] {os.path.basename(filepath)}: {self._original_sizes[filepath]} -> {len(compressed)} bytes ({compression_ratio:.1%})')

                return compressed

        except Exception as e:
            logger.error(f'Audio compression failed {filepath}: {e}')
            return None

    def decompress(self, compressed_data: bytes) -> Optional[bytes]:
        """Decode a cached audio payload."""
        if not compressed_data or len(compressed_data) < 5:
            return compressed_data

        try:
            compression_type = compressed_data[0]
            if compression_type == 0x01:
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                ulaw_data = compressed_data[header_size:]

                pcm_data = AudioCompressor.ulaw_to_pcm16(ulaw_data)

                return pcm_data
            elif compression_type == 0x02:
                header_size = 5
                original_length = struct.unpack('!I', compressed_data[1:5])[0]
                adpcm_data = compressed_data[header_size:]

                pcm_data = AudioCompressor.adpcm_to_pcm16(adpcm_data)

                return pcm_data
            else:
                return compressed_data

        except Exception as e:
            logger.error(f'Audio decompression failed: {e}')
            return compressed_data

    def get_compression_stats(self) -> dict:
        """Return cache sizes and compression statistics."""
        total_original = sum(self._original_sizes.values())
        total_compressed = sum(len(data) for data in self._cache.values())

        return {
            "files_cached": len(self._cache),
            "total_original_size": total_original,
            "total_compressed_size": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 0,
            "bytes_saved": total_original - total_compressed
        }

# IMA ADPCM encodes each 16-bit sample using 4 bits.
# NAVGUIDE_COMPRESS_TYPE accepts none, ulaw, or adpcm.
compression_type = os.getenv("NAVGUIDE_COMPRESS_TYPE", "adpcm").lower()
if compression_type not in ["none", "ulaw", "adpcm"]:
    compression_type = "adpcm"
compressed_audio_cache = CompressedAudioCache(compression_type=compression_type, use_downsample=False)
