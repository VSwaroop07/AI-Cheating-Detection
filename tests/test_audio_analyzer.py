"""Comprehensive unit and integration test suite for AudioAnalyzer.

Tests:
1. Audio decoding (base64, raw PCM, numpy int16/float32, WAV files).
2. RMS and dBFS energy calculation.
3. Ambient noise baseline calibration over initial 20 seconds.
4. Voice activity detection on silence, loud noise, and speech.
5. GEMINI.md Rule 4 immunity: brief speech (<3.0s) produces NO events.
6. Sustained speech (>3.0s) triggers standard SuspiciousEvent.
7. Debouncing: continuous 5s speech produces 1 event, not 5.
8. Sudden loud noise detection with baseline ratio and absolute threshold.
9. Cross-modal speech without lip movement (motionless lips while speaking).
10. Multi-session concurrency and state isolation.
"""

import base64
import os
import numpy as np
import pytest
import scipy.io.wavfile as wavfile

from app.ml.audio_analyzer import AudioAnalyzer, AudioSessionState, AudioAnalysisResult
from app.schemas.event import ModalityEnum, SeverityEnum, SuspiciousEvent


@pytest.fixture(scope="module")
def audio_analyzer() -> AudioAnalyzer:
    """Fixture providing singleton AudioAnalyzer instance with ONNX VAD."""
    analyzer = AudioAnalyzer(use_onnx=True)
    return analyzer


@pytest.fixture
def sample_paths() -> dict:
    """Paths to test audio assets."""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return {
        "speech": os.path.join(data_dir, "speech_sample.wav"),
        "silence": os.path.join(data_dir, "silence.wav"),
        "loud_noise": os.path.join(data_dir, "loud_noise.wav"),
    }


# =====================================================================
# 1. Audio Decoding Tests
# =====================================================================

def test_decode_audio_numpy(audio_analyzer: AudioAnalyzer):
    """Test decoding numpy arrays of both int16 and float32 types."""
    # int16 array
    int16_arr = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
    decoded = audio_analyzer.decode_audio(int16_arr)
    assert decoded.dtype == np.float32
    assert np.isclose(decoded[0], 0.0)
    assert np.isclose(decoded[1], 0.5, atol=1e-3)
    assert np.isclose(decoded[2], -0.5, atol=1e-3)

    # float32 array
    float_arr = np.array([-1.0, 0.0, 1.0], dtype=np.float32)
    decoded_float = audio_analyzer.decode_audio(float_arr)
    assert np.array_equal(decoded_float, float_arr)


def test_decode_audio_raw_pcm_bytes(audio_analyzer: AudioAnalyzer):
    """Test decoding raw 16-bit little-endian PCM bytes."""
    samples = np.array([0, 16384, -16384], dtype=np.int16)
    raw_bytes = samples.tobytes()
    decoded = audio_analyzer.decode_audio(raw_bytes)
    assert len(decoded) == 3
    assert np.isclose(decoded[1], 0.5, atol=1e-3)


def test_decode_audio_base64_string(audio_analyzer: AudioAnalyzer):
    """Test decoding base64-encoded PCM bytes (with and without data URL prefix)."""
    samples = np.array([0, 32767, -32768], dtype=np.int16)
    b64_str = base64.b64encode(samples.tobytes()).decode("utf-8")

    # Plain base64
    decoded1 = audio_analyzer.decode_audio(b64_str)
    assert len(decoded1) == 3
    assert np.isclose(decoded1[1], 1.0, atol=1e-3)

    # Base64 with data URL prefix
    data_url = f"data:audio/pcm;base64,{b64_str}"
    decoded2 = audio_analyzer.decode_audio(data_url)
    assert len(decoded2) == 3
    assert np.isclose(decoded2[1], 1.0, atol=1e-3)


def test_decode_audio_wav_file(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test decoding WAV file bytes with RIFF headers."""
    with open(sample_paths["silence"], "rb") as f:
        wav_bytes = f.read()

    decoded = audio_analyzer.decode_audio(wav_bytes)
    assert len(decoded) > 0
    assert decoded.dtype == np.float32
    assert -1.0 <= np.max(decoded) <= 1.0


# =====================================================================
# 2. RMS Energy & dBFS Calculation Tests
# =====================================================================

def test_compute_rms_silence(audio_analyzer: AudioAnalyzer):
    """Test RMS and dBFS on complete digital silence."""
    silence = np.zeros(16000, dtype=np.float32)
    rms, dbfs = audio_analyzer.compute_rms(silence)
    assert rms == 0.0
    assert dbfs <= -100.0


def test_compute_rms_sine_wave(audio_analyzer: AudioAnalyzer):
    """Test RMS on known amplitude sine wave."""
    # Peak amplitude 1.0 sine wave -> theoretical RMS = 1.0 / sqrt(2) ≈ 0.7071
    t = np.linspace(0, 1.0, 16000, endpoint=False)
    sine = (np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rms, dbfs = audio_analyzer.compute_rms(sine)
    assert np.isclose(rms, 0.7071, atol=1e-2)
    assert np.isclose(dbfs, -3.01, atol=0.5)


# =====================================================================
# 3. Ambient Noise Calibration Tests
# =====================================================================

def test_ambient_noise_calibration_progression(audio_analyzer: AudioAnalyzer):
    """Test that baseline RMS calibrates over the first 20 seconds of session."""
    session_id = "test_calib_sess"
    audio_analyzer.reset_session(session_id)

    np.random.seed(42)
    # 25 1-second chunks of quiet ambient noise (RMS ≈ 0.01)
    for sec in range(25):
        chunk = np.random.normal(0, 0.01, 16000).astype(np.float32)
        res = audio_analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=chunk,
            timestamp=float(sec),
        )
        if sec < 20:
            assert res.is_calibrated is False
        else:
            assert res.is_calibrated is True

    state = audio_analyzer.get_session_state(session_id)
    assert state.is_calibrated is True
    assert np.isclose(state.baseline_rms, 0.01, atol=0.005)
    assert len(state.calibration_rms_samples) >= 15


# =====================================================================
# 4. Voice Activity Detection (VAD) Tests
# =====================================================================

def test_vad_on_silence(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test VAD returns False on pure silence WAV file."""
    sr, silence = wavfile.read(sample_paths["silence"])
    # 1 second of silence
    chunk = silence[:sr]
    is_speech, ratio, ts = audio_analyzer.detect_speech(
        audio_analyzer.decode_audio(chunk)
    )
    assert is_speech is False
    assert ratio == 0.0
    assert len(ts) == 0


def test_vad_on_speech(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test VAD detects speech in speech sample WAV file."""
    sr, speech = wavfile.read(sample_paths["speech"])
    # Use first 2 seconds where speaker vocalizes
    chunk = speech[:sr]
    is_speech, ratio, ts = audio_analyzer.detect_speech(
        audio_analyzer.decode_audio(chunk)
    )
    assert is_speech is True
    assert ratio > 0.30
    assert len(ts) > 0


# =====================================================================
# 5. GEMINI.md Rule 4 & Sustained Speech Tests
# =====================================================================

def test_brief_speech_immunity(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test Rule 4: Single brief vocalization (<3.0s) MUST NEVER trigger an event."""
    session_id = "test_rule4_brief_speech"
    audio_analyzer.reset_session(session_id)
    sr, speech = wavfile.read(sample_paths["speech"])

    # Feed 2 seconds of speech (below 3.0s threshold)
    events_collected = []
    for sec in range(2):
        chunk = speech[sec * sr : (sec + 1) * sr]
        res = audio_analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=chunk,
            timestamp=float(sec),
        )
        events_collected.extend(res.events)

    # Assert no sustained speech alert triggered
    speech_events = [e for e in events_collected if e.type == "SUSTAINED_SPEECH"]
    assert len(speech_events) == 0


def test_sustained_speech_trigger_and_schema(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test that continuous speech exceeding 3.0 seconds triggers SUSTAINED_SPEECH event."""
    session_id = "test_sustained_speech"
    audio_analyzer.reset_session(session_id)
    sr, speech = wavfile.read(sample_paths["speech"])

    events_collected = []
    # Feed 4 consecutive seconds of speech
    for sec in range(4):
        chunk = speech[sec * sr : (sec + 1) * sr]
        res = audio_analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=chunk,
            timestamp=float(sec),
        )
        events_collected.extend(res.events)

    speech_events = [e for e in events_collected if e.type == "SUSTAINED_SPEECH"]
    assert len(speech_events) >= 1

    event = speech_events[0]
    # Check strict GEMINI.md event schema conformance
    assert event.session_id == session_id
    assert event.modality == ModalityEnum.AUDIO
    assert event.type == "SUSTAINED_SPEECH"
    assert event.severity in (SeverityEnum.MEDIUM, SeverityEnum.HIGH)
    assert 0.80 <= event.confidence <= 1.0
    assert "Sustained speech detected" in event.description
    assert event.metadata["duration_sec"] >= 3.0


# =====================================================================
# 6. Debouncing of Continuous Speech Tests
# =====================================================================

def test_speech_debouncing_continuous_stream(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test debouncing: Continuous 6 seconds of speech produces exactly ONE event, not one per chunk."""
    session_id = "test_debouncing_sess"
    audio_analyzer.reset_session(session_id)
    sr, speech = wavfile.read(sample_paths["speech"])

    sustained_events = []
    # Process 6 consecutive seconds of speech
    for sec in range(6):
        chunk = speech[sec * sr : (sec + 1) * sr]
        res = audio_analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=chunk,
            timestamp=float(sec),
        )
        for ev in res.events:
            if ev.type == "SUSTAINED_SPEECH":
                sustained_events.append((sec, ev))

    # Exactly one alert should be produced for the speech episode
    assert len(sustained_events) == 1
    # Alert emitted at the 3-second mark (sec == 2)
    assert sustained_events[0][0] == 2


# =====================================================================
# 7. Sudden Loud Noise Detection Tests
# =====================================================================

def test_sudden_loud_noise_trigger(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test detection of sudden loud non-speech transient burst."""
    session_id = "test_loud_noise_sess"
    audio_analyzer.reset_session(session_id)

    sr, loud_audio = wavfile.read(sample_paths["loud_noise"])
    total_sec = len(loud_audio) // sr

    loud_events = []
    for sec in range(total_sec):
        chunk = loud_audio[sec * sr : (sec + 1) * sr]
        res = audio_analyzer.analyze_chunk(
            session_id=session_id,
            audio_chunk=chunk,
            timestamp=float(sec),
        )
        for ev in res.events:
            if ev.type == "SUDDEN_LOUD_NOISE":
                loud_events.append(ev)

    assert len(loud_events) >= 1
    event = loud_events[0]
    assert event.modality == ModalityEnum.AUDIO
    assert event.type == "SUDDEN_LOUD_NOISE"
    assert event.metadata["noise_ratio"] >= 3.0


# =====================================================================
# 8. Cross-Modal: Speech Without Lip Movement Tests
# =====================================================================

def test_speech_without_lip_movement(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test anomaly: candidate face visible but lips motionless while speech is detected."""
    session_id = "test_motionless_lips"
    audio_analyzer.reset_session(session_id)
    sr, speech = wavfile.read(sample_paths["speech"])

    # Chunk with speech, candidate face visible, but lip_motion is 0.5px (< threshold 2.0px)
    chunk = speech[:sr]
    res = audio_analyzer.analyze_chunk(
        session_id=session_id,
        audio_chunk=chunk,
        timestamp=1.0,
        is_face_visible=True,
        lip_motion=0.5,
    )

    lip_events = [e for e in res.events if e.type == "SPEECH_WITHOUT_LIP_MOVEMENT"]
    assert len(lip_events) == 1
    ev = lip_events[0]
    assert ev.severity == SeverityEnum.HIGH
    assert ev.type == "SPEECH_WITHOUT_LIP_MOVEMENT"
    assert "motionless lips" in ev.description


def test_speech_with_active_lips_no_alert(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test normal candidate speech: active lip movement (>2.0px) does NOT trigger motionless alert."""
    session_id = "test_active_lips"
    audio_analyzer.reset_session(session_id)
    sr, speech = wavfile.read(sample_paths["speech"])

    # Chunk with speech, candidate face visible, active lip_motion = 8.5px
    chunk = speech[:sr]
    res = audio_analyzer.analyze_chunk(
        session_id=session_id,
        audio_chunk=chunk,
        timestamp=1.0,
        is_face_visible=True,
        lip_motion=8.5,
    )

    lip_events = [e for e in res.events if e.type == "SPEECH_WITHOUT_LIP_MOVEMENT"]
    assert len(lip_events) == 0


# =====================================================================
# 9. Concurrent Multi-Session State Isolation Tests
# =====================================================================

def test_concurrent_sessions_isolation(audio_analyzer: AudioAnalyzer, sample_paths: dict):
    """Test that concurrent sessions maintain completely independent states and baselines."""
    session_a = "session_quiet_room"
    session_b = "session_noisy_room"
    audio_analyzer.reset_session(session_a)
    audio_analyzer.reset_session(session_b)

    # Session A: Quiet baseline (RMS ≈ 0.005)
    chunk_quiet = np.random.normal(0, 0.005, 16000).astype(np.float32)
    # Session B: Noisy baseline (RMS ≈ 0.04)
    chunk_noisy = np.random.normal(0, 0.04, 16000).astype(np.float32)

    for i in range(10):
        audio_analyzer.analyze_chunk(session_a, chunk_quiet, timestamp=float(i))
        audio_analyzer.analyze_chunk(session_b, chunk_noisy, timestamp=float(i))

    state_a = audio_analyzer.get_session_state(session_a)
    state_b = audio_analyzer.get_session_state(session_b)

    assert state_a.baseline_rms < state_b.baseline_rms
    assert np.isclose(state_a.baseline_rms, 0.005, atol=0.003)
    assert np.isclose(state_b.baseline_rms, 0.04, atol=0.015)
