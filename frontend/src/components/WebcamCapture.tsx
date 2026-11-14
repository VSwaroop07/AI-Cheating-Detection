/**
 * WebcamCapture component: Accesses webcam stream, captures frames, and provides image preview.
 */

import React, { useRef, useState, useEffect, useCallback } from "react";

interface WebcamCaptureProps {
  onCapture: (imageBase64: string) => void;
  onClear?: () => void;
  title?: string;
}

export const WebcamCapture: React.FC<WebcamCaptureProps> = ({
  onCapture,
  onClear,
  title = "Live Webcam Feed",
}) => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [isCameraActive, setIsCameraActive] = useState<boolean>(false);

  const startCamera = useCallback(async () => {
    setCameraError(null);
    try {
      if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
        const mediaStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
          audio: false,
        });
        setStream(mediaStream);
        if (videoRef.current) {
          videoRef.current.srcObject = mediaStream;
          await videoRef.current.play();
        }
        setIsCameraActive(true);
      } else {
        setCameraError("Webcam access is not supported by your browser environment.");
      }
    } catch (err: unknown) {
      if (err instanceof Error) {
        setCameraError(`Camera Error: ${err.message}. (You can also upload a sample photo below).`);
      } else {
        setCameraError("Unable to access camera device.");
      }
      setIsCameraActive(false);
    }
  }, []);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      setStream(null);
    }
    setIsCameraActive(false);
  }, [stream]);

  useEffect(() => {
    startCamera();
    return () => {
      stopCamera();
    };
  }, [startCamera, stopCamera]);

  const handleCapture = () => {
    if (!videoRef.current || !canvasRef.current) return;
    const video = videoRef.current;
    const canvas = canvasRef.current;
    const width = video.videoWidth || 640;
    const height = video.videoHeight || 480;

    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    ctx.drawImage(video, 0, 0, width, height);
    const dataUrl = canvas.toDataURL("image/jpeg", 0.92);
    setCapturedImage(dataUrl);
    onCapture(dataUrl);
  };

  const handleRetake = () => {
    setCapturedImage(null);
    if (onClear) onClear();
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      setCapturedImage(result);
      onCapture(result);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="webcam-capture-container card">
      <div className="webcam-header">
        <h4>{title}</h4>
        <span className={`camera-indicator ${isCameraActive ? "active" : "offline"}`}>
          {isCameraActive ? "● Camera Live" : "○ Camera Offline"}
        </span>
      </div>

      {cameraError && (
        <div className="camera-error-banner">
          <p>{cameraError}</p>
          <div className="camera-fallback-actions">
            <button type="button" className="btn btn-sm btn-outline" onClick={startCamera}>
              Retry Camera
            </button>
            <label className="btn btn-sm btn-secondary file-upload-label">
              Upload Photo Instead
              <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: "none" }} />
            </label>
          </div>
        </div>
      )}

      <div className="webcam-viewport">
        {!capturedImage ? (
          <div className="video-wrapper">
            <video ref={videoRef} playsInline muted className="webcam-video" />
            <div className="face-guide-overlay">
              <div className="face-guide-oval"></div>
              <span>Align face inside the oval</span>
            </div>
          </div>
        ) : (
          <div className="preview-wrapper">
            <img src={capturedImage} alt="Captured Preview" className="captured-preview" />
            <span className="preview-badge">Snapshot Captured</span>
          </div>
        )}
      </div>

      <canvas ref={canvasRef} style={{ display: "none" }} />

      <div className="webcam-controls">
        {!capturedImage ? (
          <button
            type="button"
            className="btn btn-primary btn-capture"
            onClick={handleCapture}
            disabled={!isCameraActive}
          >
            📸 Capture Photo
          </button>
        ) : (
          <button type="button" className="btn btn-secondary" onClick={handleRetake}>
            🔄 Retake Photo
          </button>
        )}
      </div>
    </div>
  );
};

export default WebcamCapture;

