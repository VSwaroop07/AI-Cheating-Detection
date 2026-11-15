/**
 * Custom React hook managing candidate capture layer:
 * - Webcam, microphone, and screen share permissions & streams
 * - Video framing (approx 2-3 FPS resized to 640px)
 * - 1-second PCM audio chunking
 * - Screen and browser event monitoring (tab switch, window blur)
 * - WebSocket streaming with auto-reconnect, exponential backoff, and offline event buffering.
 */

import { useState, useEffect, useRef, useCallback } from "react";

export type PermissionStatus = "idle" | "requesting" | "granted" | "denied" | "error";
export type ConnectionState = "disconnected" | "connecting" | "connected" | "reconnecting";

export interface UseCaptureOptions {
  sessionId: string;
  token?: string;
  fps?: number; // target FPS, defaults to ~2.5
}

export interface UseCaptureReturn {
  webcamRef: React.RefObject<HTMLVideoElement>;
  screenRef: React.RefObject<HTMLVideoElement>;
  webcamStream: MediaStream | null;
  screenStream: MediaStream | null;
  permissions: {
    camera: PermissionStatus;
    microphone: PermissionStatus;
    screen: PermissionStatus;
  };
  connectionState: ConnectionState;
  bufferedEventsCount: number;
  framesSent: number;
  audioChunksSent: number;
  lastError: string | null;
  requestAllPermissions: () => Promise<boolean>;
  startCapture: () => void;
  stopCapture: () => void;
  sendTelemetryEvent: (eventType: string, details?: Record<string, unknown>) => void;
  sendScreenEvent: (payload: Record<string, unknown>) => void;
}

export function useCapture({
  sessionId,
  token,
  fps = 2.5,
}: UseCaptureOptions): UseCaptureReturn {
  // Video and screen element references
  const webcamRef = useRef<HTMLVideoElement>(null);
  const screenRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(document.createElement("canvas"));

  // Streams
  const [webcamStream, setWebcamStream] = useState<MediaStream | null>(null);
  const [screenStream, setScreenStream] = useState<MediaStream | null>(null);

  // Permission states
  const [cameraPermission, setCameraPermission] = useState<PermissionStatus>("idle");
  const [micPermission, setMicPermission] = useState<PermissionStatus>("idle");
  const [screenPermission, setScreenPermission] = useState<PermissionStatus>("idle");

  // WebSocket & Telemetry states
  const [connectionState, setConnectionState] = useState<ConnectionState>("disconnected");
  const [bufferedEventsCount, setBufferedEventsCount] = useState<number>(0);
  const [framesSent, setFramesSent] = useState<number>(0);
  const [audioChunksSent, setAudioChunksSent] = useState<number>(0);
  const [lastError, setLastError] = useState<string | null>(null);

  // Internal refs for timers and WS
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectAttemptRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<number | null>(null);
  const eventBufferRef = useRef<any[]>([]);
  const isCapturingRef = useRef<boolean>(false);
  const frameIntervalRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const frameIndexRef = useRef<number>(0);

  // Derive WebSocket URL from window location or env
  const getWebSocketUrl = useCallback(() => {
    const rawApiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
    const wsBase = rawApiUrl.replace(/^http/, "ws");
    const authToken = token || localStorage.getItem("proctor_token") || "";
    return `${wsBase}/ws/session/${sessionId}?token=${encodeURIComponent(authToken)}`;
  }, [sessionId, token]);

  // Transmit or buffer message
  const sendMessage = useCallback((envelope: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(envelope));
    } else {
      // Buffer outgoing non-video events while offline
      if (envelope.type === "screen" || envelope.type === "event") {
        eventBufferRef.current.push(envelope);
        setBufferedEventsCount(eventBufferRef.current.length);
      }
    }
  }, []);

  // Flush offline buffered events upon reconnection
  const flushEventBuffer = useCallback(() => {
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
    while (eventBufferRef.current.length > 0) {
      const evt = eventBufferRef.current.shift();
      wsRef.current.send(JSON.stringify(evt));
    }
    setBufferedEventsCount(0);
  }, []);

  // Connect WebSocket with exponential backoff
  const connectWebSocket = useCallback(() => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) return;

    setConnectionState((prev) => (prev === "disconnected" ? "connecting" : "reconnecting"));
    const url = getWebSocketUrl();

    try {
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionState("connected");
        reconnectAttemptRef.current = 0;
        setLastError(null);
        flushEventBuffer();
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          // Handshake or ack receipt
        } catch (e) {
          // ignore
        }
      };

      ws.onclose = (event) => {
        if (!isCapturingRef.current) {
          setConnectionState("disconnected");
          return;
        }

        setConnectionState("reconnecting");
        // Exponential backoff: 1s, 2s, 4s, 8s, up to 16s + jitter
        const delay = Math.min(1000 * Math.pow(2, reconnectAttemptRef.current), 16000) + Math.random() * 500;
        reconnectAttemptRef.current += 1;

        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = window.setTimeout(() => {
          connectWebSocket();
        }, delay);
      };

      ws.onerror = () => {
        setLastError("WebSocket connection encountered an error.");
      };
    } catch (err: unknown) {
      setConnectionState("disconnected");
      if (err instanceof Error) {
        setLastError(err.message);
      }
    }
  }, [getWebSocketUrl, flushEventBuffer]);

  // Request camera and microphone
  const requestMediaPermissions = useCallback(async (): Promise<boolean> => {
    setCameraPermission("requesting");
    setMicPermission("requesting");
    try {
      const userStream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: { channelCount: 1, sampleRate: 16000, echoCancellation: true },
      });

      setWebcamStream(userStream);
      setCameraPermission("granted");
      setMicPermission("granted");

      if (webcamRef.current) {
        webcamRef.current.srcObject = userStream;
        await webcamRef.current.play().catch(() => {});
      }
      return true;
    } catch (err: unknown) {
      setCameraPermission("denied");
      setMicPermission("denied");
      setLastError("Webcam or Microphone permission denied by user.");
      return false;
    }
  }, []);

  // Request screen-share
  const requestScreenPermission = useCallback(async (): Promise<boolean> => {
    setScreenPermission("requesting");
    try {
      const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: { cursor: "always" } as any,
        audio: false,
      });

      setScreenStream(displayStream);
      setScreenPermission("granted");

      if (screenRef.current) {
        screenRef.current.srcObject = displayStream;
        await screenRef.current.play().catch(() => {});
      }

      // Track screen share termination event
      const track = displayStream.getVideoTracks()[0];
      if (track) {
        track.addEventListener("ended", () => {
          setScreenPermission("denied");
          sendMessage({
            type: "screen",
            session_id: sessionId,
            payload: {
              event_type: "SCREEN_SHARE_ENDED",
              details: { reason: "User stopped sharing screen" },
              timestamp: new Date().toISOString(),
            },
          });
        });
      }

      return true;
    } catch (err: unknown) {
      setScreenPermission("denied");
      setLastError("Screen sharing permission denied by user.");
      return false;
    }
  }, [sessionId, sendMessage]);

  // Request all permissions in sequence
  const requestAllPermissions = useCallback(async (): Promise<boolean> => {
    const mediaOk = await requestMediaPermissions();
    if (!mediaOk) return false;
    const screenOk = await requestScreenPermission();
    return screenOk;
  }, [requestMediaPermissions, requestScreenPermission]);

  // Video Frame Sampling Loop (approx 2-3 FPS, 640px wide)
  const startVideoSampling = useCallback(() => {
    const intervalMs = Math.round(1000 / fps);
    const canvas = canvasRef.current;

    frameIntervalRef.current = window.setInterval(() => {
      const video = webcamRef.current;
      if (!video || video.readyState < 2) return;

      const vWidth = video.videoWidth || 640;
      const vHeight = video.videoHeight || 480;
      const targetWidth = 640;
      const targetHeight = Math.round((vHeight / vWidth) * targetWidth);

      canvas.width = targetWidth;
      canvas.height = targetHeight;
      const ctx = canvas.getContext("2d");
      if (!ctx) return;

      ctx.drawImage(video, 0, 0, targetWidth, targetHeight);
      const dataUrl = canvas.toDataURL("image/jpeg", 0.72);

      frameIndexRef.current += 1;
      setFramesSent((prev) => prev + 1);

      sendMessage({
        type: "video",
        session_id: sessionId,
        payload: {
          image_base64: dataUrl,
          width: targetWidth,
          height: targetHeight,
          frame_index: frameIndexRef.current,
          timestamp: new Date().toISOString(),
        },
      });
    }, intervalMs);
  }, [fps, sessionId, sendMessage]);

  // Audio 1-second PCM Chunking Loop
  const startAudioSampling = useCallback((stream: MediaStream) => {
    try {
      const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)({
        sampleRate: 16000,
      });
      audioContextRef.current = audioContext;

      const source = audioContext.createMediaStreamSource(stream);
      // Buffer size: 4096 samples
      const processor = audioContext.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      let pcmBuffer: Float32Array = new Float32Array(0);
      const targetSamplesPerChunk = 16000; // 1 second at 16kHz

      processor.onaudioprocess = (e) => {
        if (!isCapturingRef.current) return;
        const inputData = e.inputBuffer.getChannelData(0);

        // Append incoming samples
        const newBuf = new Float32Array(pcmBuffer.length + inputData.length);
        newBuf.set(pcmBuffer);
        newBuf.set(inputData, pcmBuffer.length);
        pcmBuffer = newBuf;

        if (pcmBuffer.length >= targetSamplesPerChunk) {
          const chunk = pcmBuffer.slice(0, targetSamplesPerChunk);
          pcmBuffer = pcmBuffer.slice(targetSamplesPerChunk);

          // Convert Float32 [-1.0, 1.0] to 16-bit PCM Int16 bytes
          const int16Array = new Int16Array(chunk.length);
          for (let i = 0; i < chunk.length; i++) {
            const s = Math.max(-1, Math.min(1, chunk[i]));
            int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
          }

          // Base64 encode PCM binary
          let binary = "";
          const bytes = new Uint8Array(int16Array.buffer);
          for (let i = 0; i < bytes.byteLength; i++) {
            binary += String.fromCharCode(bytes[i]);
          }
          const base64Pcm = window.btoa(binary);

          setAudioChunksSent((prev) => prev + 1);
          sendMessage({
            type: "audio",
            session_id: sessionId,
            payload: {
              audio_base64: base64Pcm,
              format: "pcm_16bit",
              sample_rate: 16000,
              channels: 1,
              duration_sec: 1.0,
              timestamp: new Date().toISOString(),
            },
          });
        }
      };

      source.connect(processor);
      processor.connect(audioContext.destination);
    } catch (err: unknown) {
      console.warn("AudioContext initialization warning:", err);
    }
  }, [sessionId, sendMessage]);

  // Screen and browser event listeners (tab switch, window blur, fullscreen)
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (!isCapturingRef.current) return;
      sendMessage({
        type: "screen",
        session_id: sessionId,
        payload: {
          event_type: document.hidden ? "TAB_SWITCH" : "TAB_FOCUSED",
          details: { visibilityState: document.visibilityState, isHidden: document.hidden },
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleWindowBlur = () => {
      if (!isCapturingRef.current) return;
      sendMessage({
        type: "screen",
        session_id: sessionId,
        payload: {
          event_type: "WINDOW_BLUR",
          details: { message: "Window lost focus" },
          timestamp: new Date().toISOString(),
        },
      });
    };

    const handleWindowFocus = () => {
      if (!isCapturingRef.current) return;
      sendMessage({
        type: "screen",
        session_id: sessionId,
        payload: {
          event_type: "WINDOW_FOCUS",
          details: { message: "Window regained focus" },
          timestamp: new Date().toISOString(),
        },
      });
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    window.addEventListener("blur", handleWindowBlur);
    window.addEventListener("focus", handleWindowFocus);

    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
      window.removeEventListener("blur", handleWindowBlur);
      window.removeEventListener("focus", handleWindowFocus);
    };
  }, [sessionId, sendMessage]);

  // Start Capture Routine
  const startCapture = useCallback(() => {
    isCapturingRef.current = true;
    connectWebSocket();
    startVideoSampling();
    if (webcamStream) {
      startAudioSampling(webcamStream);
    }
  }, [connectWebSocket, startVideoSampling, startAudioSampling, webcamStream]);

  // Stop Capture Routine
  const stopCapture = useCallback(() => {
    isCapturingRef.current = false;
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    setConnectionState("disconnected");
  }, []);

  // Send arbitrary custom telemetry event
  const sendTelemetryEvent = useCallback((eventType: string, details: Record<string, unknown> = {}) => {
    sendMessage({
      type: "event",
      session_id: sessionId,
      payload: {
        event_type: eventType,
        details,
        timestamp: new Date().toISOString(),
      },
    });
  }, [sessionId, sendMessage]);

  // Send screen monitoring violation event
  const sendScreenEvent = useCallback((eventPayload: Record<string, unknown>) => {
    sendMessage({
      type: "screen",
      session_id: sessionId,
      payload: {
        ...eventPayload,
        timestamp: eventPayload.timestamp || new Date().toISOString(),
      },
    });
  }, [sessionId, sendMessage]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopCapture();
      if (webcamStream) webcamStream.getTracks().forEach((t) => t.stop());
      if (screenStream) screenStream.getTracks().forEach((t) => t.stop());
    };
  }, [stopCapture, webcamStream, screenStream]);

  return {
    webcamRef,
    screenRef,
    webcamStream,
    screenStream,
    permissions: {
      camera: cameraPermission,
      microphone: micPermission,
      screen: screenPermission,
    },
    connectionState,
    bufferedEventsCount,
    framesSent,
    audioChunksSent,
    lastError,
    requestAllPermissions,
    startCapture,
    stopCapture,
    sendTelemetryEvent,
    sendScreenEvent,
  };
}

export default useCapture;

