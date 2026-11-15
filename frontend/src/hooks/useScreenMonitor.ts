/**
 * Custom React hook for screen, browser, and environment activity monitoring:
 * - Page Visibility API (document.visibilitychange) for tab switching
 * - Window blur and focus for application or window switching
 * - Fullscreen enforcement and exit detection
 * - Prohibited clipboard actions (copy, paste, cut)
 * - Mouse context menu inspection (right-click)
 * - Screen-share termination by candidate (MediaStreamTrack.onended)
 * - Evidence screenshot capture of the shared screen on every violation
 * - Precise away duration tracking for tab switches and blur intervals
 */

import { useState, useEffect, useRef, useCallback } from "react";

export type ScreenEventType =
  | "TAB_SWITCH"
  | "WINDOW_BLUR"
  | "WINDOW_FOCUS"
  | "FULLSCREEN_EXIT"
  | "FULLSCREEN_ENTER"
  | "CLIPBOARD_COPY"
  | "CLIPBOARD_PASTE"
  | "CLIPBOARD_CUT"
  | "RIGHT_CLICK"
  | "SCREEN_SHARE_STOPPED";

export interface ScreenViolation {
  event_type: ScreenEventType;
  timestamp: string; // ISO 8601 UTC string
  away_duration_sec: number;
  screenshot_base64: string | null;
  details?: Record<string, unknown>;
}

export interface UseScreenMonitorOptions {
  sessionId: string;
  enabled?: boolean;
  screenStream?: MediaStream | null;
  screenVideoRef?: React.RefObject<HTMLVideoElement>;
  onViolation?: (violation: ScreenViolation) => void;
  onScreenShareEnded?: () => void;
  forceFullscreen?: boolean;
}

export interface UseScreenMonitorReturn {
  isFullscreen: boolean;
  violationsCount: {
    tabSwitches: number;
    windowBlurs: number;
    fullscreenExits: number;
    clipboardEvents: number;
    rightClicks: number;
    total: number;
  };
  lastViolation: ScreenViolation | null;
  requestFullscreen: () => Promise<boolean>;
  exitFullscreen: () => Promise<void>;
  captureScreenshot: () => string | null;
}

export function useScreenMonitor({
  sessionId,
  enabled = true,
  screenStream = null,
  screenVideoRef,
  onViolation,
  onScreenShareEnded,
  forceFullscreen = true,
}: UseScreenMonitorOptions): UseScreenMonitorReturn {
  const [isFullscreen, setIsFullscreen] = useState<boolean>(
    typeof document !== "undefined" && Boolean(document.fullscreenElement)
  );

  const [violationsCount, setViolationsCount] = useState({
    tabSwitches: 0,
    windowBlurs: 0,
    fullscreenExits: 0,
    clipboardEvents: 0,
    rightClicks: 0,
    total: 0,
  });

  const [lastViolation, setLastViolation] = useState<ScreenViolation | null>(null);

  // Hidden offscreen canvas for capturing evidence screenshots
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  if (!canvasRef.current && typeof document !== "undefined") {
    canvasRef.current = document.createElement("canvas");
  }

  // Tracking timestamps for calculating away duration
  const tabHiddenTimeRef = useRef<number | null>(null);
  const blurTimeRef = useRef<number | null>(null);
  const fullscreenExitTimeRef = useRef<number | null>(null);

  // Internal video element used when no screenVideoRef is supplied
  const internalVideoRef = useRef<HTMLVideoElement | null>(null);

  // Capture screenshot of the shared screen
  const captureScreenshot = useCallback((): string | null => {
    try {
      const canvas = canvasRef.current;
      if (!canvas) return null;

      // Priority 1: Capture directly from active video element ref
      const videoEl = screenVideoRef?.current || internalVideoRef.current;

      if (
        videoEl &&
        videoEl.readyState >= 2 &&
        videoEl.videoWidth > 0 &&
        videoEl.videoHeight > 0
      ) {
        canvas.width = videoEl.videoWidth;
        canvas.height = videoEl.videoHeight;
        const ctx = canvas.getContext("2d");
        if (ctx) {
          ctx.drawImage(videoEl, 0, 0, canvas.width, canvas.height);
          return canvas.toDataURL("image/jpeg", 0.65);
        }
      }

      // Priority 2: If stream has active tracks but no video element ready, attach to internal element
      if (screenStream && screenStream.active && screenStream.getVideoTracks().length > 0) {
        if (!internalVideoRef.current) {
          const v = document.createElement("video");
          v.muted = true;
          v.playsInline = true;
          v.srcObject = screenStream;
          v.play().catch(() => {});
          internalVideoRef.current = v;
        } else if (internalVideoRef.current.srcObject !== screenStream) {
          internalVideoRef.current.srcObject = screenStream;
          internalVideoRef.current.play().catch(() => {});
        }

        const v = internalVideoRef.current;
        if (v && v.videoWidth > 0 && v.videoHeight > 0) {
          canvas.width = v.videoWidth;
          canvas.height = v.videoHeight;
          const ctx = canvas.getContext("2d");
          if (ctx) {
            ctx.drawImage(v, 0, 0, canvas.width, canvas.height);
            return canvas.toDataURL("image/jpeg", 0.65);
          }
        }
      }

      return null;
    } catch (err) {
      console.warn("[ScreenMonitor] Could not capture screenshot evidence:", err);
      return null;
    }
  }, [screenStream, screenVideoRef]);

  // Request entering fullscreen
  const requestFullscreen = useCallback(async (): Promise<boolean> => {
    try {
      const elem = document.documentElement as any;
      if (elem.requestFullscreen) {
        await elem.requestFullscreen();
      } else if (elem.webkitRequestFullscreen) {
        await elem.webkitRequestFullscreen();
      } else if (elem.mozRequestFullScreen) {
        await elem.mozRequestFullScreen();
      } else if (elem.msRequestFullscreen) {
        await elem.msRequestFullscreen();
      }
      setIsFullscreen(true);
      return true;
    } catch (err) {
      console.warn("[ScreenMonitor] Fullscreen request rejected or unsupported:", err);
      return false;
    }
  }, []);

  // Exit fullscreen
  const exitFullscreen = useCallback(async (): Promise<void> => {
    try {
      const doc = document as any;
      if (doc.exitFullscreen) {
        await doc.exitFullscreen();
      } else if (doc.webkitExitFullscreen) {
        await doc.webkitExitFullscreen();
      } else if (doc.mozCancelFullScreen) {
        await doc.mozCancelFullScreen();
      } else if (doc.msExitFullscreen) {
        await doc.msExitFullscreen();
      }
      setIsFullscreen(false);
    } catch (err) {
      console.warn("[ScreenMonitor] Exit fullscreen error:", err);
    }
  }, []);

  // Internal dispatcher for raising a validated violation
  const emitViolation = useCallback(
    (
      eventType: ScreenEventType,
      awayDurationSec: number = 0,
      details: Record<string, unknown> = {}
    ) => {
      const screenshot = captureScreenshot();
      const violation: ScreenViolation = {
        event_type: eventType,
        timestamp: new Date().toISOString(),
        away_duration_sec: Math.max(0, Math.round(awayDurationSec * 10) / 10),
        screenshot_base64: screenshot,
        details: {
          sessionId,
          ...details,
        },
      };

      // Update counters
      setViolationsCount((prev) => {
        const next = { ...prev, total: prev.total + 1 };
        if (eventType === "TAB_SWITCH") next.tabSwitches += 1;
        else if (eventType === "WINDOW_BLUR") next.windowBlurs += 1;
        else if (eventType === "FULLSCREEN_EXIT") next.fullscreenExits += 1;
        else if (
          eventType === "CLIPBOARD_COPY" ||
          eventType === "CLIPBOARD_PASTE" ||
          eventType === "CLIPBOARD_CUT"
        )
          next.clipboardEvents += 1;
        else if (eventType === "RIGHT_CLICK") next.rightClicks += 1;
        return next;
      });

      setLastViolation(violation);
      if (onViolation) {
        onViolation(violation);
      }
    },
    [sessionId, captureScreenshot, onViolation]
  );

  // -------------------------------------------------------------
  // Fullscreen Management & Forced Fullscreen Check
  // -------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    if (forceFullscreen && !document.fullscreenElement) {
      // Attempt entering fullscreen upon activation
      requestFullscreen().catch(() => {});
    }

    const handleFullscreenChange = () => {
      const isCurrentlyFullscreen = Boolean(document.fullscreenElement);
      setIsFullscreen(isCurrentlyFullscreen);

      if (!isCurrentlyFullscreen) {
        // Fullscreen lost
        fullscreenExitTimeRef.current = Date.now();
        emitViolation("FULLSCREEN_EXIT", 0, {
          message: "Candidate exited fullscreen exam environment",
        });
      } else {
        // Regained fullscreen
        let awaySec = 0;
        if (fullscreenExitTimeRef.current) {
          awaySec = (Date.now() - fullscreenExitTimeRef.current) / 1000;
          fullscreenExitTimeRef.current = null;
        }
        emitViolation("FULLSCREEN_ENTER", awaySec, {
          message: "Candidate returned to fullscreen exam mode",
        });
      }
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    document.addEventListener("webkitfullscreenchange", handleFullscreenChange);
    document.addEventListener("mozfullscreenchange", handleFullscreenChange);
    document.addEventListener("MSFullscreenChange", handleFullscreenChange);

    return () => {
      document.removeEventListener("fullscreenchange", handleFullscreenChange);
      document.removeEventListener("webkitfullscreenchange", handleFullscreenChange);
      document.removeEventListener("mozfullscreenchange", handleFullscreenChange);
      document.removeEventListener("MSFullscreenChange", handleFullscreenChange);
    };
  }, [enabled, forceFullscreen, requestFullscreen, emitViolation]);

  // -------------------------------------------------------------
  // Page Visibility API (Tab Switch Monitoring)
  // -------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    const handleVisibilityChange = () => {
      if (document.visibilityState === "hidden") {
        tabHiddenTimeRef.current = Date.now();
        emitViolation("TAB_SWITCH", 0, {
          state: "hidden",
          message: "Candidate switched browser tabs or minimized window",
        });
      } else if (document.visibilityState === "visible") {
        let awaySec = 0;
        if (tabHiddenTimeRef.current) {
          awaySec = (Date.now() - tabHiddenTimeRef.current) / 1000;
          tabHiddenTimeRef.current = null;
        }
        emitViolation("TAB_SWITCH", awaySec, {
          state: "visible",
          message: `Candidate returned to exam tab after ${awaySec.toFixed(1)}s`,
        });
      }
    };

    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [enabled, emitViolation]);

  // -------------------------------------------------------------
  // Window Focus / Blur (Application or Monitor Switching)
  // -------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    const handleWindowBlur = () => {
      blurTimeRef.current = Date.now();
      emitViolation("WINDOW_BLUR", 0, {
        message: "Exam window lost focus (switched application or second monitor)",
      });
    };

    const handleWindowFocus = () => {
      let awaySec = 0;
      if (blurTimeRef.current) {
        awaySec = (Date.now() - blurTimeRef.current) / 1000;
        blurTimeRef.current = null;
      }
      emitViolation("WINDOW_FOCUS", awaySec, {
        message: `Exam window regained focus after ${awaySec.toFixed(1)}s`,
      });
    };

    window.addEventListener("blur", handleWindowBlur);
    window.addEventListener("focus", handleWindowFocus);

    return () => {
      window.removeEventListener("blur", handleWindowBlur);
      window.removeEventListener("focus", handleWindowFocus);
    };
  }, [enabled, emitViolation]);

  // -------------------------------------------------------------
  // Clipboard (Copy, Paste, Cut) & Context Menu (Right Click)
  // -------------------------------------------------------------
  useEffect(() => {
    if (!enabled) return;

    const handleCopy = (e: ClipboardEvent) => {
      emitViolation("CLIPBOARD_COPY", 0, {
        message: "Prohibited clipboard copy attempt detected",
      });
    };

    const handlePaste = (e: ClipboardEvent) => {
      emitViolation("CLIPBOARD_PASTE", 0, {
        message: "Prohibited clipboard paste attempt detected",
      });
    };

    const handleCut = (e: ClipboardEvent) => {
      emitViolation("CLIPBOARD_CUT", 0, {
        message: "Prohibited clipboard cut attempt detected",
      });
    };

    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault(); // Prevent inspect element / context menu
      emitViolation("RIGHT_CLICK", 0, {
        x: e.clientX,
        y: e.clientY,
        message: "Prohibited right-click context menu attempt blocked",
      });
    };

    document.addEventListener("copy", handleCopy);
    document.addEventListener("paste", handlePaste);
    document.addEventListener("cut", handleCut);
    document.addEventListener("contextmenu", handleContextMenu);

    return () => {
      document.removeEventListener("copy", handleCopy);
      document.removeEventListener("paste", handlePaste);
      document.removeEventListener("cut", handleCut);
      document.removeEventListener("contextmenu", handleContextMenu);
    };
  }, [enabled, emitViolation]);

  // -------------------------------------------------------------
  // Screen Share Termination Monitoring (track.onended)
  // -------------------------------------------------------------
  useEffect(() => {
    if (!enabled || !screenStream) return;

    const videoTracks = screenStream.getVideoTracks();
    if (videoTracks.length === 0) return;

    const handleTrackEnded = () => {
      emitViolation("SCREEN_SHARE_STOPPED", 0, {
        message: "Candidate manually stopped screen sharing stream",
      });
      if (onScreenShareEnded) {
        onScreenShareEnded();
      }
    };

    videoTracks.forEach((track) => {
      track.addEventListener("ended", handleTrackEnded);
    });

    return () => {
      videoTracks.forEach((track) => {
        track.removeEventListener("ended", handleTrackEnded);
      });
    };
  }, [enabled, screenStream, emitViolation, onScreenShareEnded]);

  return {
    isFullscreen,
    violationsCount,
    lastViolation,
    requestFullscreen,
    exitFullscreen,
    captureScreenshot,
  };
}
