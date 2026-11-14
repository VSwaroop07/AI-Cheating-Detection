/**
 * Axios API client configuration and typed endpoint handlers.
 */

import axios, { AxiosInstance } from "axios";

const baseURL: string = import.meta.env.VITE_API_URL || "http://localhost:8000";

export const apiClient: AxiosInstance = axios.create({
  baseURL,
  timeout: 15000,
  headers: {
    "Content-Type": "application/json",
  },
});

export interface HealthResponse {
  status: string;
  service: string;
  environment: string;
  database: string;
  timestamp: string;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: "candidate" | "proctor" | "admin";
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  user: UserProfile;
}

export interface FaceRegistrationPayload {
  image_base64: string;
  consent_monitoring: boolean;
  consent_store_raw_image: boolean;
}

export interface FaceRegistrationResponse {
  success: boolean;
  candidate_id: string;
  message: string;
  faces_detected: number;
  embedding_stored: boolean;
  raw_image_stored: boolean;
  raw_image_path: string | null;
  registered_at: string;
}

export interface FaceVerificationPayload {
  image_base64: string;
  threshold?: number;
}

export interface FaceVerificationResponse {
  session_id: string;
  candidate_id: string;
  status: "Verified" | "Not Verified";
  verified: boolean;
  confidence: number;
  distance: number;
  threshold: number;
  timestamp: string;
}

/**
 * Fetch health status from the backend API.
 */
export async function getHealthStatus(): Promise<HealthResponse> {
  const response = await apiClient.get<HealthResponse>("/health");
  return response.data;
}

/**
 * Register candidate face biometric template.
 */
export async function registerCandidateFace(
  payload: FaceRegistrationPayload,
  token?: string,
): Promise<FaceRegistrationResponse> {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.post<FaceRegistrationResponse>(
    "/candidates/register-face",
    payload,
    { headers },
  );
  return response.data;
}

/**
 * Verify candidate live webcam frame against session registered template.
 */
export async function verifyCandidateFace(
  sessionId: string,
  payload: FaceVerificationPayload,
  token?: string,
): Promise<FaceVerificationResponse> {
  const headers = token ? { Authorization: `Bearer ${token}` } : {};
  const response = await apiClient.post<FaceVerificationResponse>(
    `/sessions/${sessionId}/verify`,
    payload,
    { headers },
  );
  return response.data;
}

export default apiClient;
