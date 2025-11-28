import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(false);

// 1. Get the Modal URL from environment variables
// This ensures all API calls go to your remote Modal backend
const baseURL = process.env.NEXT_PUBLIC_MODAL_API_URL || '';

export const apiClient = axios.create({
  baseURL: baseURL,
});

// 2. Add a request interceptor to add token from localStorage
apiClient.interceptors.request.use(config => {
  const token = localStorage.getItem('AI_TOOLKIT_AUTH');

  // Pass the token if it exists.
  // For Cloudflare Access, the browser sends cookies automatically,
  // but we keep this for the 'password' auth flow in your Python script.
  if (token) {
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

// 3. Add a response interceptor to handle 401 errors
apiClient.interceptors.response.use(
  response => response, // Return successful responses as-is
  error => {
    // Check if the error is a 401 Unauthorized
    if (error.response && error.response.status === 401) {
      // Clear the auth token from localStorage
      localStorage.removeItem('AI_TOOLKIT_AUTH');
      isAuthorizedState.set(false);
    }

    // Reject the promise with the error so calling code can still catch it
    return Promise.reject(error);
  },
);