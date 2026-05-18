import axios from 'axios';
import { createGlobalState } from 'react-global-hooks';

export const isAuthorizedState = createGlobalState(true);

// 1. Get the Modal URL from environment variables
const baseURL = process.env.NEXT_PUBLIC_MODAL_API_URL || '';

export const apiBaseURL = baseURL.replace(/\/$/, '');

export const buildApiFileURL = (filePath: string) => {
  const normalizedPath = filePath.replace(/\\/g, '/').replace(/^\/+/, '');
  const encodedPath = normalizedPath
    .split('/')
    .map(segment => encodeURIComponent(segment))
    .join('/');
  return `${apiBaseURL}/api/files/${encodedPath}`;
};

export const apiClient = axios.create({
  baseURL: baseURL,
});

// 2. Interceptors cleaned up to remove auth token logic
apiClient.interceptors.request.use(config => {
  return config;
});

apiClient.interceptors.response.use(
  response => response,
  error => {
    return Promise.reject(error);
  },
);
