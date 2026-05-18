'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { apiClient } from '@/utils/api';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const statusRef = useRef<'idle' | 'loading' | 'success' | 'error'>('idle');
  const requestController = useRef<AbortController | null>(null);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  useEffect(() => () => requestController.current?.abort(), []);

  const refreshSampleImages = useCallback(() => {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;

    if (statusRef.current === 'idle') setStatus('loading');

    // Calls Modal: GET /api/jobs/{jobID}/samples
    apiClient
      .get(`/api/jobs/${jobID}/samples`, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        const nextSamples = Array.isArray(data.samples) ? data.samples : [];
        setSampleImages((prev) => {
          if (prev.length === nextSamples.length && prev.every((item, index) => item === nextSamples[index])) {
            return prev;
          }
          return nextSamples;
        });

        if (statusRef.current !== 'success') setStatus('success');
      })
      .catch(error => {
         if (axios.isAxiosError(error)) {
            if (error.code === 'ECONNABORTED' || error.code === 'ERR_CANCELED') {
              return;
            }
         }
         if (error.response?.status !== 404) {
            console.error('Error fetching samples:', error);
            if (statusRef.current !== 'error') setStatus('error');
         } else {
            setSampleImages((prev) => (prev.length === 0 ? prev : []));
            if (statusRef.current !== 'success') setStatus('success');
         }
      });
  }, [jobID]);

  useEffect(() => {
    if(!jobID) return;
    refreshSampleImages();

    if (reloadInterval) {
      const interval = setInterval(refreshSampleImages, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refreshSampleImages]);

  return { sampleImages, setSampleImages, status, refreshSampleImages };
}
