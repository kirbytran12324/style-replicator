'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useSampleImages(jobID: string, reloadInterval: null | number = null) {
  const [sampleImages, setSampleImages] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshSampleImages = () => {
    setStatus('loading');

    // Calls Modal: GET /api/jobs/{jobID}/samples
    apiClient
      .get(`/api/jobs/${jobID}/samples`)
      .then(res => res.data)
      .then(data => {
        if (data.samples) {
          // Ensure paths are full URLs or relative to API proxy
          // If the backend returns paths like "/api/files/...", we are good.
          setSampleImages(data.samples);
        }
        setStatus('success');
      })
      .catch(error => {
         if (error.response?.status !== 404) {
            console.error('Error fetching samples:', error);
            setStatus('error');
         }
      });
  };

  useEffect(() => {
    if(!jobID) return;
    refreshSampleImages();

    if (reloadInterval) {
      const interval = setInterval(refreshSampleImages, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID]);

  return { sampleImages, setSampleImages, status, refreshSampleImages };
}