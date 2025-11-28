'use client';

import { useEffect, useState, useRef } from 'react';
import { apiClient } from '@/utils/api';

// Cleans ANSI escape codes (colors) if backend sends raw terminal output
const clean = (text: string): string => {
  // eslint-disable-next-line no-control-regex
  return text.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, '');
};

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  const didInitialLoadRef = useRef(false);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  const refresh = () => {
    let loadStatus: 'loading' | 'refreshing' = 'loading';
    if (didInitialLoadRef.current) {
      loadStatus = 'refreshing';
    }
    setStatus(loadStatus);

    // Calls Modal: GET /api/jobs/{jobID}/log
    apiClient
      .get(`/api/jobs/${jobID}/log`)
      .then(res => res.data)
      .then(data => {
        if (data.log) {
          setLog(clean(data.log));
        }
        setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        // Log file might not exist yet if job just started
        if (error.response?.status !== 404) {
             console.error('Error fetching log:', error);
             setStatus('error');
        }
      });
  };

  useEffect(() => {
    if(!jobID) return;
    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID]);

  return { log, setLog, status, refresh };
}