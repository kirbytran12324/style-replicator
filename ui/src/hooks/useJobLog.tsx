'use client';

import { useEffect, useState, useRef, useCallback } from 'react';
import axios from 'axios';
import { apiClient } from '@/utils/api';

const clean = (text: string): string => {
  return text.replace(/\x1B\[[0-9;]*[a-zA-Z]/g, '');
};

export default function useJobLog(jobID: string, reloadInterval: null | number = null) {
  const [log, setLog] = useState<string>('');
  const didInitialLoadRef = useRef(false);
  const requestController = useRef<AbortController | null>(null);
  const statusRef = useRef<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');
  const lastLogRef = useRef('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error' | 'refreshing'>('idle');

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const refresh = useCallback(() => {
    if (!didInitialLoadRef.current && statusRef.current !== 'loading') {
      setStatus('loading');
    }

    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;

    apiClient
      .get(`/api/jobs/${jobID}/log`, { signal: controller.signal })
      .then(res => res.data)
      .then(data => {
        const cleanedLog = data.log ? clean(data.log) : '';
        if (lastLogRef.current !== cleanedLog) {
          lastLogRef.current = cleanedLog;
          setLog(cleanedLog);
        }
        if (statusRef.current !== 'success') setStatus('success');
        didInitialLoadRef.current = true;
      })
      .catch(error => {
        if (axios.isAxiosError(error)) {
          if (error.code === 'ECONNABORTED' || error.code === 'ERR_CANCELED') {
            return;
          }
        }
        if (error.response?.status !== 404) {
             console.error('Error fetching log:', error);
             if (statusRef.current !== 'error') setStatus('error');
        }
      });
  }, [jobID]);

  useEffect(() => () => requestController.current?.abort(), []);

  useEffect(() => {
    if(!jobID) return;
    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refresh]);

  return { log, setLog, status, refresh };
}