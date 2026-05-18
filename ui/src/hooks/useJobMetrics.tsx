'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';
import { apiClient, buildApiFileURL } from '@/utils/api';
import { JobMetricsResponse, JobReportsResponse } from '@/utils/types';

const DEFAULT_LIMIT = 1000;

export default function useJobMetrics(jobID: string, reloadInterval: null | number = null, limit = DEFAULT_LIMIT) {
  const [metrics, setMetrics] = useState<JobMetricsResponse | null>(null);
  const [reports, setReports] = useState<JobReportsResponse | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const requestController = useRef<AbortController | null>(null);
  const statusRef = useRef(status);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const refresh = useCallback(async () => {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;

    if (statusRef.current === 'idle') setStatus('loading');

    try {
      const [metricsResp, reportsResp] = await Promise.allSettled([
        apiClient.get(`/api/jobs/${jobID}/metrics`, {
          params: { limit },
          signal: controller.signal,
        }),
        apiClient.get(`/api/jobs/${jobID}/reports`, { signal: controller.signal }),
      ]);

      if (metricsResp.status === 'fulfilled') {
        setMetrics(metricsResp.value.data as JobMetricsResponse);
        setErrorMessage(null);
        if (statusRef.current !== 'success') setStatus('success');
      } else {
        const error = metricsResp.reason;
        if (axios.isAxiosError(error)) {
          if (error.code === 'ECONNABORTED' || error.code === 'ERR_CANCELED') {
            return;
          }
          const detail = error.response?.data?.detail;
          const statusCode = error.response?.status;
          setErrorMessage(detail ? `${detail}` : statusCode ? `HTTP ${statusCode}` : 'Unknown error');
        } else {
          setErrorMessage('Unknown error');
        }
        if (statusRef.current !== 'error') setStatus('error');
      }

      if (reportsResp.status === 'fulfilled') {
        setReports(reportsResp.value.data as JobReportsResponse);
      } else {
        setReports(prev => prev ?? { reports: { html: [], png: [] }, latest_dir: null });
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        if (error.code === 'ECONNABORTED' || error.code === 'ERR_CANCELED') {
          return;
        }
      }
      console.error('Error fetching metrics:', error);
      setErrorMessage('Unknown error');
      if (statusRef.current !== 'error') setStatus('error');
    }
  }, [jobID, limit]);

  const generateReports = useCallback(async () => {
    const response = await apiClient.post(`/api/jobs/${jobID}/reports/generate`);
    setReports(response.data as JobReportsResponse);
  }, [jobID]);

  const downloadZip = useCallback(
    async (format: 'all' | 'png' | 'html') => {
      const response = await apiClient.post(`/api/jobs/${jobID}/reports/zip`, null, {
        params: { format },
      });
      const zipPath = response.data?.zip_path as string | undefined;
      if (!zipPath) return;

      const downloadPath = buildApiFileURL(zipPath);
      const a = document.createElement('a');
      a.href = downloadPath;
      a.download = `training-reports-${format}.zip`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    },
    [jobID],
  );

  useEffect(() => () => requestController.current?.abort(), []);

  useEffect(() => {
    if (!jobID) return;
    refresh();

    if (reloadInterval) {
      const interval = setInterval(refresh, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID, reloadInterval, refresh]);

  return {
    metrics,
    reports,
    status,
    errorMessage,
    refresh,
    generateReports,
    downloadZip,
  };
}
