'use client';

import { useCallback, useEffect, useState, useRef } from 'react';
import axios from 'axios';

import { apiClient } from '@/utils/api';
import { Job } from '@/utils/types';

export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  // Create a Ref to track the job state without triggering re-renders
  const jobRef = useRef<Job | null>(null);
  const requestController = useRef<AbortController | null>(null);

  // Keep the Ref in sync with state
  useEffect(() => {
    jobRef.current = job;
  }, [job]);

  const refreshJob = useCallback(() => {
    requestController.current?.abort();
    const controller = new AbortController();
    requestController.current = controller;

    if (status === 'idle') setStatus('loading');

    apiClient
      .get(`/api/job-status/${jobID}`, { signal: controller.signal })
      .then(res => res.data)
      .then(async data => {
        const normalizedJob: Job & { job_config_text?: string | null } = { ...data, id: data.job_id };
        if (data.job_config) {
          normalizedJob.job_config = data.job_config;
        }

        //Read from the Ref instead of the state variable 'job'
        const currentJob = jobRef.current;

        if (!currentJob || !currentJob.job_config_text) {
          try {
            const configResp = await apiClient.get(`/api/jobs/${jobID}/config`);
            normalizedJob.job_config_text = configResp.data?.config_yaml ?? null;
          } catch (configErr) {
            console.warn('Unable to fetch config.yaml for job', jobID, configErr);
            normalizedJob.job_config_text = null;
          }
        } else {
          normalizedJob.job_config_text = currentJob.job_config_text;
        }

        setJob(normalizedJob);
        if (status !== 'success') setStatus('success');
      })
      .catch(error => {
        if (axios.isAxiosError(error)) {
          if (error.code === 'ECONNABORTED' || error.code === 'ERR_CANCELED') {
            return;
          }
        }
        console.error('Error fetching job:', error);
        setStatus('error');
      });
  }, [jobID, status]);

  useEffect(() => () => requestController.current?.abort(), []);

  useEffect(() => {
    if(!jobID) return;
    refreshJob();
  }, [jobID, refreshJob]);

  useEffect(() => {
    if (!reloadInterval) return;
    if (!(job?.status === 'running' || job?.status === 'started' || !job)) return;

    const interval = setInterval(refreshJob, reloadInterval);
    return () => clearInterval(interval);
  }, [reloadInterval, job?.status, jobID, refreshJob]);

  return { job, setJob, status, refreshJob };
}