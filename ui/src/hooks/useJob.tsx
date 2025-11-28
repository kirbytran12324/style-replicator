'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { Job } from '@/utils/types';

export default function useJob(jobID: string, reloadInterval: null | number = null) {
  const [job, setJob] = useState<Job | null>(null);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJob = () => {
    if (status === 'idle') setStatus('loading');

    apiClient
      .get(`/api/job-status/${jobID}`)
      .then(res => res.data)
      .then(data => {
        // Normalize
        const normalizedJob = { ...data, id: data.job_id };
        setJob(normalizedJob);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching job:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    if(!jobID) return;
    refreshJob();

    if (reloadInterval) {
      const interval = setInterval(refreshJob, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [jobID]);

  return { job, setJob, status, refreshJob };
}