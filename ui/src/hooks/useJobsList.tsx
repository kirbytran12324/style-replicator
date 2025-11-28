'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { Job } from '@/utils/types';

export default function useJobsList(onlyActive = false, reloadInterval: null | number = null) {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshJobs = () => {
    // Don't set loading on every refresh to avoid flicker
    if (status === 'idle') setStatus('loading');

    apiClient
      .get('/api/jobs')
      .then(res => {
        let fetchedJobs: Job[] = res.data.jobs || [];

        // Map 'job_id' to 'id' for compatibility with existing UI components if needed
        fetchedJobs = fetchedJobs.map(j => ({ ...j, id: j.job_id }));

        if (onlyActive) {
          fetchedJobs = fetchedJobs.filter((job) =>
            ['started', 'running', 'queued'].includes(job.status)
          );
        }

        // Sort by created_at desc (assuming ISO strings)
        fetchedJobs.sort((a, b) => {
            return (b.created_at || '').localeCompare(a.created_at || '');
        });

        setJobs(fetchedJobs);
        setStatus('success');
      })
      .catch(error => {
        console.error('Error fetching jobs:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshJobs();

    if (reloadInterval) {
      const interval = setInterval(refreshJobs, reloadInterval);
      return () => clearInterval(interval);
    }
  }, [onlyActive, reloadInterval]);

  return { jobs, setJobs, status, refreshJobs };
}