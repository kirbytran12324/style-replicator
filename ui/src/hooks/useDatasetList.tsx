'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useDatasetList() {
  const [datasets, setDatasets] = useState<string[]>([]);
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshDatasets = () => {
    setStatus('loading');
    // MODAL API CHANGE: The endpoint is now simply /api/datasets
    apiClient
      .get('/api/datasets')
      .then(res => res.data)
      .then(data => {
        // Modal returns a simple list of strings ["dataset1", "dataset2"]
        if (Array.isArray(data)) {
            data.sort((a: string, b: string) => a.localeCompare(b));
            setDatasets(data);
            setStatus('success');
        } else {
            console.error('Unexpected response format:', data);
            setDatasets([]);
            setStatus('error');
        }
      })
      .catch(error => {
        console.error('Error fetching datasets:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    refreshDatasets();
  }, []);

  return { datasets, setDatasets, status, refreshDatasets };
}