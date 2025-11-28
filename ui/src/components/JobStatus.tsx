'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';
import { Activity, CheckCircle, XCircle } from 'lucide-react';

export default function JobStatus() {
  const [status, setStatus] = useState<'online' | 'offline' | 'loading'>('loading');

  useEffect(() => {
    const checkHealth = async () => {
      try {
        // Simple health check to the API root
        await apiClient.get('/api');
        setStatus('online');
      } catch (e) {
        setStatus('offline');
      }
    };

    checkHealth();
    const interval = setInterval(checkHealth, 30000); // Check every 30s
    return () => clearInterval(interval);
  }, []);

  if (status === 'loading') return null;

  return (
    <div className={`
      flex items-center px-4 py-3 rounded-xl border mb-6
      ${status === 'online' 
        ? 'bg-emerald-950/30 border-emerald-800/50 text-emerald-400' 
        : 'bg-red-950/30 border-red-800/50 text-red-400'}
    `}>
      {status === 'online' ? (
        <>
          <div className="relative flex h-3 w-3 mr-3">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
            <span className="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
          </div>
          <span className="font-medium">Training Backend Online</span>
        </>
      ) : (
        <>
          <XCircle className="w-5 h-5 mr-3" />
          <span className="font-medium">Backend Unreachable (Is Modal running?)</span>
        </>
      )}
    </div>
  );
}