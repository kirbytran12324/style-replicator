'use client';

import { useMemo, useEffect, useRef } from 'react';
import { Info, Loader2 } from 'lucide-react';
import useJobLog from '@/hooks/useJobLog';
import { Job } from '@/utils/types';

const OVERVIEW_LOG_LINE_LIMIT = 25;

interface JobOverviewProps {
  job: Job;
}

export default function JobOverview({ job }: JobOverviewProps) {
  const { log, status: statusLog } = useJobLog(job.job_id || job.id || '', (job.status === 'running' || job.status === 'started') ? 2000 : null);
  const logRef = useRef<HTMLDivElement>(null);
  const progressInfo = useMemo(() => {
    const backend = job.progress;
    const step = backend?.step ?? 0;
    const total = backend?.total ?? 0;
    const percent = backend?.percent ?? (total ? (step / total) * 100 : 0);
    return {
      current: step,
      total,
      percent,
      message: backend?.message,
      phase: backend?.phase,
    };
  }, [job.progress]);

  const logLines = useMemo(() => {
    return log.split('\n').slice(-OVERVIEW_LOG_LINE_LIMIT);
  }, [log]);

  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [log]);

  return (
    <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
      {/* Main Info Panel */}
      <div className="col-span-2 bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col">
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between">
          <h2 className="text-gray-100 flex items-center font-medium">
            <Info className="w-5 h-5 mr-2 text-blue-400" /> 
            {job.config_name || 'Training Job'}
          </h2>
          <span className={`px-3 py-1 rounded-full text-xs font-medium uppercase tracking-wide bg-gray-700 text-gray-300`}>
            {job.status}
            {progressInfo.phase && ` · ${progressInfo.phase}`}
          </span>
        </div>

        <div className="p-4 space-y-6 flex flex-col flex-grow">
          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex items-center justify-between text-sm">
              <span className="text-gray-400">Progress</span>
              <span className="text-gray-200 font-mono">
                {progressInfo.current} / {progressInfo.total || '?'} Steps
              </span>
            </div>
            <div className="w-full bg-gray-800 rounded-full h-2 overflow-hidden">
              <div 
                className="h-2 rounded-full bg-blue-500 transition-all duration-500 ease-out" 
                style={{ width: `${progressInfo.percent}%` }} 
              />
            </div>
            {progressInfo.message && (
              <div className="text-xs text-gray-400 font-mono">{progressInfo.message}</div>
            )}
           </div>

          {/* Log Window */}
          <div className="bg-gray-950 rounded-lg border border-gray-800 relative flex-grow min-h-[400px] flex flex-col">
            <div className="px-4 py-2 border-b border-gray-800 text-xs text-gray-500 font-mono uppercase tracking-wider">
              Live Logs
            </div>
            <div className="px-4 py-1 text-[11px] text-gray-500 border-b border-gray-900">
              Showing last {OVERVIEW_LOG_LINE_LIMIT} lines. Open the Logs tab for the full log file.
            </div>
            <div
              ref={logRef}
              className="text-xs text-gray-300 flex-1 p-4 overflow-y-auto font-mono whitespace-pre-wrap leading-relaxed"
            >
              {statusLog === 'loading' && (
                <div className="flex items-center text-gray-500">
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" /> Connecting to remote logs...
                </div>
              )}
              {statusLog === 'success' && logLines.length === 0 && (
                <div className="text-gray-500">Logs will appear once the job starts writing to log.txt.</div>
              )}
               {logLines.map((line, index) => (
                 <div key={index}>{line}</div>
               ))}
             </div>
           </div>
         </div>
       </div>

      {/* Side Panel - Just Metadata now */}
      <div className="col-span-1 space-y-4">
        <div className="bg-gray-900 rounded-xl p-5 border border-gray-800">
            <h3 className="text-gray-400 text-xs uppercase font-bold tracking-wider mb-4">Job Details</h3>
            <div className="space-y-3">
                <div className="flex justify-between items-center">
                    <span className="text-gray-500 text-sm">ID</span>
                    <span className="text-gray-300 text-xs font-mono bg-gray-800 px-2 py-1 rounded">{job.job_id || job.id}</span>
                </div>
                <div className="flex justify-between items-center">
                    <span className="text-gray-500 text-sm">Created</span>
                    <span className="text-gray-300 text-sm">
                      {job.created_at ? new Date(job.created_at).toLocaleDateString() : '—'}
                    </span>
                </div>
                <div className="pt-2 border-t border-gray-800 mt-2">
                  <div className="text-xs text-gray-500 text-center">
                    Checkpoints are stored securely in the Modal Volume.
                  </div>
                </div>
            </div>
        </div>
      </div>
    </div>
  );
}