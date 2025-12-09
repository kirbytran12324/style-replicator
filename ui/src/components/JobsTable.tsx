'use client';

import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { Job } from '@/utils/types';
import { apiClient } from '@/utils/api'; //
import { CgSpinner } from 'react-icons/cg';
import { CheckCircle, XCircle, Clock, PlayCircle, PauseCircle, Trash2, RotateCcw } from 'lucide-react';

interface JobsTableProps {
  onlyActive?: boolean;
}

export default function JobsTable({ onlyActive = false }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList(onlyActive, 5000);

  async function handleDelete(jobId: string) {
    try {
      // Changed from fetch to apiClient.delete to use the correct baseURL
      await apiClient.delete(`/api/jobs/${jobId}`);
      await refreshJobs();
    } catch (err: any) {
      console.error(err);
      // Optional: Add UI error handling here (e.g. toast notification)
      // const msg = err.response?.data?.detail || `Failed to delete job ${jobId}`;
    }
  }

  async function handlePause(jobId: string) {
    try {
      // Changed from fetch to apiClient.post
      await apiClient.post(`/api/jobs/${jobId}/pause`);
      await refreshJobs();
    } catch (err: any) {
      console.error(err);
    }
  }

  async function handleResume(jobId: string) {
    try {
      // Changed from fetch to apiClient.post
      await apiClient.post(`/api/jobs/${jobId}/resume`);
      await refreshJobs();
    } catch (err: any) {
      console.error(err);
    }
  }

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'config_name',
      render: (row: Job) => (
        <Link
          href={`/jobs/${row.job_id}`}
          className="font-medium whitespace-nowrap flex items-center hover:text-blue-400 transition-colors"
        >
          {['running', 'started'].includes(row.status) ? (
            <CgSpinner className="inline animate-spin mr-2 text-blue-400" />
          ) : null}
          {row.config_name || row.job_id}
        </Link>
      ),
    },
    {
      title: 'ID',
      key: 'job_id',
      className: 'font-mono text-xs text-gray-500 truncate max-w-[100px]',
    },
    {
      title: 'Status',
      key: 'status',
      render: (row: Job) => {
        let color = 'text-gray-400';
        let icon = <Clock className="w-4 h-4 mr-1 inline" />;

        switch (row.status) {
          case 'completed':
            color = 'text-emerald-400';
            icon = <CheckCircle className="w-4 h-4 mr-1 inline" />;
            break;
          case 'failed':
            color = 'text-red-400';
            icon = <XCircle className="w-4 h-4 mr-1 inline" />;
            break;
          case 'running':
          case 'started':
            color = 'text-blue-400';
            icon = <PlayCircle className="w-4 h-4 mr-1 inline" />;
            break;
          case 'paused':
            color = 'text-yellow-400';
            icon = <PauseCircle className="w-4 h-4 mr-1 inline" />;
            break;
        }

        const percent = row.progress?.percent ?? null;
        return (
          <div className="flex flex-col gap-1">
            <span className={`flex items-center capitalize ${color}`}>
              {icon} {row.status}
              {percent !== null && row.status !== 'completed' && (
                <span className="ml-2 text-xs text-gray-400">{percent.toFixed(1)}%</span>
              )}
            </span>
            {percent !== null && row.status !== 'completed' && (
              <div className="h-1 rounded bg-gray-800 overflow-hidden">
                <div className="h-full bg-blue-500" style={{ width: `${Math.min(100, Math.max(0, percent))}%` }} />
              </div>
            )}
          </div>
        );
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: (row: Job) => (
        <div className="flex justify-end gap-2">
          <Link
            href={`/jobs/${row.job_id}`}
            className="text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded border border-gray-700 transition-colors"
          >
            View
          </Link>

          {['running', 'started'].includes(row.status) && (
            <button
              onClick={() => handlePause(row.job_id)}
              className="text-xs flex items-center bg-yellow-900/40 hover:bg-yellow-800/60 px-2 py-1 rounded border border-yellow-700 text-yellow-300 transition-colors"
            >
              <PauseCircle className="w-3 h-3 mr-1" /> Pause
            </button>
          )}

          {row.status === 'paused' && (
            <button
              onClick={() => handleResume(row.job_id)}
              className="text-xs flex items-center bg-blue-900/40 hover:bg-blue-800/60 px-2 py-1 rounded border border-blue-700 text-blue-300 transition-colors"
            >
              <RotateCcw className="w-3 h-3 mr-1" /> Resume
            </button>
          )}

          <button
            onClick={() => handleDelete(row.job_id)}
            className="text-xs flex items-center bg-red-900/40 hover:bg-red-800/60 px-2 py-1 rounded border border-red-700 text-red-300 transition-colors"
          >
            <Trash2 className="w-3 h-3 mr-1" /> Delete
          </button>
        </div>
      ),
    },
  ];

  return (
    <div className="space-y-6">
      <UniversalTable
        columns={columns}
        rows={jobs}
        isLoading={status === 'loading' && jobs.length === 0}
        onRefresh={refreshJobs}
      />
    </div>
  );
}
