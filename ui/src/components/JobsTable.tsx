'use client';

import useJobsList from '@/hooks/useJobsList';
import Link from 'next/link';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { Job } from '@/utils/types';
import { CgSpinner } from 'react-icons/cg';
import { CheckCircle, XCircle, Clock, PlayCircle } from 'lucide-react';

interface JobsTableProps {
  onlyActive?: boolean;
}

export default function JobsTable({ onlyActive = false }: JobsTableProps) {
  const { jobs, status, refreshJobs } = useJobsList(onlyActive, 5000);

  const columns: TableColumn[] = [
    {
      title: 'Name',
      key: 'config_name',
      render: (row: Job) => (
        <Link href={`/jobs/${row.job_id}`} className="font-medium whitespace-nowrap flex items-center hover:text-blue-400 transition-colors">
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
        }

        return (
          <span className={`flex items-center capitalize ${color}`}>
            {icon} {row.status}
          </span>
        );
      },
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'text-right',
      render: (row: Job) => (
        <Link 
          href={`/jobs/${row.job_id}`}
          className="text-xs bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded border border-gray-700 transition-colors"
        >
          View Details
        </Link>
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