'use client';

import JobStatus from '@/components/JobStatus';
import JobsTable from '@/components/JobsTable';
import { TopBar, MainContent } from '@/components/layout';
import Link from 'next/link';

export default function Dashboard() {
  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg font-semibold">Dashboard</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <JobStatus />

        <div className="w-full mt-4">
          <div className="flex justify-between items-center mb-4">
            <h2 className="text-lg font-medium text-gray-200">Recent Jobs</h2>
            <div className="text-sm text-gray-500 hover:text-gray-300 transition-colors">
              <Link href="/jobs">View All History &rarr;</Link>
            </div>
          </div>
          <JobsTable onlyActive />
        </div>
      </MainContent>
    </>
  );
}