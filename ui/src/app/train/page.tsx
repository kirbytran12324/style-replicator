'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { defaultJobConfig, defaultDatasetConfig } from './jobConfig';
import { JobConfig } from '@/utils/types';
import { objectCopy } from '@/utils/basic';
import { useNestedState } from '@/utils/hooks';
import useSettings from '@/hooks/useSettings';
import useDatasetList from '@/hooks/useDatasetList';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { FaChevronLeft } from 'react-icons/fa';
import SimpleJob from './SimpleJob';
import AdvancedJob from './AdvancedJob';
import ErrorBoundary from '@/components/ErrorBoundary';
import { apiClient } from '@/utils/api';

// Mock GPU list for the UI since Modal manages hardware
const MOCK_GPU_LIST = [{ index: 0, name: "Cloud A100 (Managed)" }];

export default function TrainingForm() {
  const router = useRouter();
  const [gpuIDs, setGpuIDs] = useState<string | null>("0");
  const { settings, isSettingsLoaded } = useSettings();
  const { datasets, status: datasetFetchStatus } = useDatasetList();
  const [datasetOptions, setDatasetOptions] = useState<{ value: string; label: string }[]>([]);
  const [showAdvancedView, setShowAdvancedView] = useState(false);

  const [jobConfig, setJobConfig] = useNestedState<JobConfig>(objectCopy(defaultJobConfig));
  const [status, setStatus] = useState<'idle' | 'saving' | 'success' | 'error'>('idle');

  // Populate dataset dropdown
  useEffect(() => {
    if (datasetFetchStatus !== 'success') return;

    // Modal paths are virtual, but we keep the structure consistent
    // The value here is what gets sent to the python script
    const datasetOptions = datasets.map(name => ({
        value: `/root/modal_output/datasets/${name}`,
        label: name
    }));
    setDatasetOptions(datasetOptions);

    // Auto-select first dataset if default is still set
    const defaultDatasetPath = defaultDatasetConfig.folder_path;
    for (let i = 0; i < jobConfig.config.process[0].datasets.length; i++) {
      const dataset = jobConfig.config.process[0].datasets[i];
      if (dataset.folder_path === defaultDatasetPath && datasetOptions.length > 0) {
        setJobConfig(datasetOptions[0].value, `config.process[0].datasets[${i}].folder_path`);
      }
    }
  }, [datasets, datasetFetchStatus]);

  const saveJob = async () => {
    if (status === 'saving') return;
    setStatus('saving');

    try {
      // MODAL API CALL
      const res = await apiClient.post('/api/train', {
        name: jobConfig.config.name,
        config: jobConfig, // Send the whole config object
        recover: false,
        hf_token: settings.HF_TOKEN
      });

      setStatus('success');
      // Redirect to the new job's details page
      if (res.data.job_id) {
        router.push(`/jobs/${res.data.job_id}`);
      }

    } catch (error: any) {
        console.error('Error saving training:', error);
        alert(`Failed to start job: ${error.response?.data?.detail || error.message}`);
        setStatus('error');
    } finally {
        setTimeout(() => setStatus('idle'), 2000);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    saveJob();
  };

  return (
    <>
      <TopBar>
        <div>
          <Button className="text-gray-500 dark:text-gray-300 px-3 mt-1" onClick={() => router.push('/dashboard')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg">New Training Job</h1>
        </div>
        <div className="flex-1"></div>

        {/* Toggle View Button */}
        <div className="pr-2">
          <Button
            className="text-gray-200 bg-gray-800 px-3 py-1 rounded-md text-sm border border-gray-700"
            onClick={() => setShowAdvancedView(!showAdvancedView)}
          >
            {showAdvancedView ? 'Simple View' : 'Advanced View'}
          </Button>
        </div>

        {/* Save Button */}
        <div>
          <Button
            className={`
                px-4 py-1 rounded-md text-white font-medium transition-colors
                ${status === 'saving' ? 'bg-blue-800 cursor-wait' : 'bg-blue-600 hover:bg-blue-500'}
            `}
            onClick={() => saveJob()}
            disabled={status === 'saving'}
          >
            {status === 'saving' ? 'Starting...' : 'Start Training'}
          </Button>
        </div>
      </TopBar>

      <div className="h-full w-full bg-gray-950 overflow-hidden relative">
      {showAdvancedView ? (
        <div className="pt-[48px] h-full overflow-auto px-4 pb-20">
          <AdvancedJob
            jobConfig={jobConfig}
            setJobConfigAction={setJobConfig}
            status={status}
            handleSubmitAction={handleSubmit}
            runId={null}
            gpuIDs={gpuIDs}
            setGpuIDsAction={setGpuIDs}
            gpuList={MOCK_GPU_LIST} // Pass mock list
            datasetOptions={datasetOptions}
            settings={settings}
          />
        </div>
      ) : (
        <MainContent>
          <ErrorBoundary fallback={<div className="text-red-500">Error loading simple job form.</div>}>
            <SimpleJob
              jobConfig={jobConfig}
              setJobConfigAction={setJobConfig}
              status={status}
              handleSubmitAction={handleSubmit}
              runId={null}
              gpuIDs={gpuIDs}
              setGpuIDsAction={setGpuIDs}
              gpuList={MOCK_GPU_LIST} // Pass mock list
              datasetOptions={datasetOptions}
            />
          </ErrorBoundary>
          <div className="pb-20"></div>
        </MainContent>
      )}
      </div>
    </>
  );
}