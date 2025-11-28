'use client';

import React, { useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);

  // Transform datasets array into rows with objects
  const tableRows = datasets.map(dataset => ({
    name: dataset,
    actions: dataset,
  }));

  const columns: TableColumn[] = [
    {
      title: 'Dataset Name',
      key: 'name',
      render: (row: any) => (
        <Link href={`/datasets/${row.name}`} className="text-gray-200 hover:text-blue-400 font-medium transition-colors">
          {row.name}
        </Link>
      ),
    },
    {
      title: 'Actions',
      key: 'actions',
      className: 'w-20 text-right',
      render: (row: any) => (
        <button
          className="text-gray-400 hover:bg-red-900/30 hover:text-red-400 p-2 rounded-lg transition-all"
          onClick={() => handleDeleteDataset(row.name)}
          title="Delete Dataset"
        >
          <FaRegTrashAlt />
        </button>
      ),
    },
  ];

  const handleDeleteDataset = (datasetName: string) => {
    openConfirm({
      title: 'Delete Dataset',
      message: `Are you sure you want to delete the dataset "${datasetName}"? This action cannot be undone.`,
      type: 'warning',
      confirmText: 'Delete',
      onConfirm: async () => {
        try {
          // Calls Modal: POST /api/datasets/delete
          await apiClient.post('/api/datasets/delete', { name: datasetName });
          refreshDatasets();
        } catch (error) {
          console.error('Error deleting dataset:', error);
          alert('Failed to delete dataset');
        }
      },
    });
  };

  const handleCreateDataset = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newDatasetName.trim()) return;

    try {
      // Calls Modal: POST /api/datasets/create
      const data = await apiClient.post('/api/datasets/create', { name: newDatasetName }).then(res => res.data);
      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);

      if (data.name) {
        router.push(`/datasets/${data.name}`);
      }
    } catch (error) {
      console.error('Error creating new dataset:', error);
      alert('Failed to create dataset');
    }
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg font-semibold text-gray-100">Datasets</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-white bg-blue-600 px-4 py-1.5 rounded-md hover:bg-blue-500 transition-colors text-sm font-medium"
            onClick={() => setIsNewDatasetModalOpen(true)}
          >
            New Dataset
          </Button>
        </div>
      </TopBar>

      <MainContent>
        <UniversalTable
          columns={columns}
          rows={tableRows}
          isLoading={status === 'loading'}
          onRefresh={refreshDatasets}
        />
      </MainContent>

      <Modal
        isOpen={isNewDatasetModalOpen}
        onClose={() => setIsNewDatasetModalOpen(false)}
        title="New Dataset"
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              Enter a name for your new dataset folder.
            </div>
            <div className="mt-4">
              <TextInput
                label="Dataset Name"
                value={newDatasetName}
                onChange={value => setNewDatasetName(value)}
                placeholder="my-new-concept"
                required
              />
            </div>

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-800 px-4 py-2 text-gray-300 hover:bg-gray-700 transition-colors"
                onClick={() => setIsNewDatasetModalOpen(false)}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 transition-colors"
              >
                Create
              </button>
            </div>
          </form>
        </div>
      </Modal>
    </>
  );
}