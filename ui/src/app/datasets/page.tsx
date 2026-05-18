'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Modal } from '@/components/Modal';
import Link from 'next/link';
import { TextInput } from '@/components/formInputs';
import useDatasetList from '@/hooks/useDatasetList';
import { Button } from '@headlessui/react';
import { FaRegTrashAlt, FaUpload } from 'react-icons/fa';
import { openConfirm } from '@/components/ConfirmModal';
import { TopBar, MainContent } from '@/components/layout';
import UniversalTable, { TableColumn } from '@/components/UniversalTable';
import { apiClient } from '@/utils/api';
import { useRouter } from 'next/navigation';
import { useDropzone } from 'react-dropzone';

export default function Datasets() {
  const router = useRouter();
  const { datasets, status, refreshDatasets } = useDatasetList();
  const [newDatasetName, setNewDatasetName] = useState('');
  const [isNewDatasetModalOpen, setIsNewDatasetModalOpen] = useState(false);
  const [pendingUploadFiles, setPendingUploadFiles] = useState<File[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dropOverlayVisible, setDropOverlayVisible] = useState(false);
  const dragDepthRef = useRef(0);

  const isCreateAndUpload = pendingUploadFiles.length > 0;

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
      setIsUploading(isCreateAndUpload);
      setUploadProgress(0);
      // Calls Modal: POST /api/datasets/create
      const datasetName = newDatasetName.trim();
      const data = await apiClient.post('/api/datasets/create', { name: datasetName }).then(res => res.data);

      if (pendingUploadFiles.length > 0) {
        const formData = new FormData();
        pendingUploadFiles.forEach(file => formData.append('files', file));
        formData.append('name', datasetName);

        await apiClient.post('/api/datasets/upload', formData, {
          headers: {
            'Content-Type': 'multipart/form-data',
          },
          onUploadProgress: progressEvent => {
            if (progressEvent.total) {
              setUploadProgress(Math.round((progressEvent.loaded * 100) / progressEvent.total));
            }
          },
          timeout: 0,
        });
      }

      refreshDatasets();
      setNewDatasetName('');
      setIsNewDatasetModalOpen(false);
      setPendingUploadFiles([]);
      setIsUploading(false);
      setUploadProgress(0);

      if (data.name) {
        router.push(`/datasets/${encodeURIComponent(data.name)}`);
      }
    } catch (error) {
      console.error('Error creating new dataset:', error);
      alert(isCreateAndUpload ? 'Failed to create and upload dataset' : 'Failed to create dataset');
      setIsUploading(false);
    }
  };

  const closeNewDatasetModal = () => {
    if (isUploading) return;
    setIsNewDatasetModalOpen(false);
    setPendingUploadFiles([]);
    setUploadProgress(0);
  };

  const onDatasetDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length === 0) {
      setDropOverlayVisible(false);
      return;
    }

    setPendingUploadFiles(acceptedFiles);
    setIsNewDatasetModalOpen(true);
    setDropOverlayVisible(false);
  }, []);

  const dropAccept = useMemo(
    () => ({
      'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'],
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.m4v', '.flv'],
      'text/*': ['.txt'],
      'application/json': ['.json'],
    }),
    [],
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop: onDatasetDrop,
    accept: dropAccept,
    multiple: true,
    noClick: true,
    noKeyboard: true,
    preventDropOnDocument: true,
  });

  useEffect(() => {
    const isFileDrag = (e: DragEvent) => {
      const types = e.dataTransfer?.types;
      return !!types && Array.from(types).includes('Files');
    };

    const onDragEnter = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      dragDepthRef.current += 1;
      setDropOverlayVisible(true);
      e.preventDefault();
    };
    const onDragOver = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      setDropOverlayVisible(true);
    };
    const onDragLeave = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      dragDepthRef.current = Math.max(0, dragDepthRef.current - 1);
      if (dragDepthRef.current === 0) {
        setDropOverlayVisible(false);
      }
    };
    const onDrop = (e: DragEvent) => {
      if (!isFileDrag(e)) return;
      e.preventDefault();
      dragDepthRef.current = 0;
    };

    window.addEventListener('dragenter', onDragEnter);
    window.addEventListener('dragover', onDragOver);
    window.addEventListener('dragleave', onDragLeave);
    window.addEventListener('drop', onDrop);

    return () => {
      window.removeEventListener('dragenter', onDragEnter);
      window.removeEventListener('dragover', onDragOver);
      window.removeEventListener('dragleave', onDragLeave);
      window.removeEventListener('drop', onDrop);
    };
  }, []);

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
            onClick={() => {
              setPendingUploadFiles([]);
              setIsNewDatasetModalOpen(true);
            }}
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
        onClose={closeNewDatasetModal}
        title={isCreateAndUpload ? 'Create Dataset and Upload' : 'New Dataset'}
        size="md"
      >
        <div className="space-y-4 text-gray-200">
          <form onSubmit={handleCreateDataset}>
            <div className="text-sm text-gray-400">
              {isCreateAndUpload
                ? `Enter a name for the new dataset folder. ${pendingUploadFiles.length} files will be uploaded after it is created.`
                : 'Enter a name for your new dataset folder.'}
            </div>
            <div className="mt-4">
              <TextInput
                label="Dataset Name"
                value={newDatasetName}
                onChange={value => setNewDatasetName(value)}
                placeholder="my-new-concept"
                required
                disabled={isUploading}
              />
            </div>
            {isUploading && (
              <div className="mt-4">
                <div className="w-full bg-gray-700 rounded-full h-2.5">
                  <div className="bg-blue-600 h-2.5 rounded-full" style={{ width: `${uploadProgress}%` }}></div>
                </div>
                <p className="text-sm text-gray-300 mt-2 text-center">Uploading... {uploadProgress}%</p>
              </div>
            )}

            <div className="mt-6 flex justify-end space-x-3">
              <button
                type="button"
                className="rounded-md bg-gray-800 px-4 py-2 text-gray-300 hover:bg-gray-700 transition-colors"
                onClick={closeNewDatasetModal}
                disabled={isUploading}
              >
                Cancel
              </button>
              <button
                type="submit"
                className="rounded-md bg-blue-600 px-4 py-2 text-white hover:bg-blue-500 transition-colors"
                disabled={isUploading}
              >
                {isCreateAndUpload ? 'Create & Upload' : 'Create'}
              </button>
            </div>
          </form>
        </div>
      </Modal>

      <div
        className={`fixed inset-0 z-[9999] transition-opacity duration-200 ${
          dropOverlayVisible ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'
        }`}
        aria-hidden={!dropOverlayVisible}
        {...getRootProps()}
      >
        <input {...getInputProps()} />
        <div className="absolute inset-0 bg-gray-900/40" />
        <div className="absolute inset-0 flex items-center justify-center p-6">
          <div
            className={`w-full max-w-2xl rounded-2xl border-2 border-dashed px-8 py-10 text-center shadow-2xl backdrop-blur-sm
            ${isDragActive ? 'border-blue-400 bg-white/10' : 'border-white/30 bg-white/5'}`}
          >
            <div className="flex flex-col items-center gap-4">
              <FaUpload className="size-10 opacity-80" />
              <p className="text-lg font-semibold">Drop files to create a dataset</p>
              <p className="text-sm opacity-80">Images, videos, .txt captions, and .json captions are supported.</p>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
