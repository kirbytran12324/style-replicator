'use client';

import { useEffect, useState, use, useMemo } from 'react';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaChevronLeft } from 'react-icons/fa';
import DatasetImageCard from '@/components/DatasetImageCard';
import { Button } from '@headlessui/react';
import AddImagesModal, { openImagesModal } from '@/components/AddImagesModal';
import { TopBar, MainContent } from '@/components/layout';
import { apiClient } from '@/utils/api';
import FullscreenDropOverlay from '@/components/FullscreenDropOverlay';
import { useRouter } from 'next/navigation';

export default function DatasetPage({ params }: { params: { datasetName: string } }) {
  const router = useRouter();
  const [imgList, setImgList] = useState<{ img_path: string }[]>([]);
  // Unwrap params using React.use()
  const usableParams = use(params as any) as { datasetName: string };
  const datasetName = usableParams.datasetName; // Note: this is already decoded by Next.js
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');

  const refreshImageList = (dbName: string) => {
    setStatus('loading');
    // Calls Modal: GET /api/datasets/{name}/images
    apiClient
      .get(`/api/datasets/${encodeURIComponent(dbName)}/images`)
      .then((res: any) => {
        const data = res.data;
        if(data.images) {
            // Sort by filename
            data.images.sort((a: { img_path: string }, b: { img_path: string }) => a.img_path.localeCompare(b.img_path));
            setImgList(data.images);
            setStatus('success');
        } else {
            setImgList([]);
            setStatus('success');
        }
      })
      .catch(error => {
        console.error('Error fetching images:', error);
        setStatus('error');
      });
  };

  useEffect(() => {
    if (datasetName) {
      // Decode URI component just in case it came in encoded
      refreshImageList(decodeURIComponent(datasetName));
    }
  }, [datasetName]);

  const PageInfoContent = useMemo(() => {
    if (status === 'success' && imgList.length > 0) return null;

    let icon = null;
    let text = '';
    let subtitle = '';
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Images';
      subtitle = 'Fetching dataset contents...';
      bgColor = 'bg-gray-900/50';
      textColor = 'text-gray-200';
      iconColor = 'text-blue-400';
    } else if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Images';
      subtitle = 'Could not load this dataset. It might not exist.';
      bgColor = 'bg-red-950/20';
      textColor = 'text-red-200';
      iconColor = 'text-red-500';
    } else if (status == 'success' && imgList.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'Empty Dataset';
      subtitle = 'Drag and drop images here to get started.';
      bgColor = 'bg-gray-900/50';
      textColor = 'text-gray-400';
      iconColor = 'text-gray-600';
    }

    return (
      <div
        className={`mt-20 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-dashed border-gray-800 ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>

        {status === 'success' && imgList.length === 0 && (
             <Button
                className="mt-6 text-white bg-blue-600 px-4 py-2 rounded-md hover:bg-blue-500 transition-colors"
                onClick={() => openImagesModal(decodeURIComponent(datasetName), () => refreshImageList(datasetName))}
              >
                Upload Images
              </Button>
        )}
      </div>
    );
  }, [status, imgList.length, datasetName]);

  return (
    <>
      <TopBar>
        <div>
          <Button className="text-gray-400 hover:text-white px-3 mt-1" onClick={() => router.push('/datasets')}>
            <FaChevronLeft />
          </Button>
        </div>
        <div>
          <h1 className="text-lg font-semibold">{decodeURIComponent(datasetName)}</h1>
        </div>
        <div className="flex-1"></div>
        <div>
          <Button
            className="text-gray-200 bg-gray-800 border border-gray-700 hover:bg-gray-700 px-3 py-1.5 rounded-md text-sm transition-colors"
            onClick={() => openImagesModal(decodeURIComponent(datasetName), () => refreshImageList(datasetName))}
          >
            Add Images
          </Button>
        </div>
      </TopBar>
      <MainContent>
        {PageInfoContent}
        {status === 'success' && imgList.length > 0 && (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-4 pb-20">
            {imgList.map(img => (
              <DatasetImageCard
                key={img.img_path}
                alt="image"
                imageUrl={img.img_path}
                onDelete={() => refreshImageList(datasetName)}
              />
            ))}
          </div>
        )}
      </MainContent>
      <AddImagesModal />
      <FullscreenDropOverlay
        datasetName={decodeURIComponent(datasetName)}
        onComplete={() => refreshImageList(datasetName)}
      />
    </>
  );
}