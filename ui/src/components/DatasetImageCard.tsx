import React, { useRef, useEffect, useState, KeyboardEvent, useMemo, useCallback } from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient, buildApiFileURL } from '@/utils/api';
import { isVideo } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string; // relative path under MOUNT_DIR, e.g. "datasets/default_user/ds1/img.png"
  alt: string;
  className?: string;
  onDelete?: () => void;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  className = '',
  onDelete = () => {},
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [hasEverBeenInViewport, setHasEverBeenInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);

  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [captionShort, setCaptionShort] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const [savedCaptionShort, setSavedCaptionShort] = useState<string>('');
  const [captionExt, setCaptionExt] = useState<'txt' | 'json'>('txt');
  const isGettingCaption = useRef<boolean>(false);

  const fullImageSrc = useMemo(() => {
    return buildApiFileURL(imageUrl);
  }, [imageUrl]);

  const fetchCaption = useCallback(async () => {
    if (isGettingCaption.current || isCaptionLoaded) return;
    isGettingCaption.current = true;

    try {
      const params = new URLSearchParams({ path: imageUrl });
      const captionUrl = `/api/files/caption?${params.toString()}`;

      const data = await apiClient.get(captionUrl).then(res => res.data);
      const text = (data?.caption as string) ?? '';
      const shortText = (data?.caption_short as string) ?? '';
      const sourceExt = data?.caption_ext === 'json' ? 'json' : 'txt';
      setCaption(text);
      setCaptionShort(shortText);
      setSavedCaption(text);
      setSavedCaptionShort(shortText);
      setCaptionExt(sourceExt);
    } catch {
      // treat as no caption
      setCaption('');
      setCaptionShort('');
      setSavedCaption('');
      setSavedCaptionShort('');
      setCaptionExt('txt');
    } finally {
      setIsCaptionLoaded(true);
      isGettingCaption.current = false;
    }
  }, [imageUrl, isCaptionLoaded]);

  const saveCaption = useCallback(async () => {
    const trimmedCaption = caption.trim();
    const trimmedCaptionShort = captionShort.trim();
    if (trimmedCaption === savedCaption && trimmedCaptionShort === savedCaptionShort) return;

    try {
      await apiClient.post('/api/files/caption', {
        path: imageUrl,
        caption: trimmedCaption,
        caption_short: captionExt === 'json' ? trimmedCaptionShort : undefined,
        caption_ext: captionExt,
      });
      setSavedCaption(trimmedCaption);
      setSavedCaptionShort(trimmedCaptionShort);
    } catch (error) {
      console.error('Error saving caption:', error);
    }
  }, [caption, captionExt, captionShort, imageUrl, savedCaption, savedCaptionShort]);

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        const entry = entries[0];
        if (entry.isIntersecting) {
          if (!isVisible) setIsVisible(true);
          if (!hasEverBeenInViewport) setHasEverBeenInViewport(true);
        }
      },
      { threshold: 0.1 },
    );

    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, [hasEverBeenInViewport, isVisible]);

  useEffect(() => {
    if (hasEverBeenInViewport && !isCaptionLoaded) {
      fetchCaption();
    }
  }, [fetchCaption, hasEverBeenInViewport, isCaptionLoaded]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption && captionShort.trim() === savedCaptionShort;
  const isItAVideo = isVideo(imageUrl);
  const isJsonCaption = captionExt === 'json';

  return (
    <div className={`flex flex-col ${className} group`}>
      <div
        ref={cardRef}
        className="relative w-full bg-gray-900 rounded-t-lg border border-gray-800 border-b-0 overflow-hidden"
        style={{ paddingBottom: '100%' }}
      >
        <div className="absolute inset-0">
          {hasEverBeenInViewport && (
            <>
              {isItAVideo ? (
                <video
                  src={fullImageSrc}
                  className="w-full h-full object-contain"
                  controls
                  preload="metadata"
                />
              ) : (
                <img
                  src={fullImageSrc}
                  alt={alt}
                  onLoad={() => setLoaded(true)}
                  className={`w-full h-full object-contain transition-opacity duration-300 ${
                    loaded ? 'opacity-100' : 'opacity-0'
                  }`}
                />
              )}
            </>
          )}

          <div className="absolute top-2 right-2 flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              className="bg-black/60 hover:bg-red-600 text-white rounded-md p-1.5 backdrop-blur-sm transition-colors"
              title="Delete Image"
              onClick={() => {
                openConfirm({
                  title: 'Delete File',
                  message: `Delete this ${isItAVideo ? 'video' : 'image'}?`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    apiClient
                      .delete(`/api/files/${encodeURIComponent(imageUrl)}`)
                      .then(() => onDelete())
                      .catch(error => console.error('Error deleting:', error));
                  },
                });
              }}
            >
              <FaTrashAlt size={12} />
            </button>
          </div>
        </div>

        <div className="absolute bottom-0 left-0 w-full bg-black/60 backdrop-blur-[2px] p-1 opacity-0 group-hover:opacity-100 transition-opacity">
          <p className="text-[10px] text-gray-300 truncate font-mono text-center">
            {imageUrl.split('/').pop()}
          </p>
        </div>
      </div>

      <div
        className={classNames(
          'w-full bg-gray-800 border-x border-b border-gray-700 rounded-b-lg relative transition-colors',
          isJsonCaption ? 'h-[148px]' : 'h-[80px]',
          {
            'border-blue-500/50 ring-1 ring-blue-500/20': !isCaptionCurrent,
          },
        )}
      >
        {isCaptionLoaded ? (
          <form
            className="h-full"
            onSubmit={e => {
              e.preventDefault();
              saveCaption();
            }}
            onBlur={saveCaption}
          >
            {isJsonCaption ? (
              <div className="grid h-full grid-rows-2 divide-y divide-gray-700">
                <label className="min-h-0">
                  <span className="sr-only">Long caption</span>
                  <textarea
                    className="w-full h-full bg-transparent p-2 text-xs text-gray-200 resize-none outline-none focus:bg-gray-750 transition-colors"
                    value={caption}
                    placeholder="Long caption..."
                    onChange={e => setCaption(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                </label>
                <label className="min-h-0">
                  <span className="sr-only">Short caption</span>
                  <textarea
                    className="w-full h-full bg-transparent p-2 text-xs text-gray-200 resize-none outline-none focus:bg-gray-750 transition-colors"
                    value={captionShort}
                    placeholder="Short caption..."
                    onChange={e => setCaptionShort(e.target.value)}
                    onKeyDown={handleKeyDown}
                  />
                </label>
              </div>
            ) : (
              <textarea
                className="w-full h-full bg-transparent p-2 text-xs text-gray-200 resize-none outline-none focus:bg-gray-750 transition-colors"
                value={caption}
                placeholder="Add a caption..."
                onChange={e => setCaption(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            )}
          </form>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500 text-xs">
            {isVisible ? 'Loading...' : '...'}
          </div>
        )}
      </div>
    </div>
  );
};

export default DatasetImageCard;
