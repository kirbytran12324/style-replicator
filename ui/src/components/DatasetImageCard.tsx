import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent, useMemo } from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import { isVideo } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string; // relative path under MOUNT_DIR, e.g. "datasets/default_user/ds1/img.png"
  alt: string;
  children?: ReactNode;
  className?: string;
  onDelete?: () => void;
}

const DatasetImageCard: React.FC<DatasetImageCardProps> = ({
  imageUrl,
  alt,
  children,
  className = '',
  onDelete = () => {},
}) => {
  const cardRef = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState<boolean>(false);
  const [inViewport, setInViewport] = useState<boolean>(false);
  const [hasEverBeenInViewport, setHasEverBeenInViewport] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);

  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const isGettingCaption = useRef<boolean>(false);

  const fullImageSrc = useMemo(() => {
    const baseUrl = process.env.NEXT_PUBLIC_MODAL_API_URL || '';
    const cleanBase = baseUrl.replace(/\/$/, '');
    return `${cleanBase}/api/files/${imageUrl}`;
  }, [imageUrl]);

  const fetchCaption = async () => {
    if (isGettingCaption.current || isCaptionLoaded) return;
    isGettingCaption.current = true;

    try {
      const baseUrl = process.env.NEXT_PUBLIC_MODAL_API_URL || '';
      const cleanBase = baseUrl.replace(/\/$/, '');
      const params = new URLSearchParams({ path: imageUrl });
      const captionUrl = `${cleanBase}/api/files/caption?${params.toString()}`;

      const res = await fetch(captionUrl);
      if (!res.ok) {
        // caption endpoint returns 200 with { caption: "" } on missing files,
        // so non-2xx is a real error
        throw new Error(`Caption fetch failed: ${res.status}`);
      }
      const data = await res.json();
      const text = (data?.caption as string) ?? '';
      setCaption(text);
      setSavedCaption(text);
    } catch (_err) {
      // treat as no caption
      setCaption('');
      setSavedCaption('');
    } finally {
      setIsCaptionLoaded(true);
      isGettingCaption.current = false;
    }
  };

  const saveCaption = async () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;

    try {
      await apiClient.post('/api/files/caption', {
        path: imageUrl,
        caption: trimmedCaption,
      });
      setSavedCaption(trimmedCaption);
    } catch (error) {
      console.error('Error saving caption:', error);
    }
  };

  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        const entry = entries[0];
        if (entry.isIntersecting) {
          setInViewport(true);
          if (!isVisible) setIsVisible(true);
          if (!hasEverBeenInViewport) setHasEverBeenInViewport(true);
        } else {
          setInViewport(false);
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
  }, [hasEverBeenInViewport, isCaptionLoaded]);

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>): void => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      saveCaption();
    }
  };

  const isCaptionCurrent = caption.trim() === savedCaption;
  const isItAVideo = isVideo(imageUrl);

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
          'w-full bg-gray-800 border-x border-b border-gray-700 rounded-b-lg h-[80px] relative transition-colors',
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
            <textarea
              className="w-full h-full bg-transparent p-2 text-xs text-gray-200 resize-none outline-none focus:bg-gray-750 transition-colors"
              value={caption}
              placeholder="Add a caption..."
              onChange={e => setCaption(e.target.value)}
              onKeyDown={handleKeyDown}
            />
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
