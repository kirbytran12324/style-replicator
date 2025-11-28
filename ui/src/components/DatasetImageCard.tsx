'use client';

import React, { useRef, useEffect, useState, ReactNode, KeyboardEvent, useMemo } from 'react';
import { FaTrashAlt } from 'react-icons/fa';
import { openConfirm } from './ConfirmModal';
import classNames from 'classnames';
import { apiClient } from '@/utils/api';
import { isVideo } from '@/utils/basic';

interface DatasetImageCardProps {
  imageUrl: string; // This is the relative path from the API response
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
  const [loaded, setLoaded] = useState<boolean>(false);
  const [isCaptionLoaded, setIsCaptionLoaded] = useState<boolean>(false);
  const [caption, setCaption] = useState<string>('');
  const [savedCaption, setSavedCaption] = useState<string>('');
  const isGettingCaption = useRef<boolean>(false);

  // Construct the full image URL for the <img> tag
  const fullImageSrc = useMemo(() => {
    const baseUrl = process.env.NEXT_PUBLIC_MODAL_API_URL || '';
    // Ensure we don't double slash
    const cleanBase = baseUrl.replace(/\/$/, '');
    // The endpoint we defined in python: @api.get("/api/files/{file_path:path}")
    // imageUrl usually comes in as "datasets/my-dataset/img.jpg"
    return `${cleanBase}/api/files/${imageUrl}`;
  }, [imageUrl]);

  const fetchCaption = async () => {
    if (isGettingCaption.current || isCaptionLoaded) return;
    isGettingCaption.current = true;
    
    // Call Modal: GET /api/files/caption?path=...
    apiClient
      .get(`/api/files/caption`, { params: { path: imageUrl } })
      .then(res => res.data)
      .then(data => {
        // data.caption should be the string
        const txt = data.caption || '';
        setCaption(txt);
        setSavedCaption(txt);
        setIsCaptionLoaded(true);
      })
      .catch(error => {
        // 404 means no caption file yet, which is fine
        if (error.response?.status !== 404) {
            console.error('Error fetching caption:', error);
        } else {
            setIsCaptionLoaded(true); // Loaded, just empty
        }
      })
      .finally(() => {
        isGettingCaption.current = false;
      });
  };

  const saveCaption = () => {
    const trimmedCaption = caption.trim();
    if (trimmedCaption === savedCaption) return;
    
    // Call Modal: POST /api/files/caption
    apiClient
      .post('/api/files/caption', { path: imageUrl, caption: trimmedCaption })
      .then(() => {
        setSavedCaption(trimmedCaption);
      })
      .catch(error => {
        console.error('Error saving caption:', error);
      });
  };

  // Intersection Observer to lazy load images/captions
  useEffect(() => {
    const observer = new IntersectionObserver(
      entries => {
        if (entries[0].isIntersecting) {
          setInViewport(true);
          if (!isVisible) setIsVisible(true);
        } else {
          setInViewport(false);
        }
      },
      { threshold: 0.1 },
    );

    if (cardRef.current) observer.observe(cardRef.current);
    return () => observer.disconnect();
  }, []);

  // Fetch caption when visible
  useEffect(() => {
    if (inViewport && isVisible) {
      fetchCaption();
    }
  }, [inViewport, isVisible]);

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
      {/* Square container */}
      <div
        ref={cardRef}
        className="relative w-full bg-gray-900 rounded-t-lg border border-gray-800 border-b-0 overflow-hidden"
        style={{ paddingBottom: '100%' }} 
      >
        <div className="absolute inset-0">
          {inViewport && isVisible && (
            <>
              {isItAVideo ? (
                <video
                  src={fullImageSrc}
                  className={`w-full h-full object-contain`}
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
          
          {/* Overlay Actions */}
          <div className="absolute top-2 right-2 flex space-x-2 opacity-0 group-hover:opacity-100 transition-opacity">
            <button
              className="bg-black/60 hover:bg-red-600 text-white rounded-md p-1.5 backdrop-blur-sm transition-colors"
              title="Delete Image"
              onClick={() => {
                openConfirm({
                  title: `Delete File`,
                  message: `Delete this ${isItAVideo ? 'video' : 'image'}?`,
                  type: 'warning',
                  confirmText: 'Delete',
                  onConfirm: () => {
                    apiClient
                      .delete(`/api/files/${encodeURIComponent(imageUrl)}`) // Use DELETE method
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
        
        {/* Filename overlay */}
        <div className="absolute bottom-0 left-0 w-full bg-black/60 backdrop-blur-[2px] p-1 opacity-0 group-hover:opacity-100 transition-opacity">
            <p className="text-[10px] text-gray-300 truncate font-mono text-center">
                {imageUrl.split('/').pop()}
            </p>
        </div>
      </div>

      {/* Caption Box */}
      <div
        className={classNames('w-full bg-gray-800 border-x border-b border-gray-700 rounded-b-lg h-[80px] relative transition-colors', {
          'border-blue-500/50 ring-1 ring-blue-500/20': !isCaptionCurrent,
        })}
      >
        {isCaptionLoaded ? (
          <form
            className="h-full"
            onSubmit={e => { e.preventDefault(); saveCaption(); }}
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