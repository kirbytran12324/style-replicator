'use client';

import React, { useState } from 'react';
import { Download, Check, ExternalLink } from 'lucide-react';

interface ImageGeneratorProps {
  src: string;
  alt: string;
  index: number;
}

export default function ImageGenerator({ src, alt, index }: ImageGeneratorProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleDownload = () => {
    const a = document.createElement('a');
    a.href = src;
    a.download = `generated-${Date.now()}-${index}.png`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  };

  return (
    <div
      className="relative group rounded-xl overflow-hidden bg-gray-950 border border-gray-800 shadow-md transition-all hover:shadow-xl hover:border-gray-700"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
    >
      {/* Image Display */}
      <div className="aspect-square w-full relative flex items-center justify-center bg-[url('/transparent-grid.svg')]">
        <img
          src={src}
          alt={alt}
          className="w-full h-full object-contain"
          loading="lazy"
        />
      </div>

      {/* Overlay Actions */}
      <div className={`
        absolute inset-0 bg-black/60 backdrop-blur-[2px] 
        flex flex-col items-center justify-center gap-3
        transition-opacity duration-200
        ${isHovered ? 'opacity-100' : 'opacity-0'}
      `}>
        <div className="flex gap-2">
          <button
            onClick={handleDownload}
            className="flex items-center gap-2 px-4 py-2 bg-white text-black rounded-full font-medium hover:bg-gray-200 transition-colors transform hover:scale-105 active:scale-95"
          >
            <Download className="w-4 h-4" />
            Download
          </button>

          <button
            onClick={() => {
                window.open(src, '_blank');
            }}
            className="p-2 bg-gray-800 text-white rounded-full hover:bg-gray-700 transition-colors border border-gray-600"
            title="Open in new tab"
          >
            <ExternalLink className="w-4 h-4" />
          </button>
        </div>
      </div>
    </div>
  );
}