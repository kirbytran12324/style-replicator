'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export interface ModelInfo {
  name: string;
  base_model: string;
}

export default function useModelList() {
  // Change state to hold ModelInfo objects
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refreshModels = () => {
    setIsLoading(true);
    apiClient.get('/api/models')
      .then(res => {
        if (res.data.models) {
          // Backend will now return objects like [{ name: 'my-lora', base_model: 'flux-dev' }]
          setModels(res.data.models);
        }
      })
      .catch(console.error)
      .finally(() => setIsLoading(false));
  };

  useEffect(() => {
    refreshModels();
  }, []);

  return { models, isLoading, refreshModels };
}