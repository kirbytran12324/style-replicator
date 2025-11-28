'use client';

import { useEffect, useState } from 'react';
import { apiClient } from '@/utils/api';

export default function useModelList() {
  const [models, setModels] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(false);

  const refreshModels = () => {
    setIsLoading(true);
    // Calls Modal: GET /api/models
    // Backend should return: { models: ["my-cat-lora", "flux-test-1"] }
    apiClient.get('/api/models')
      .then(res => {
        if (res.data.models) {
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