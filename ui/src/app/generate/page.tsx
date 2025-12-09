'use client';

import { useState, useEffect } from 'react';
import { apiClient } from '@/utils/api';
import { Button } from '@headlessui/react';
import { TextInput, NumberInput, SelectInput } from '@/components/formInputs';
import { TopBar, MainContent } from '@/components/layout';
import ImageGenerator from '@/components/ImageGenerator';
import { Loader2, Sparkles, AlertCircle } from 'lucide-react';
import useModelList from '@/hooks/useModelList';
import useSettings from '@/hooks/useSettings'; // Import settings hook

export default function GeneratePage() {
  // We assume models is now a list of objects { name: string, base_model: string }
  // based on the previous step's plan.
  const { models, isLoading: modelsLoading } = useModelList();
  const { settings } = useSettings(); // Get the token

  const [prompt, setPrompt] = useState('');
  const [numSamples, setNumSamples] = useState(1);
  const [selectedModel, setSelectedModel] = useState<string>('');

  // New state for the base model, defaulting to Flux
  const [baseModel, setBaseModel] = useState('black-forest-labs/FLUX.1-dev');

  const [images, setImages] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Convert models array to options format
  // Note: We access m.name since models are now objects
  const modelOptions = [
    { value: '', label: 'None (Base Model Only)' },
    ...models.map(m => ({ value: m.name, label: m.name }))
  ];

  // Auto-update the Base Model input when a LoRA is selected
  useEffect(() => {
    if (!selectedModel) return;

    // Find the metadata for the selected LoRA
    const modelInfo = models.find(m => m.name === selectedModel);

    // If we found it and it has a base_model recorded, update the input
    if (modelInfo && modelInfo.base_model) {
        setBaseModel(modelInfo.base_model);
    }
  }, [selectedModel, models]);

  const handleGenerate = async () => {
    if (!prompt) return;
    setLoading(true);
    setError(null);
    setImages([]);

    try {
      const res = await apiClient.post('/api/generate', {
        prompt,
        num_samples: numSamples,
        model_name: selectedModel || null,
        base_model: baseModel,       // Send the base model
        hf_token: settings.HF_TOKEN  // Send the auth token
      });

      if (res.data && res.data.images) {
        const baseUrl = (process.env.NEXT_PUBLIC_MODAL_API_URL || '').replace(/\/$/, '');

        const processedImages = res.data.images.map((img: string) => {
          if (img.startsWith('/')) {
            return `${baseUrl}${img}`;
          } else if (!img.startsWith('data:')) {
             return `${baseUrl}/api/files/${img}`;
          }
          return img;
        });

        setImages(processedImages);
      }
    } catch (e: any) {
      console.error(e);
      setError(e.response?.data?.detail || 'Failed to generate image.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <TopBar>
        <h1 className="text-lg font-semibold">Generate</h1>
      </TopBar>
      <MainContent>
        <div className="max-w-4xl mx-auto space-y-8 pb-20">
          <div className="bg-gray-900 p-6 rounded-xl border border-gray-800 space-y-6 shadow-lg">

            {/* Models Configuration Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Base Model Input */}
                <div>
                    <TextInput
                        label="Base Model (HuggingFace ID)"
                        value={baseModel}
                        onChange={setBaseModel}
                        placeholder="black-forest-labs/FLUX.1-dev"
                        disabled={loading}
                    />
                    <p className="text-xs text-gray-500 mt-1">
                        If using a LoRA, this will auto-fill with the training model.
                    </p>
                </div>

                {/* LoRA Selection */}
                <div>
                    <SelectInput
                        label="Select Trained Model (LoRA)"
                        value={selectedModel}
                        onChange={setSelectedModel}
                        options={modelOptions}
                        disabled={modelsLoading}
                        placeholder={modelsLoading ? "Loading models..." : "Select a model"}
                    />
                </div>
            </div>

            {/* Prompt & Generation Controls */}
            <div className="flex flex-col md:flex-row gap-4">
              <div className="flex-grow">
                <TextInput
                    label="Prompt"
                    value={prompt}
                    onChange={setPrompt}
                    placeholder="A cinematic shot of..."
                    disabled={loading}
                />
              </div>
              <div className="md:w-32">
                <NumberInput
                    label="Count"
                    value={numSamples}
                    onChange={(v) => setNumSamples(v || 1)}
                    min={1}
                    max={4}
                    disabled={loading}
                />
              </div>
            </div>

            <div className="flex justify-end pt-2">
              <Button
                  onClick={handleGenerate}
                  disabled={loading || !prompt}
                  className={`
                    flex items-center px-6 py-2.5 rounded-lg font-medium transition-all
                    ${loading || !prompt 
                      ? 'bg-gray-800 text-gray-500 cursor-not-allowed border border-gray-700' 
                      : 'bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 hover:shadow-blue-900/40'}
                  `}
              >
                  {loading ? <Loader2 className="animate-spin mr-2 h-5 w-5" /> : <Sparkles className="mr-2 h-5 w-5" />}
                  {loading ? 'Dreaming...' : 'Generate'}
              </Button>
            </div>
          </div>

          {error && (
            <div className="p-4 bg-red-900/20 border border-red-800/50 rounded-lg flex items-center text-red-200">
              <AlertCircle className="h-5 w-5 mr-2" />
              {error}
            </div>
          )}

          {images.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
              {images.map((img, i) => (
                <ImageGenerator
                  key={i}
                  src={img}
                  alt={`Generated image ${i + 1}`}
                  index={i}
                />
              ))}
            </div>
          )}
        </div>
      </MainContent>
    </>
  );
}