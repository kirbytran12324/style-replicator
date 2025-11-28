'use client';

import { useState, useEffect } from 'react';
import useSettings from '@/hooks/useSettings';
import { TopBar, MainContent } from '@/components/layout';
import { Button } from '@headlessui/react';
import { Save } from 'lucide-react';

export default function Settings() {
  const { settings, saveSettings, isSettingsLoaded } = useSettings();
  const [token, setToken] = useState('');
  const [status, setStatus] = useState<'idle' | 'saving' | 'success'>('idle');

  // Sync local state with loaded settings
  useEffect(() => {
    if (isSettingsLoaded) {
      setToken(settings.HF_TOKEN);
    }
  }, [settings, isSettingsLoaded]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setStatus('saving');

    // Save to LocalStorage
    saveSettings({ HF_TOKEN: token });

    // Fake loading for UX
    setTimeout(() => {
        setStatus('success');
        setTimeout(() => setStatus('idle'), 2000);
    }, 500);
  };

  return (
    <>
      <TopBar>
        <div>
          <h1 className="text-lg font-semibold">Settings</h1>
        </div>
        <div className="flex-1"></div>
      </TopBar>
      <MainContent>
        <div className="max-w-2xl mx-auto">
            <form onSubmit={handleSubmit} className="bg-gray-900 rounded-xl border border-gray-800 p-6 space-y-6">

                <div className="space-y-4">
                    <div>
                        <h2 className="text-xl font-medium text-gray-200 mb-1">API Configuration</h2>
                        <p className="text-sm text-gray-500">
                            Configure external services for your training jobs.
                        </p>
                    </div>

                    <div className="pt-4">
                        <label htmlFor="HF_TOKEN" className="block text-sm font-medium text-gray-300 mb-2">
                            Hugging Face Token
                        </label>
                        <input
                            type="password"
                            id="HF_TOKEN"
                            value={token}
                            onChange={(e) => setToken(e.target.value)}
                            className="w-full px-4 py-2.5 bg-gray-950 border border-gray-700 rounded-lg focus:ring-2 focus:ring-blue-600 focus:border-transparent text-gray-200 placeholder-gray-600"
                            placeholder="hf_..."
                        />
                        <p className="text-xs text-gray-500 mt-2">
                            Required if you are training on gated models (like FLUX.1-dev).
                            <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="text-blue-400 hover:underline ml-1">
                                Get your token here.
                            </a>
                        </p>
                    </div>
                </div>

                <div className="pt-4 border-t border-gray-800 flex items-center justify-between">
                    <p className="text-xs text-gray-500">
                        Settings are saved locally in your browser.
                    </p>
                    <Button
                        type="submit"
                        disabled={status === 'saving'}
                        className={`
                            flex items-center px-6 py-2 rounded-lg font-medium transition-all
                            ${status === 'success' 
                                ? 'bg-green-600 hover:bg-green-500 text-white' 
                                : 'bg-blue-600 hover:bg-blue-500 text-white'}
                        `}
                    >
                        <Save className="w-4 h-4 mr-2" />
                        {status === 'saving' ? 'Saving...' : status === 'success' ? 'Saved!' : 'Save Settings'}
                    </Button>
                </div>
            </form>
        </div>
      </MainContent>
    </>
  );
}