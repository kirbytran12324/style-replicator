'use client';

import { useEffect, useState } from 'react';

export interface Settings {
  HF_TOKEN: string;
}

export default function useSettings() {
  const [settings, setSettings] = useState<Settings>({
    HF_TOKEN: '',
  });
  const [isSettingsLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Load from LocalStorage on mount
    const storedToken = localStorage.getItem('AI_TOOLKIT_HF_TOKEN');
    if (storedToken) {
      setSettings({ HF_TOKEN: storedToken });
    }
    setIsLoaded(true);
  }, []);

  const saveSettings = (newSettings: Settings) => {
    localStorage.setItem('AI_TOOLKIT_HF_TOKEN', newSettings.HF_TOKEN);
    setSettings(newSettings);
  };

  return { settings, setSettings, saveSettings, isSettingsLoaded };
}