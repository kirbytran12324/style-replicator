import { useMemo, useState, useRef, useEffect } from 'react';
import useSampleImages from '@/hooks/useSampleImages';
import SampleImageCard from './SampleImageCard';
import { JobConfig, Job } from '@/utils/types';
import { LuImageOff, LuLoader, LuBan } from 'react-icons/lu';
import { FaCaretDown, FaCaretUp } from 'react-icons/fa';
import SampleImageViewer from './SampleImageViewer';


interface SampleImagesProps {
  job: Job;
}

export default function SampleImages({ job }: SampleImagesProps) {
  // CHANGED: Use job_id to be explicit, though id is likely aliased in your hook
  const { sampleImages, status, refreshSampleImages } = useSampleImages(job.job_id, 5000);
  const [selectedSamplePath, setSelectedSamplePath] = useState<string | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const didFirstScroll = useRef(false);

  const numSamples = useMemo(() => {
    // CHANGED: Cast to any because the local Job interface might not have job_config
    // defined, and the python backend might not return it.
    const jConfig = (job as any).job_config;

    if (jConfig) {
      const jobConfig = JSON.parse(jConfig) as JobConfig;
      const sampleConfig = jobConfig.config.process[0].sample;
      const numPrompts = sampleConfig.prompts ? sampleConfig.prompts.length : 0;
      const numSamples = sampleConfig.samples.length;
      return Math.max(numPrompts, numSamples, 1);
    }
    return 10;
  }, [job]);

  const scrollToBottom = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: containerRef.current.scrollHeight, behavior: 'instant' });
    }
  };

  const scrollToTop = () => {
    if (containerRef.current) {
      containerRef.current.scrollTo({ top: 0, behavior: 'instant' });
    }
  };

  const PageInfoContent = useMemo(() => {
    let icon = null;
    let text = '';
    let subtitle = '';
    let showIt = false;
    let bgColor = '';
    let textColor = '';
    let iconColor = '';

    if (sampleImages.length > 0) return null;

    if (status == 'loading') {
      icon = <LuLoader className="animate-spin w-8 h-8" />;
      text = 'Loading Samples';
      subtitle = 'Please wait while we fetch your samples...';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }
    if (status == 'error') {
      icon = <LuBan className="w-8 h-8" />;
      text = 'Error Loading Samples';
      subtitle = 'There was a problem fetching the samples.';
      showIt = true;
      bgColor = 'bg-red-50 dark:bg-red-950/20';
      textColor = 'text-red-900 dark:text-red-100';
      iconColor = 'text-red-600 dark:text-red-400';
    }
    if (status == 'success' && sampleImages.length === 0) {
      icon = <LuImageOff className="w-8 h-8" />;
      text = 'No Samples Found';
      subtitle = 'No samples have been generated yet';
      showIt = true;
      bgColor = 'bg-gray-50 dark:bg-gray-800/50';
      textColor = 'text-gray-900 dark:text-gray-100';
      iconColor = 'text-gray-500 dark:text-gray-400';
    }

    if (!showIt) return null;

    return (
      <div
        className={`mt-10 flex flex-col items-center justify-center py-16 px-8 rounded-xl border-2 border-gray-700 border-dashed ${bgColor} ${textColor} mx-auto max-w-md text-center`}
      >
        <div className={`${iconColor} mb-4`}>{icon}</div>
        <h3 className="text-lg font-semibold mb-2">{text}</h3>
        <p className="text-sm opacity-75 leading-relaxed">{subtitle}</p>
      </div>
    );
  }, [status, sampleImages.length]);

  const gridTemplateColumns = useMemo(() => {
    const cols = Math.min(numSamples, 40);
    return `repeat(${Math.max(cols, 3)}, minmax(0, 1fr))`;
  }, [numSamples]);

  const sampleConfig = useMemo(() => {
    // CHANGED: Cast to any for the same reason
    const jConfig = (job as any).job_config;
    if (jConfig) {
      const jobConfig = JSON.parse(jConfig) as JobConfig;
      return jobConfig.config.process[0].sample;
    }
    return null;
  }, [job]);

  // scroll to bottom on first load of samples
  useEffect(() => {
    if (status === 'success' && sampleImages.length > 0 && !didFirstScroll.current) {
      didFirstScroll.current = true;
      setTimeout(() => {
        scrollToBottom();
      }, 100);
    }
  }, [status, sampleImages.length]);

  return (
    <div ref={containerRef} className="absolute top-[80px] left-0 right-0 bottom-0 overflow-y-auto">
      <div className="pb-4">
        {PageInfoContent}
        {sampleImages && (
          <div className="grid gap-1" style={{ gridTemplateColumns }}>
            {sampleImages.map((sample: string, idx: number) => {
              const groupIndex = Math.floor(idx / numSamples);
              const groupStart = groupIndex * numSamples;
              const groupEnd = Math.min(groupStart + numSamples, sampleImages.length);
              const groupSize = groupEnd - groupStart;
              const isEndOfGroup = idx === groupEnd - 1;

              const MIN_COLS = 3;
              const shouldPad = numSamples < MIN_COLS && groupSize < MIN_COLS;
              const padsNeeded = shouldPad ? MIN_COLS - groupSize : 0;

              return (
                <div key={sample} className="contents">
                  <SampleImageCard
                    imageUrl={sample}
                    alt="Sample Image"
                    onClick={() => setSelectedSamplePath(sample)}
                    observerRoot={containerRef.current}
                  />

                  {isEndOfGroup &&
                    padsNeeded > 0 &&
                    Array.from({ length: padsNeeded }).map((_, i) => (
                      <div key={`pad-${groupIndex}-${i}`} className="invisible" />
                    ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
      <SampleImageViewer
        imgPath={selectedSamplePath}
        numSamples={numSamples}
        sampleImages={sampleImages}
        onChange={setPath => setSelectedSamplePath(setPath)}
        sampleConfig={sampleConfig}
        refreshSampleImages={refreshSampleImages}
      />
      <div
        className="fixed top-20 mt-4 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToTop}
        title="Scroll to Top"
      >
        <FaCaretUp className="text-gray-500 dark:text-gray-400" />
      </div>
      <div
        className="fixed bottom-5 right-6 w-10 h-10 rounded-full bg-gray-900 shadow-lg flex items-center justify-center text-white opacity-80 hover:opacity-100 cursor-pointer"
        onClick={scrollToBottom}
        title="Scroll to Bottom"
      >
        <FaCaretDown className="text-gray-500 dark:text-gray-400" />
      </div>
    </div>
  );
}
