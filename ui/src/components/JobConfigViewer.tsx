'use client';
import { useEffect, useState } from 'react';
import YAML from 'yaml';
import Editor from '@monaco-editor/react';

import { Job } from '@/utils/types';

interface Props {
  job: Job;
}

const yamlConfig: YAML.DocumentOptions &
  YAML.SchemaOptions &
  YAML.ParseOptions &
  YAML.CreateNodeOptions &
  YAML.ToStringOptions = {
  indent: 2,
  lineWidth: 999999999999,
  defaultStringType: 'QUOTE_DOUBLE',
  defaultKeyType: 'PLAIN',
  directives: true,
};

export default function JobConfigViewer({ job }: Props) {
  const [editorValue, setEditorValue] = useState<string>('');
  useEffect(() => {
    if (job?.job_config_text) {
      setEditorValue(job.job_config_text);
    } else if (job?.job_config) {
      try {
        const yamlContent = YAML.stringify(JSON.parse(job.job_config), yamlConfig);
        setEditorValue(yamlContent);
      } catch (err) {
        console.error('Unable to parse legacy job_config JSON', err);
        setEditorValue('');
      }
    } else {
      setEditorValue('');
    }
  }, [job]);
  const isEmpty = !editorValue?.trim();
  return (
    <div className="h-full">
      {isEmpty && (
        <div className="text-gray-400 text-sm mb-2 px-4">Config not available yet.</div>
      )}
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme="vs-dark"
        options={{
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          automaticLayout: true,
          readOnly: true,
        }}
      />
    </div>
  );
}
