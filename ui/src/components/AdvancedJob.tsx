'use client';
import { useEffect, useState, useRef } from 'react';
import { JobConfig } from '@/utils/types';
import YAML from 'yaml';
import Editor, { OnMount } from '@monaco-editor/react';
import type { editor } from 'monaco-editor';
import { Settings } from '@/hooks/useSettings';
import { migrateJobConfig } from '@/app/train/jobConfig';

type Props = {
  jobConfig: JobConfig;
  setJobConfigAction: (value: any, key?: string) => void;
  status: 'idle' | 'saving' | 'success' | 'error';
  handleSubmitAction: (event: React.FormEvent<HTMLFormElement>) => void;
  runId: string | null;
  gpuIDs: string | null;
  setGpuIDsAction: (value: string | null) => void;
  gpuList: any;
  datasetOptions: any;
  settings: Settings;
};

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

export default function AdvancedJob({ jobConfig, setJobConfigAction, settings: _settings }: Props) {
  const [editorValue, setEditorValue] = useState<string>('');
  const lastJobConfigUpdateStringRef = useRef('');
  const editorRef = useRef<editor.IStandaloneCodeEditor | null>(null);

  const isEditorMounted = useRef(false);

  const handleEditorDidMount: OnMount = editor => {
    editorRef.current = editor;
    isEditorMounted.current = true;

    try {
      const yamlContent = YAML.stringify(jobConfig, yamlConfig);
      setEditorValue(yamlContent);
      lastJobConfigUpdateStringRef.current = JSON.stringify(jobConfig);
    } catch (e) {
      console.warn(e);
    }
  };

  useEffect(() => {
    const lastUpdate = lastJobConfigUpdateStringRef.current;
    const currentUpdate = JSON.stringify(jobConfig);

    if (lastUpdate === currentUpdate || !isEditorMounted.current) {
      return;
    }

    try {
      const editor = editorRef.current;
      if (editor) {
        const position = editor.getPosition();
        const selection = editor.getSelection();
        const scrollTop = editor.getScrollTop();

        const yamlContent = YAML.stringify(jobConfig, yamlConfig);
        if (yamlContent !== editor.getValue()) {
          editor.getModel()?.setValue(yamlContent);

          if (position) editor.setPosition(position);
          if (selection) editor.setSelection(selection);
          editor.setScrollTop(scrollTop);
        }

        lastJobConfigUpdateStringRef.current = currentUpdate;
      }
    } catch (e) {
      console.warn(e);
    }
  }, [jobConfig]);

  const handleChange = (value: string | undefined) => {
    if (value === undefined) return;

    try {
      const parsed = YAML.parse(value);
      if (JSON.stringify(parsed) !== lastJobConfigUpdateStringRef.current) {
        lastJobConfigUpdateStringRef.current = JSON.stringify(parsed);

        try {
          parsed.config.process[0].sqlite_db_path = './aitk_db.db';
          parsed.config.process[0].training_folder = 'output';
          parsed.config.process[0].device = 'cuda';
          parsed.config.process[0].performance_log_every = 10;
        } catch (e) {
          console.warn(e);
        }
        migrateJobConfig(parsed);
        setJobConfigAction(parsed);
      }
    } catch (e) {
      console.warn(e);
    }
  };

  return (
    <>
      <Editor
        height="100%"
        width="100%"
        defaultLanguage="yaml"
        value={editorValue}
        theme="vs-dark"
        onChange={handleChange}
        onMount={handleEditorDidMount}
        options={{
          minimap: { enabled: true },
          scrollBeyondLastLine: false,
          automaticLayout: true,
        }}
      />
    </>
  );
}
