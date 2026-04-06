interface AdvancedJobProps {
// ...existing code...
  datasetOptions: { value: string; label: string }[];
  settings: Settings;
  isDatasetsLoading?: boolean;
}

export default function AdvancedJob({
// ...existing code...
  datasetOptions,
  settings,
  isDatasetsLoading = false,
}: AdvancedJobProps) {
// ...existing code...
                <SelectInput
                  label="Dataset"
                  value={dataset.folder_path}
                  onChange={value =>
                    setJobConfigAction(value, `config.process[0].datasets[${index}].folder_path`)
                  }
                  options={datasetOptions}
                  disabled={isDatasetsLoading}
                />
                {isDatasetsLoading && <p className="text-xs text-gray-500 mt-1">Loading datasets...</p>}
// ...existing code...

