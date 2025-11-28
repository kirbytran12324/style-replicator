// src/utils/jobs.ts
import { JobConfig, Job } from '@/utils/types';
import { apiClient } from '@/utils/api';

export const startJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/start`) // Changed to POST
      .then(res => res.data)
      .then(data => {
        console.log('Job started:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error starting job:', error);
        reject(error);
      });
  });
};

export const stopJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/stop`) // Changed to POST
      .then(res => res.data)
      .then(data => {
        console.log('Job stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error stopping job:', error);
        reject(error);
      });
  });
};

export const deleteJob = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .delete(`/api/jobs/${jobID}`) // Changed to DELETE
      .then(res => res.data)
      .then(data => {
        console.log('Job deleted:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error deleting job:', error);
        reject(error);
      });
  });
};

export const markJobAsStopped = (jobID: string) => {
  return new Promise<void>((resolve, reject) => {
    apiClient
      .post(`/api/jobs/${jobID}/mark_stopped`) // Changed to POST
      .then(res => res.data)
      .then(data => {
        console.log('Job marked as stopped:', data);
        resolve();
      })
      .catch(error => {
        console.error('Error marking job as stopped:', error);
        reject(error);
      });
  });
};

export const getJobConfig = (job: Job): JobConfig | null => {
  if (!job.job_config) return null;
  try {
    return JSON.parse(job.job_config) as JobConfig;
  } catch (e) {
    console.error("Failed to parse job config", e);
    return null;
  }
};

export const getAvaliableJobActions = (job: Job) => {
  const jobConfig = getJobConfig(job);
  // Default to false/0 if fields are missing from backend
  const isStopping = (job.stop || false) && job.status === 'running';
  const currentStep = job.step || 0;
  
  // Normalize status to lowercase for comparison
  const status = job.status.toLowerCase();

  const canDelete = ['queued', 'completed', 'stopped', 'error', 'failed'].includes(status) && !isStopping;
  const canEdit = ['queued','completed', 'stopped', 'error', 'failed'].includes(status) && !isStopping;
  const canRemoveFromQueue = status === 'queued';
  const canStop = (status === 'running' || status === 'started') && !isStopping;
  
  let canStart = ['stopped', 'error', 'failed'].includes(status) && !isStopping;
  
  // Can resume if more steps were added
  // We check if jobConfig exists first
  if (status === 'completed' && jobConfig && jobConfig.config.process[0].train.steps > currentStep && !isStopping) {
    canStart = true;
  }
  
  return { canDelete, canEdit, canStop, canStart, canRemoveFromQueue };
};

export const getNumberOfSamples = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig?.config.process[0].sample?.prompts?.length || 0;
};

export const getTotalSteps = (job: Job) => {
  const jobConfig = getJobConfig(job);
  return jobConfig?.config.process[0].train.steps || 0;
};