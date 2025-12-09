from collections import OrderedDict
from jobs import TrainJob
from jobs.process import BaseSDTrainProcess  # Changed from BaseTrainProcess


# Inherit from BaseSDTrainProcess to get the training loop and progress updates
class TrainFineTuneProcess(BaseSDTrainProcess):
    def __init__(self, process_id: int, job: TrainJob, config: OrderedDict):
        super().__init__(process_id, job, config)

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        return loss_dict