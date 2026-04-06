import os
from collections import OrderedDict
from jobs import BaseJob
from toolkit.extension import get_all_extensions_process_dict
from toolkit.paths import CONFIG_ROOT
from jobs.process.TrainFineTuneProcess import TrainFineTuneProcess


class ExtensionJob(BaseJob):

    def __init__(self, config: OrderedDict):
        super().__init__(config)
        self.device = self.get_conf('device', 'cpu')
        self.process_dict = get_all_extensions_process_dict()

        if 'diffusion_trainer' not in self.process_dict:
            self.process_dict['diffusion_trainer'] = TrainFineTuneProcess

        self.load_processes(self.process_dict)

    def run(self):
        super().run()

        print("")
        print(f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}")

        for process in self.process:
            process.run()