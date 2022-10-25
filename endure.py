#!/usr/bin/env python
import logging
import os
import toml
import sys

from jobs.gen_data import DataGenJob
from jobs.train import TrainJob


class EndureDriver:
    def __init__(self, config):
        self.config = config

        logging.basicConfig(format=config['log']['format'],
                            datefmt=config['log']['datefmt'])

        self.log = logging.getLogger(config['log']['name'])
        self.log.setLevel(config['log']['level'])

    def run(self):
        self.log.info(f'Staring app {self.config["app"]["name"]}')

        jobs = {
            'DataGen': DataGenJob,
            'Train': TrainJob
        }

        jobs_list = self.config['app']['jobs']

        for job_name in jobs_list:
            job = jobs.get(job_name, None)
            if (None):
                self.log.warn(f'No job associated with {job_name}')
                continue
            job = job(config)
            job.run()

        self.log.info('All jobs finished, exiting')


if __name__ == "__main__":
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        file_dir = os.path.dirname(__file__)
        config_path = os.path.join(file_dir, 'endure.toml')

    with open(config_path) as fid:
        config = toml.load(fid)

    driver = EndureDriver(config)
    driver.run()
