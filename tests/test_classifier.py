import os.path
import subprocess
import unittest
from pathlib import Path

import project


class DeepHashingFastDevUnitTests(unittest.TestCase):
    """These tests run the entire flow of the PyTorch lightning modules in fast_dev mode, ensuring that there are
    no syntax errors or runtime errors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_dir = Path(os.path.dirname(project.__file__))
        self.dataset_dir = ''
        self.logs_dir = ''

    def test_sota_2016_CVPR_DSH(self):
        test_file = str(self.project_dir.joinpath('sota_2016_CVPR_DSH.py'))
        subprocess.run([r'python.exe', test_file, '--fast_dev=1', '--accelerator=cpu', '--num_workers=0'],
                       check=True)

    def test_sota_2017_NIPS_DSDH(self):
        test_file = str(self.project_dir.joinpath('sota_2017_NIPS_DSDH.py'))
        subprocess.run([r'python.exe', test_file, '--fast_dev=1', '--accelerator=cpu', '--num_workers=0'],
                       check=True)


if __name__ == '__main__':
    unittest.main()
