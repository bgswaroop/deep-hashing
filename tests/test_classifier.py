import os.path
import sys
import unittest
from pathlib import Path

import project


class DeepHashingFastDevUnitTests(unittest.TestCase):
    """These tests run the entire flow of the PyTorch lightning modules in fast_dev mode, ensuring that there are
    no syntax errors or runtime errors."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_dir = Path(os.path.dirname(project.__file__))
        self.dataset_dir = self.project_dir.joinpath('dataset_dir')
        self.dataset_dir.mkdir(exist_ok=True)
        self.logs_dir = self.project_dir.joinpath('logs_dir')
        self.logs_dir.mkdir(exist_ok=True)

    def test_sota_2016_CVPR_DSH(self):
        from project.sota_2016_CVPR_DSH import main
        test_file = str(self.project_dir.joinpath('sota_2016_CVPR_DSH.py'))
        sys.argv = [test_file, '--fast_dev=1', '--accelerator=cpu', '--num_workers=0', '--no-finetune',
                    f'--dataset_dir={str(self.dataset_dir)}', f'--logs_dir={str(self.logs_dir)}']
        main()

    def test_sota_2017_NIPS_DSDH(self):
        from project.sota_2017_NIPS_DSDH import main
        test_file = str(self.project_dir.joinpath('sota_2017_NIPS_DSDH.py'))
        sys.argv = [test_file, '--fast_dev=1', '--accelerator=cpu', '--num_workers=0',
                    f'--dataset_dir={str(self.dataset_dir)}', f'--logs_dir={str(self.logs_dir)}']
        main()


if __name__ == '__main__':
    unittest.main()
