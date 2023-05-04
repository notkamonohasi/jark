from setuptools import setup, find_packages
import os
from setuptools import setup
from setuptools.command.build import build

import os
from setuptools import setup, find_packages

class CustomBuild(build):
    def run(self):
        super().run()
        os.makedirs(self.build_platlib, exist_ok=True)
setup(
    name="master",
    version="1.0.0",
    packages=find_packages(),
    cmdclass={'build': CustomBuild },
    install_requires=[
        'matplotlib==3.4.2',
        'ipykernel==6.19.4',
        'pandas==1.5.2',
        'openpyxl==3.0.10',
        'xlrd==2.0.1',
        'numpy==1.24.1',
    ],
)
