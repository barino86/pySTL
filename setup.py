import os
from setuptools import setup, find_packages
here = os.path.abspath(os.path.dirname(__file__))

about = {}
with open(os.path.join(here, 'src', 'anomaly_detection', 'main',  '__version__.py'), 'r') as f:
    exec(f.read(), about)

requirements = []
with open(os.path.join(here, 'requirements.txt'), 'r') as f:
    for line in f.readlines():
        if line.startswith("#"):
            continue
        else:
            requirements.append(line)

setup(
    name=about['__title__'],
    version=about['__version__'],
    description=about['__description__'],
    author=about['__author__'],
    author_email=about['__author_email__'],
    packages=find_packages('src',
                           include=['anomaly*'],
                           exclude=['*.tests', '*.tests.*', 'tests.*', 'tests', 'testlib', '*tests_it*']),
    package_dir={'': 'src'},
    python_requires='~=3.5',
    install_requires=requirements
)
