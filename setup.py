from os import path
from codecs import open
from setuptools import setup, find_packages

# using a README.md, if you do not have this in your folder, simply
# replace this with a string.

# get current directory
here = path.abspath(path.dirname(__file__))

# get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# # read the API version from disk
# with open(path.join(here, 'vantage6', 'tools', 'VERSION')) as fp:
#     __version__ = fp.read()

# setup the package
# Here you specify the meta-local of your package. The `name` argument is
# needed in some other steps.
setup(
    name='v6-ppsdg-py',
    version="2.1.1",
    description='ppsdg project with vantage6 implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/simontkl/torch-vantage6',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=[

    ]
    # ,
    # extras_require={
    # },
    # package_data={
    #     'vantage6.tools': [
    #         'VERSION'
    #     ],
    # }
)