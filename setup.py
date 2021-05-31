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
    name='ppsdg-v6',
    version="2.1.0",
    description='ppsdg project with vantage6 implementation',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/IKNL/v6_boilerplate-py',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        # 'vantage6-client'
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