from setuptools import setup

setup(
    name='splashback_tools',
    description='Package containing code for splashback calculations.',
    url='https://github.com/samgolds/splashback_tools',
    author='Sam Goldstein',
    author_email='samgolds@sas.upenn.edu',
    packages=['splashback_tools',
              'splashback_tools.profiles', 'splashback_tools.utilities'],
)