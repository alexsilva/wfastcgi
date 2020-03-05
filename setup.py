import codecs
import os

import pip
from setuptools import setup

BASE_DIR = os.path.dirname(os.path.basename(__file__))

_pip_version = pip.__version__

if _pip_version < "10.0.0":
    from pip.req import parse_requirements
    from pip.download import PipSession
elif _pip_version < "20.0.0":
    from pip._internal.req import parse_requirements
    from pip._internal.req import PipSession
else:
    from pip._internal.req import parse_requirements
    from pip._internal.network.session import PipSession


def parse(filepath, links=False):
    """Returns a list of strings with the requirments registered in the file"""
    requirements = []
    for lib in parse_requirements(filepath, session=PipSession()):
        if links:
            if hasattr(lib.link, 'url'):
                requirements.append(lib.link.url)
        elif lib.req is not None:
            requirements.append(str(lib.req))
    return requirements


with codecs.open(os.path.join(BASE_DIR, 'README.rst'),
                 encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='wfastcgi',
    version='3.3.4',

    description='An IIS-Python bridge based on WSGI and FastCGI.',
    long_description=long_description,
    url='http://aka.ms/python',

    # Author details
    author='Microsoft Corporation',
    author_email='ptvshelp@microsoft.com',
    license='Apache License 2.0',
    dependency_links=parse(os.path.join(BASE_DIR, 'requirements.txt'),
                           links=True),
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 6 - Mature',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Topic :: Internet',
        'Topic :: Internet :: WWW/HTTP',
        'Topic :: Internet :: WWW/HTTP :: WSGI',
        'Topic :: Internet :: WWW/HTTP :: WSGI :: Server',
    ],

    keywords='iis fastcgi wsgi windows server mod_python',
    py_modules=['wfastcgi'],
    install_requires=parse(os.path.join(BASE_DIR, 'requirements.txt')),
    entry_points={
        # 'console_scripts': [
        #     'wfastcgi = wfastcgi:main',
        #     'wfastcgi-enable = wfastcgi:enable',
        #     'wfastcgi-disable = wfastcgi:disable',
        # ]
    },
)
