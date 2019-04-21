from distutils.core import setup
from distutils.extension import Extension

bmp_reader = Extension(
    'bmp_reader',
    sources=['bmp_reader.cpp'],
    libraries=['boost_python27-mt'],
)

setup(
    name='bmp-reader',
    version='0.1',
    ext_modules=[bmp_reader])