from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

# Fetch the numpy include directory.
numpy_include_dir = numpy.get_include()

# Define an extension that includes the numpy directory.
extensions = [
    Extension("nms", ["nms.pyx"], include_dirs=[numpy_include_dir]),
]

setup(
    name='nms_module',
    ext_modules=cythonize(extensions),
)
