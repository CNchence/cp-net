from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

from sys import platform as _platform

import numpy

include_dirs = [numpy.get_include()]
extra_compile_args=['-O3', '-std=c++11']
extra_link_args=['-std=c++11']

if _platform =='linux' or _platform=='linux2':
    include_dirs.append('/usr/include/eigen3')
elif _platform=='darwin':
    include_dirs.append('/usr/local/include/eigen3')
    extra_compile_args.append('-stdlib=libc++')
    extra_link_args.append('-stdlib=libc++')

ext_modules = [Extension('cp_net.utils.model_base_ransac_estimation',
                         ["cp_net/utils/model_base_ransac_estimation.pyx"],
                         include_dirs = include_dirs,
                         extra_compile_args=extra_compile_args,
                         extra_link_args=extra_link_args,
                         language = 'c++')]
                         # extra_compile_args=['-O3', '-std=c++11', '-stdlib=libc++'])]

setup(
    name='cp_net',
    version='1.0',
    description='Center pose proposal network for pose estimation',
    # long_description=open('README.md').read(),
    author='Yusuke Oshiro',
    author_email='oshiroy0501@gmail.com',
    url='https://github.com/oshiroy/cp-net.git',
    packages=['cp_net'],
    license='The MIT License',
    install_requires=['Cython >= 0.20.1',
                      'scipy', 'numpy', 'glumpy', 'matplotlib'],
    # cython build
    # include_dirs = [numpy.get_include()],
    # cmdclass = {'build_ext': build_ext},
    ext_modules = cythonize(ext_modules)
    # ext_modules=cythonize('cp_net/utils/model_base_ransac_estimation.pyx')
    #                       # language="c++",),
)
