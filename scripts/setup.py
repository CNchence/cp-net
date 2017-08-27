from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize

import numpy


ext_modules = [Extension('cp_net.utils.model_base_ransac_estimation',
                         ["cp_net/utils/model_base_ransac_estimation.pyx",
                          "cp_net/utils/misc.c"],
                         include_dirs = [numpy.get_include()],
                         extra_compile_args=['-O3'])]
                         # language = 'c++',
                         # extra_link_args=['-std=c++11', '-stdlib=libc++'],
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
