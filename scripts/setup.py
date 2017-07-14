from distutils.core import setup

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
    install_requires=['scipy', 'numpy', 'glumpy', 'matplotlib']
)