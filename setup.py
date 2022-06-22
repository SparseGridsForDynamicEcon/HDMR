from setuptools import setup, dist

with open("README.md", "r") as rd:
  long_description = rd.read()

setup(
    name='HDMR',
    version='0.0.1',
    packages=['DDSG', 'IRBC'],
    description='High-Dimensional Dynamic Stochastic Model Representation',
    url='https://github.com/SparseGridsForDynamicEcon/HDMR',
    author='Aryan Eftekhari, Simon Scheidegger',
    author_email='aryan.eftekhari@unil.ch, simon.scheidegger@unil.ch',
    license='MIT',
    install_requires=['mpi4py',
                      'numpy',
                      'scipy',
                      'matplotlib',
                      'tabulate',
                      'Tasmanian',
                      'dill',
                      ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
