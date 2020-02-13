import setuptools

setuptools.setup(
    name='NeuralSimODE',
    version='1.0',
    description='this package approximate ODE solution by neural network',
    author='Yue Wu',
    author_email='Yue.Wu@uga.edu',
    url='https://github.com/artedison/NeuralSimODE',
    license='MIT',
    packages=['NeuralSimODE'],
    package_dir={'depfinder':'src'},
    install_requires=[
        'coverage',
        'coveralls'
    ])
