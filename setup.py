from setuptools import setup
import setuptools

with open("README.md") as f:
   readme = f.read()

setup(
     name='differentiable_sorting',  
     packages=setuptools.find_packages(),
     version='0.0.3',                     
     description = 'Differentiable approximate sorting/argsorting with bitonic networks',
     author = 'John H Williamson',
     long_description_content_type="text/markdown",
     long_description=readme,
     author_email = 'johnhw@gmail.com',
     url = 'https://github.com/johnhw/differentiable_sorting', 
    download_url = 'https://github.com/johnhw/differentiable_sorting/tarball/0.1',
    keywords=["sorting", "differentiable", "autograd", "sort", "bitonic"]
 )