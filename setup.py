from setuptools import setup, find_packages

# setup(
#     name='MMF-SV',
#     version='1.0.0',
#     packages=find_packages(),
#     url='https://github.com/fritzsedlazeck/Sniffles',
#     license='MIT',
#     author='Zeyu Xia',
#     author_email='xiazeyu12@nudt.edu.cn',
#     description='A fast structural variation caller for long-read sequencing data',
#     long_description=open('README.md').read(),
#     long_description_content_type='text/markdown',
# )

setup(
    name='MMF-SV',
    version='1.0.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    url='https://github.com/fritzsedlazeck/Sniffles',
    license='MIT',
    author='Zeyu Xia',
    author_email='xiazeyu12@nudt.edu.cn',
    description='A fast structural variation caller for long-read sequencing data',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    scripts=[
        'src/mmfsv/mmfsv',
        'src/mmfsv/mmfsv-pre',
        'src/mmfsv/mmfsv-image',
        'src/mmfsv/mmfsv-train',
        'src/mmfsv/mmfsv-filter'
    ],
    install_requires=[
        'pandas'
    ],
)
