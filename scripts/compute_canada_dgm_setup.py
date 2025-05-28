import os

import setuptools


def read(rel_path):
    base_path = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_path, rel_path), 'r') as f:
        return f.read()


if __name__ == '__main__':
    setuptools.setup(
        name='dgm-eval',
        author='Layer 6',
        description=('Package for evaluating deep generative models'),
        long_description=read('README.md'),
        long_description_content_type='text/markdown',
        packages=['dgm_eval'],
        classifiers=[
            'Programming Language :: Python :: 3',
            'License :: OSI Approved :: Apache Software License',
        ],
        python_requires='>=3.7',
        entry_points={
            'console_scripts': [
                'dgm-eval = dgm_eval:main',
            ],
        },
        install_requires=[
            'numpy==1.26.4',
            'opencv-python==4.9.0',
            'open_clip_torch==2.29.0',
            'pandas==2.2.3',
            'pillow==11.1.0',
            'scikit-image==0.25.1',
            'scikit-learn==1.6.1',
            'scipy==1.15.1',
            'timm==1.0.15',
            'torch>=2.0.0',
            'torchvision>=0.2.2',
            'transformers==4.52.3',
            'xformers==0.0.29.post2',
        ],
        extras_require={'dev': ['flake8',
                                'flake8-bugbear',
                                'flake8-isort',
                                'nox']},
    )