#!/usr/bin/env python
# setup.py
"""
学术论文对话系统安装脚本
"""

import os
from setuptools import setup, find_packages


# 读取README.md
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()


# 读取requirements.txt
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


# 包配置
setup(
    name='paperchat',
    version='1.0.0',
    description='基于Qwen2.5和知识图谱的学术论文对话系统',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='NJUxlj',
    author_email='Zhongyang.Hu23@alumni.xjtlu.edu.cn',
    url='https://github.com/paperchat/paperchat',
    license='MIT',
    python_requires='>=3.10',
    packages=find_packages(),
    include_package_data=True,
    install_requires=read_requirements(),
    entry_points={
        'console_scripts': [
            'paperchat=main:main',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Text Processing :: Linguistic',
        'Topic :: Education',
    ],
    keywords='academic paper, chatbot, knowledge graph, Qwen, natural language processing',
    project_urls={
        'Documentation': 'https://github.com/paperchat/paperchat/docs',
        'Source': 'https://github.com/paperchat/paperchat',
        'Tracker': 'https://github.com/paperchat/paperchat/issues',
    },
    # 数据文件
    package_data={
        'paperchat': [
            'frontend/**/*',
            'src/configs/*.yaml',
            'src/configs/*.json',
        ],
    },
    # 非Python文件
    data_files=[
        ('share/paperchat/examples', ['examples/sample_paper.pdf']),
        ('share/paperchat/docs', ['docs/usage.md', 'docs/api.md']),
    ],
)

