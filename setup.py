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
    # 数据文件， 指定需要随Python包一起安装的非Python文件
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


'''
```
    entry_points={
        'console_scripts': [
            'paperchat=main:main',
        ],
    },
```
- console_scripts ：定义命令行工具入口
- paperchat=main:main ：
    - paperchat ：安装后可在命令行直接执行的命令名称
    - main:main ：表示调用main.py文件中的main()函数
    - 格式： 命令名=模块路径:函数名

安装后效果：
    1. 系统会自动生成 paperchat 可执行文件
    2. 在命令行直接输入 paperchat 即可运行程序
    3. 相当于执行 python -m main 命令的快捷方式
    这是Python打包工具setuptools提供的标准功能，用于创建命令行接口(CLI)。


package_data
- 作用：指定需要随Python包一起安装的非Python文件
- 特点：这些文件必须属于某个Python包(paperchat)
- 安装位置：会保持原有目录结构安装到site-packages/paperchat下



data_files
- 作用：指定需要安装到系统全局目录的非包文件
    - 特点：可以安装到任意系统目录
    - 安装位置：
    - Windows: 会安装到Python安装目录的share/paperchat下
    - Linux: 通常安装到/usr/share/paperchat

主要区别：
    - package_data用于包内资源文件
    - data_files用于全局共享文件
    - 两者共同确保项目中的所有必要文件都能正确安装部署- 作用：指定需要安装到系统全局目录的非包文件
    - 特点：可以安装到任意系统目录
    - 安装位置：
    - Windows: 会安装到Python安装目录的share/paperchat下
    - Linux: 通常安装到/usr/share/paperchat
    主要区别：

    - package_data用于包内资源文件
    - data_files用于全局共享文件
    - 两者共同确保项目中的所有必要文件都能正确安装部署
'''
