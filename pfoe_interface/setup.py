from setuptools import setup

package_name = 'pfoe_interface'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kyo',
    maintainer_email='s21c1135sc@s.chibakoudai.jp',
    description='RViz interactive marker interface for PFoE particle operations.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'pf_marker_node = pfoe_interface.pf_marker_node:main',
        ],
    },
)
