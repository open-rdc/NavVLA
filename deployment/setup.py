from setuptools import find_packages, setup


package_name = "navvla"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/navigation.launch.py"]),
        (
            f"share/{package_name}/config",
            [
                "config/nav.yaml",
                "config/preprocess.yaml",
            ],
        ),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="kyo",
    maintainer_email="s21c1135sc@s.chibakoudai.jp",
    description="NavVLA wrappers for OmniVLA training and ROS2 deployment.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "navigation_node = navvla.navigation:main",
        ],
    },
)
