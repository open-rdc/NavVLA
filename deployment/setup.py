from pathlib import Path

from setuptools import find_packages, setup


package_name = "navvla"
deployment_dir = Path(__file__).resolve().parent
omnivla_dir = (deployment_dir.parent / "OmniVLA").resolve()
deployment_weight_files = [
    str(path.relative_to(deployment_dir))
    for path in sorted((deployment_dir / "weights").glob("*"))
    if path.is_file()
]

navvla_packages = find_packages(include=["navvla", "navvla.*"])
omnivla_packages = ["OmniVLA", "OmniVLA.inference"] + find_packages(
    where=str(omnivla_dir),
    include=["prismatic", "prismatic.*"],
)


setup(
    name=package_name,
    version="0.1.0",
    packages=navvla_packages + omnivla_packages,
    package_dir={
        "OmniVLA": str(omnivla_dir),
        "OmniVLA.inference": str(omnivla_dir / "inference"),
        "prismatic": str(omnivla_dir / "prismatic"),
    },
    package_data={
        "OmniVLA.inference": ["*.jpg"],
        "prismatic": ["py.typed", "vla/datasets/data_config.yaml"],
    },
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
        (
            f"share/{package_name}/OmniVLA/inference",
            ["../OmniVLA/inference/goal_img.jpg"],
        ),
        (
            f"share/{package_name}/deployment/weights",
            deployment_weight_files,
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
