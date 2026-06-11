from pathlib import Path

from setuptools import find_packages, setup


package_name = "navvla"
deployment_dir = Path(__file__).resolve().parent
deployment_weight_files = [
    str(path.relative_to(deployment_dir))
    for path in sorted((deployment_dir / "weights").glob("*"))
    if path.is_file()
]
topomap_files = [
    str(path.relative_to(deployment_dir))
    for path in sorted((deployment_dir / "config" / "topomap").glob("*.yaml"))
    if path.is_file()
]
topomap_image_files = [
    str(path.relative_to(deployment_dir))
    for path in sorted((deployment_dir / "config" / "topomap" / "images").glob("*.png"))
    if path.is_file()
]

navvla_packages = find_packages(include=["navvla", "navvla.*"])


setup(
    name=package_name,
    version="0.1.0",
    packages=navvla_packages,
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
        (
            f"share/{package_name}/config/topomap",
            topomap_files,
        ),
        (
            f"share/{package_name}/config/topomap/images",
            topomap_image_files,
        ),
    ],
    install_requires=["setuptools", "numpy", "PyYAML"],
    zip_safe=True,
    maintainer="kyo",
    maintainer_email="s21c1135sc@s.chibakoudai.jp",
    description="NavVLA wrappers for OmniVLA training and ROS2 deployment.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "navigation_node = navvla.navigation:main",
            "create_topomap = navvla.create_topomap:main",
        ],
    },
)
