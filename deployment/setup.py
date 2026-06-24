import os
from pathlib import Path

from setuptools import find_packages, setup


package_name = "navvla"
deployment_dir = Path(__file__).resolve().parent
repo_root = deployment_dir.parent
deployment_weight_files = [
    str(path.relative_to(deployment_dir))
    for path in sorted((deployment_dir / "weights").glob("*"))
    if path.is_file()
]

data_files_by_dest: dict[str, list[str]] = {}
for path in sorted((repo_root / "data").rglob("*")):
    if not path.is_file():
        continue
    dest = f"share/{package_name}/data/{path.parent.relative_to(repo_root / 'data').as_posix()}".rstrip("/.")
    data_files_by_dest.setdefault(dest, []).append(
        os.path.relpath(str(path), str(deployment_dir))
    )


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=["navvla", "navvla.*"]),
    package_data={},
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [os.path.join("resource", package_name)],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (
            f"share/{package_name}/launch",
            [os.path.join("launch", "navigation.launch.py")],
        ),
        (
            f"share/{package_name}/config",
            [
                os.path.join("config", "nav.yaml"),
                os.path.join("config", "preprocess.yaml"),
                os.path.join("config", "pfoe.yaml"),
            ],
        ),
        (
            f"share/{package_name}/deployment/weights",
            deployment_weight_files,
        ),
        *sorted(data_files_by_dest.items()),
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
