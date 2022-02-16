import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from codecs import open

if sys.version_info[:3] < (3, 0, 0):
    print("Requires Python 3 to run.")
    sys.exit(1)

with open("README.md", encoding="utf-8") as file:
    readme = file.read()

setup(
    name="CommunityNet",
    description="Hierarchical GNN for graphs with community structure",
    long_description=readme,
    long_description_content_type="text/markdown",
    version="v1.0.0",
    packages=["communitynet"],
    include_package_data=True,
    python_requires=">=3",
    url="https://github.com/shobrook/CommunityNet",
    author="shobrook",
    author_email="shobrookj@gmail.com",
    # classifiers=[],
    install_requires=["torch", "torch-geometric"],
    keywords=["graph-neural-network", "gnn", "community", "graph", "graph-convolution", "hierarchical"],
    license="MIT"
)
