from setuptools import find_packages, setup


setup(
    name="stable_diffusion_reference_only",
    version="0.1.1",
    description="Zero-threshold fast secondary painting using stable diffusion reference only",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="stable diffusion reference only automatic coloring painting",
    license="Apache",
    author="aihao",
    author_email="aihao2000@outlook.com",
    url="https://github.com/aihao2000/stable-diffusion-reference-only",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.8.0",
    install_requires=["diffusers==0.21.0", "huggingface_hub==0.23.0","transformers==4.33.1", "numpy"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
