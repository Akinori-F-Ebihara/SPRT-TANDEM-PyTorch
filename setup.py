from setuptools import setup, find_packages

setup(
    name="sprt-tandem",
    version="0.1.0",
    license="MIT",
    description="SPRT-TANDEM for sequential density ratio estimation to simultaneously optimize speed and accuracy of early-classification.",
    author="Akinori F. Ebihara",
    author_email="aebihara@nec.com",
    packages=find_packages(),
    include_package_data=True,
    url="/Akinori-F-Ebihara/SPRT-TANDEM-PyTorch",
    keywords=[
        "Sequential Probability Ratio Test" "likelihood ratio",
        "density ratio estimation",
        "early classification",
        "artificial intelligence",
        "machine learning",
    ],
    install_requires=["torch", "torchinfo", "optuna"],
    python_requires=">=3.8",
)
