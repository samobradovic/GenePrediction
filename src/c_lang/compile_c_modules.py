import logging
import os

import dvc.api
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

if __name__ == "__main__":
    params = dvc.api.params_show(stages="compile_c_modules")
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # We could iterate over the params["paths"]["c_code_folder"] to find all C files
    # and create extensions for each of them, but for now we will just use kmers.c.
    kmers_extension = Extension(
        "kmers",
        sources=[os.path.join(params["paths"]["c_lang_folder"], "kmers.c")],
    )

    class CustomBuildExtension(build_ext):
        def finalize_options(self):
            super().finalize_options()
            self.build_lib = os.path.abspath(params["paths"]["c_lang_build_folder"])

    setup(
        name="c_extensions",
        version="1.0",
        description="This package contains C extensions"
        " for speedier execution of common tasks.",
        ext_modules=[kmers_extension],
        cmdclass={"build_ext": CustomBuildExtension},
    )
