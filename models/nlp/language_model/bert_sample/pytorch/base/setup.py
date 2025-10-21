import glob
import os
import os.path as ospath

from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

PACKAGE_NAME = "ext_ops"

SOURCE_FILE_EXT = ["c", "cpp", "cu"]
HEADER_FILE_EXT = ["h", "hpp", "cuh"]

SUPPORT_EXTENSIONS = SOURCE_FILE_EXT + HEADER_FILE_EXT


def get_value_from_environ(name: str, default=None):
    if name in os.environ:
        return os.environ[name]
    if name.upper() in os.environ:
        return os.environ[name.upper()]

    return default


def check_source_dir():
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(os.path.dirname(cur_dir), 'iluvatar/csrc')
    return source_dir


def find_source_files() -> dict:
    source_dir = check_source_dir()

    if not ospath.exists(source_dir):
        return dict()

    # Search source files
    sources = dict()
    for ext in SOURCE_FILE_EXT:
        sources[ext] = glob.glob(ospath.join(source_dir, "**", f"*.{ext}"), recursive=True)

    return sources


def find_include_dirs() -> list:
    source_dir = check_source_dir()
    if not ospath.exists(source_dir):
        return []
    return glob.glob(ospath.join(source_dir, "**", "include"), recursive=True)


def get_nvcc_arguments() -> list:
    return [
        '-U__CUDA_NO_HALF_OPERATORS__',
        '-U__CUDA_NO_HALF_CONVERSIONS__',
        "-ftemplate-depth=1024",
    ]


source_files = find_source_files()
include_dirs = find_include_dirs()
c_sources = source_files.pop("c")
other_sources = []
for _sources in source_files.values():
    other_sources.extend(_sources)

nvcc_arguments = get_nvcc_arguments()

ext_modules = []

if len(c_sources) != 0:
    ext_modules.append(Extension(
        name=PACKAGE_NAME,
        sources=c_sources,
        include_dirs=include_dirs,
        extra_compile_args={
            'c': ['-O3']
        }
    ))

if len(other_sources) != 0:
    ext_modules.append(CUDAExtension(
        name=PACKAGE_NAME,
        sources=other_sources,
        extra_compile_args={
            'cxx': ['-O3', ],
            'nvcc': ['-O3'] + nvcc_arguments
        }
    ))

setup(
    name=PACKAGE_NAME,
    version="0.1",
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': BuildExtension
    }
)
