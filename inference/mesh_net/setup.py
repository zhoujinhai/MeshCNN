from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension(
        "mesh_net.network", ["network.py"]
    ),
    Extension(
        "mesh_net.mesh", ["mesh.py"]
    ),
    Extension(
        "mesh_net.mesh_process", ["mesh_process.py"]
    ),
    Extension(
        "mesh_net.mesh_union", ["mesh_union.py"]
    )
]

setup(
    name="mesh_net",
    ext_modules=cythonize(extensions)
)