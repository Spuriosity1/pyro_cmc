project('spiral_anneal_CMC', ['cpp', 'c'],
  version: '1.1.0',
  meson_version : '>= 1.5.0',
  default_options : [ 'warning_level=3', 'buildtype=release', 'cpp_std=c++2a' ]
)


json_dep = dependency('nlohmann_json', required: true)

latlib_dep = dependency('lattice_lib', version: '>= 2.2.0')

fftw_dep = dependency('fftw3', required: true)

hdf5_dep = dependency('hdf5', required: true, language: 'cpp')

cc=meson.get_compiler('c')

# Detect whether HDF5 was built with MPI support
if cc.compiles('''
    #include <hdf5.h>
    int main() { return 0; }
''',
    name: 'Check for parallel HDF5',
    dependencies: hdf5_dep
)

    message('Serial HDF5 detected')
else
 # suspect MPI is needed
    # Get MPI headers only (no linking)
    mpi_dep = dependency('mpi', required: true, include_type: 'system')

    # Combine deps so your target automatically gets mpi.h
    hdf5_dep = declare_dependency(
        dependencies: [hdf5_dep, mpi_dep]
    )
    message('parallel HDF5 detected')
endif   




CMC_inc=[
  include_directories(['include'])
  ]

g_deps = [
    latlib_dep,
    json_dep,
    fftw_dep,
    hdf5_dep
    ]


mc_lib = static_library('cmc_lib',
  files('src/MC.cpp'),
  dependencies: g_deps,
  include_directories: CMC_inc
  )

fieldcool_bin = executable('fieldcool', 
  files('src/fieldcool.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc
  )

anneal_bin = executable('anneal', 
  files('src/anneal.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc
  )

eigen_dep = dependency('eigen3', required: true, include_type: 'system')

ltgs_bin = executable('ltgs',
  files('src/luttinger_tisza.cpp'),
  dependencies: g_deps + [eigen_dep],
  link_with: mc_lib,
  include_directories: CMC_inc
  )

subdir('test')

