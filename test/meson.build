
spiral_byhand_bin = executable('spiral_byhand', 
  files('spiral_byhand.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )



bench_lifted_bin = executable('bench_lifted',
  files('bench_lifted.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )

test('lifted MC consistency', bench_lifted_bin)



bench_speed_bin = executable('bench_speed',
  files('bench_speed.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )

dipolar_ewald_test_bin = executable('dipolar_ewald_test',
  files('dipolar_ewald_test.cpp'),
  dependencies: g_deps,
  link_with: mc_lib,
  include_directories: CMC_inc,
  install: false
  )

test('dipolar Ewald consistency', dipolar_ewald_test_bin)
