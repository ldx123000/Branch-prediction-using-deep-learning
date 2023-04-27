#!/usr/bin/env python3

import subprocess
import os 

# build_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/../build/tage'
# tage_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/../src/tage'
build_dir_path = '/home2/dongxu/BranchNet/build/tage'
tage_dir_path = '/home2/dongxu/BranchNet/src/tage'

print(__file__)
print(build_dir_path)
os.makedirs(build_dir_path, exist_ok=True)
subprocess.call('cd {build_dir}; cmake {tage_dir} -DCMAKE_BUILD_TYPE=Release; make -j'.format(
    build_dir=build_dir_path,
    tage_dir=tage_dir_path,
), shell=True)