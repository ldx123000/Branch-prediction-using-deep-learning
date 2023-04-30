#!/usr/bin/env python3

import subprocess
import os 

# current_folder = os.getcwd()

build_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/../build/tracer'
tage_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/../src/tracer'

print(__file__)
print(build_dir_path)
os.makedirs(build_dir_path, exist_ok=True)
subprocess.call('cd {build_dir}; cmake {tage_dir} -DCMAKE_BUILD_TYPE=Release; make -j'.format(
    build_dir=build_dir_path,
    tage_dir=tage_dir_path,
), shell=True)