#!/usr/bin/python
# -*- coding: UTF-8 -*-
import argparse
import os
from typing import List
from DeviceBridge import DeviceBridgeFactory

def get_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run MatrixMultiplication")
    parser.add_argument("--size", help="size of data", default=1024)
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument("--check", action="store_true", help="check result")
    args = parser.parse_args()
    return args


def get_options(args) -> List[str]:
    ret = []
    if args.size is not None:
        ret.append(f'--size {args.size}')
    if args.check:
        ret.append('--check')
    return ret

if __name__ == '__main__':
    adb = DeviceBridgeFactory.create_device_bridge()
    adb.version()
    device_list = adb.devices()
    print(device_list)
    if not device_list:
        print('No device found')
        exit(1)
    adb.set_device(device_list[0])
    print(adb.get_device())
    adb.push(os.path.join('output', 'MatrixMultiplication'), '/data/local/tmp/')
    adb.shell(['chmod', '777', '/data/local/tmp/MatrixMultiplication'])
    args = get_args()
    options = get_options(args)
    if args.debug:
        for i in range(1, 12):
            adb.shell(['simpleperf stat',
                       '-e cpu-cycles',
                       '-e instructions',
                       '-e task-clock',
                       '-e cpu-clock',
                       '-e context-switches',
                       '-e stalled-cycles-frontend',
                       '-e stalled-cycles-backend',
                       '-e cache-misses',
                       '-e cache-references',
                       '-e L1-dcache-loads',
                       '-e L1-dcache-load-misses',
                       '-e LLC-loads',
                       '-e LLC-load-misses',
                       '-e branch-misses',
                       '-e branch-loads',
                       '-e branch-load-misses',
                       '-e major-faults',
                       '-e minor-faults',
                       '-e page-faults'] +
                      ['/data/local/tmp/MatrixMultiplication', f'--test {i}'] + options)
    else:
        adb.shell(['/data/local/tmp/MatrixMultiplication'] + options)
    