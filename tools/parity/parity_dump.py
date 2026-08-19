"""
Dump intermediate NumPy arrays from the Python pipeline for .NET parity comparison.

See docs/DOTNET_PORT_PLAN.md section 7. The .npy format is the interchange because it
is trivial to parse, carries dtype/shape/order explicitly, and needs no Python on the
.NET side.

Usage while porting a stage:

    from tools.parity.parity_dump import dump, dump_enabled

    if dump_enabled():
        dump('face_detector/bounding_boxes', bounding_boxes)

Enable with FACEFUSION_PARITY_DIR=/path/to/run. When unset, dump() is a no-op so the
instrumentation can stay in the Python source without affecting normal runs.
"""

import json
import os
from typing import Any, Optional

import numpy


def dump_directory() -> Optional[str]:
	return os.environ.get('FACEFUSION_PARITY_DIR')


def dump_enabled() -> bool:
	return bool(dump_directory())


def dump(name: str, array: Any) -> None:
	"""
	Write `array` as <dump_directory>/<name>.npy. Nested names create directories, so
	'face_detector/boxes' is grouped under a face_detector/ folder.
	"""
	directory_path = dump_directory()

	if not directory_path:
		return

	file_path = os.path.join(directory_path, name + '.npy')
	os.makedirs(os.path.dirname(file_path), exist_ok = True)
	numpy.save(file_path, numpy.asarray(array), allow_pickle = False)


def dump_scalar(name: str, value: Any) -> None:
	"""Record a non-array value (an int, a string, a bool) as JSON alongside the arrays."""
	directory_path = dump_directory()

	if not directory_path:
		return

	file_path = os.path.join(directory_path, name + '.json')
	os.makedirs(os.path.dirname(file_path), exist_ok = True)

	with open(file_path, 'w') as json_file:
		json.dump(value, json_file, indent = 4)
