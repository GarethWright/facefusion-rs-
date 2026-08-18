"""
Generate .npy fixtures plus a manifest for the .NET NpyReader tests.

The fixtures are committed so the parity tests run on CI machines that have no Python
or NumPy. Regenerate with:

    python3 tools/parity/generate_fixtures.py

Every fixture is described in manifest.json with its dtype, shape, Fortran-order flag
and expected values, so a test can assert the reader's output without NumPy present.
"""

import json
import os
import shutil

import numpy

FIXTURES_DIRECTORY = os.path.join('tests', 'FaceFusion.ParityTests', 'fixtures')


def build_cases():
	cases = []

	# Every dtype the pipeline actually moves across the boundary.
	for dtype_name in [ 'float32', 'float64', 'int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64', 'bool' ]:
		array = numpy.arange(6).astype(dtype_name).reshape(2, 3)
		cases.append(('dtype_' + dtype_name, array))

	# Rank coverage, including the 0-d scalar case that trips naive readers.
	cases.append(('rank0_scalar', numpy.array(42.5, dtype = 'float32')))
	cases.append(('rank1_vector', numpy.linspace(0.0, 1.0, 5).astype('float32')))
	cases.append(('rank3_hwc', numpy.arange(24).astype('uint8').reshape(2, 3, 4)))
	cases.append(('rank4_nchw', numpy.arange(16).astype('float32').reshape(1, 2, 2, 4)))

	# Empty arrays: zero elements but a real shape and dtype.
	cases.append(('empty_1d', numpy.array([], dtype = 'float32')))
	cases.append(('empty_2d', numpy.zeros((0, 3), dtype = 'float32')))

	# Fortran ordering - numpy.save records the flag and writes the buffer transposed.
	cases.append(('fortran_order', numpy.asfortranarray(numpy.arange(6).astype('float32').reshape(2, 3))))

	# Non-finite values must survive the round trip exactly, including the sign of zero.
	cases.append(('special_values', numpy.array([ numpy.nan, numpy.inf, -numpy.inf, 0.0, -0.0, 1e-45, 3.4028235e38 ], dtype = 'float32')))

	# Big-endian dtype: the header descr is '>f4' rather than '<f4'.
	cases.append(('big_endian_float32', numpy.arange(4).astype('>f4')))

	# Something shaped like a real frame, to exercise a non-trivial buffer size.
	rng = numpy.random.default_rng(20260818)
	cases.append(('frame_like_bgr', (rng.random((8, 8, 3)) * 255).astype('uint8')))

	return cases


def write_versioned_npy(file_path, array, major, minor):
	"""
	Write a .npy with an explicitly chosen format version.

	numpy.save only emits 2.0/3.0 when the header exceeds 64KB or needs non-latin-1
	characters, which plain float arrays never trigger - but a reader still has to cope
	with the wider header-length field of 2.0 and the utf8 header of 3.0, so those cases
	are constructed by hand rather than left untested.
	"""
	descr = array.dtype.str
	fortran = array.flags.f_contiguous and not array.flags.c_contiguous
	header = "{'descr': '%s', 'fortran_order': %s, 'shape': (%s), }" % (
		descr,
		'True' if fortran else 'False',
		''.join(str(dimension) + ', ' for dimension in array.shape))

	header_bytes = header.encode('utf8')
	length_size = 2 if (major, minor) == (1, 0) else 4
	# The header is padded with spaces and terminated by \n so the data starts on a
	# 64-byte boundary, exactly as numpy.lib.format does.
	prefix_length = 6 + 2 + length_size
	padding = 64 - ((prefix_length + len(header_bytes) + 1) % 64)
	header_bytes = header_bytes + b' ' * padding + b'\n'

	with open(file_path, 'wb') as npy_file:
		npy_file.write(b'\x93NUMPY')
		npy_file.write(bytes([ major, minor ]))
		npy_file.write(len(header_bytes).to_bytes(length_size, 'little'))
		npy_file.write(header_bytes)
		npy_file.write(array.tobytes(order = 'F' if fortran else 'C'))


def describe(array):
	if array.dtype == numpy.bool_:
		values = [ bool(value) for value in array.ravel(order = 'C').tolist() ]
	elif numpy.issubdtype(array.dtype, numpy.integer):
		values = [ int(value) for value in array.ravel(order = 'C').tolist() ]
	else:
		# repr keeps nan/inf/-0.0 unambiguous; the C# side parses these strings.
		values = [ repr(float(value)) for value in array.ravel(order = 'C').tolist() ]

	return \
	{
		'dtype': array.dtype.name,
		'byte_order': array.dtype.byteorder,
		'shape': list(array.shape),
		'fortran_order': bool(array.flags.f_contiguous and not array.flags.c_contiguous),
		'element_count': int(array.size),
		# Values are in C order regardless of how the file stores them: the reader is
		# expected to normalise Fortran-ordered data to C order on load.
		'values_c_order': values
	}


def main() -> None:
	if os.path.isdir(FIXTURES_DIRECTORY):
		shutil.rmtree(FIXTURES_DIRECTORY)
	os.makedirs(FIXTURES_DIRECTORY)

	manifest = {}

	for name, array in build_cases():
		numpy.save(os.path.join(FIXTURES_DIRECTORY, name + '.npy'), array, allow_pickle = False)
		manifest[name] = describe(array)

	# Hand-written format 2.0 and 3.0 headers (see write_versioned_npy).
	for name, major, minor in [ ('format_v2_header', 2, 0), ('format_v3_header', 3, 0) ]:
		array = numpy.arange(6).astype('float32').reshape(2, 3)
		write_versioned_npy(os.path.join(FIXTURES_DIRECTORY, name + '.npy'), array, major, minor)
		manifest[name] = describe(array)
		manifest[name]['npy_version'] = str(major) + '.' + str(minor)

	with open(os.path.join(FIXTURES_DIRECTORY, 'manifest.json'), 'w') as manifest_file:
		json.dump(manifest, manifest_file, indent = 4, sort_keys = True)

	print('numpy', numpy.__version__)
	print('wrote', len(manifest), 'fixtures to', FIXTURES_DIRECTORY)


if __name__ == '__main__':
	main()
