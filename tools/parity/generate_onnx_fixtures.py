"""
Generate minimal .onnx fixtures plus a manifest for the .NET ModelHelper / OnnxProtoReader
tests (FaceFusion.Inference).

facefusion/model_helper.py's get_static_model_initializer does exactly one thing:
    onnx.numpy_helper.to_array(onnx.load(model_path).graph.initializer[-1])

These fixtures are built with the real `onnx` package (onnx.helper.make_tensor /
make_graph / make_model) so the committed expected values in manifest.json come from
`onnx.numpy_helper.to_array` itself — the actual behaviour being ported — not from a
hand-rolled re-implementation of the spec. Fixtures are still committed (rather than
generated at test time) so the .NET tests run without Python/onnx present, matching the
pattern in generate_fixtures.py.

Regenerate with:

    python3 tools/parity/generate_onnx_fixtures.py

Covers:
  - multiple initializers per graph, to assert the LAST one wins
  - both raw_data (raw=True / numpy_helper.from_array) and typed-repeated-field
    (raw=False) encodings
  - FLOAT, DOUBLE, INT64, INT32, UINT8, INT8, FLOAT16 (every data_type ModelHelper.cs
    supports)

A few edge cases the `onnx` package cannot easily produce on purpose (unpacked repeated
scalar fields, external_data, a TensorProto.segment split, a truncated file) are instead
hand-built directly in ModelHelperTests.cs — see the comments there.
"""

import json
import os
import shutil

import numpy
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper

FIXTURES_DIRECTORY = os.path.join('tests', 'FaceFusion.UnitTests', 'fixtures', 'onnx')

# onnx.TensorProto.DataType -> the DType name FaceFusion.Inference.OnnxTensor uses.
DTYPE_NAMES = {
	onnx.TensorProto.FLOAT: 'float32',
	onnx.TensorProto.UINT8: 'uint8',
	onnx.TensorProto.INT8: 'int8',
	onnx.TensorProto.INT32: 'int32',
	onnx.TensorProto.INT64: 'int64',
	onnx.TensorProto.FLOAT16: 'float16',
	onnx.TensorProto.DOUBLE: 'double',
}


def make_model_with_initializers(initializers):
	# No nodes/inputs/outputs are needed: get_static_model_initializer never looks at
	# them, and omitting them keeps the fixtures tiny.
	graph = helper.make_graph([], 'g', [], [], initializer=initializers)
	model = helper.make_model(graph, opset_imports=[ helper.make_opsetid('', 17) ])
	onnx.checker.check_model(model)
	return model


def build_cases():
	cases = []

	# --- single initializer, raw_data encoding, one file per supported dtype ---
	raw_specs = [
		('single_float32_raw', onnx.TensorProto.FLOAT, numpy.array([ 1.5, -2.25, 3.0, 0.0, 100.125, -7.75 ], dtype = numpy.float32).reshape(2, 3)),
		('single_double_raw', onnx.TensorProto.DOUBLE, numpy.array([ 1.0, 2.5, -3.75, 4.0 ], dtype = numpy.float64).reshape(2, 2)),
		('single_int64_raw', onnx.TensorProto.INT64, numpy.array([ 100, -200, 300, -400 ], dtype = numpy.int64).reshape(2, 2)),
		('single_int32_raw', onnx.TensorProto.INT32, numpy.array([ -10, 20, -30 ], dtype = numpy.int32)),
		('single_uint8_raw', onnx.TensorProto.UINT8, numpy.array([ 0, 128, 255, 17 ], dtype = numpy.uint8).reshape(2, 2)),
		('single_int8_raw', onnx.TensorProto.INT8, numpy.array([ -128, -1, 0, 127 ], dtype = numpy.int8).reshape(2, 2)),
		('single_float16_raw', onnx.TensorProto.FLOAT16, numpy.array([ 1.5, -2.25, 0.0, 65504.0 ], dtype = numpy.float16).reshape(2, 2)),
	]

	for name, data_type, array in raw_specs:
		tensor = numpy_helper.from_array(array, name = 'w')
		assert tensor.data_type == data_type
		assert tensor.HasField('raw_data')
		cases.append((name, make_model_with_initializers([ tensor ]), tensor))

	# --- single initializer, typed-repeated-field encoding (raw = False) ---
	typed_specs = [
		('single_float32_typed', onnx.TensorProto.FLOAT, [ 2, 3 ], [ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 ]),
		('single_double_typed', onnx.TensorProto.DOUBLE, [ 2 ], [ 3.5, -4.5 ]),
		('single_int64_typed', onnx.TensorProto.INT64, [ 2 ], [ 3, 4 ]),
		('single_int32_typed', onnx.TensorProto.INT32, [ 3 ], [ -10, 20, -30 ]),
		('single_uint8_typed', onnx.TensorProto.UINT8, [ 4 ], [ 10, 20, 30, 255 ]),
		('single_int8_typed', onnx.TensorProto.INT8, [ 3 ], [ -5, 0, 127 ]),
		('single_float16_typed', onnx.TensorProto.FLOAT16, [ 3 ], [ 1.5, -2.25, 0.0 ]),
	]

	for name, data_type, dims, vals in typed_specs:
		tensor = helper.make_tensor(name = 'w', data_type = data_type, dims = dims, vals = vals, raw = False)
		assert not tensor.HasField('raw_data')
		cases.append((name, make_model_with_initializers([ tensor ]), tensor))

	# --- multiple initializers: assert the LAST one is what comes back, in both
	#     encodings and mixing dtypes so a naive "first initializer" or "biggest
	#     initializer" bug would fail these ---
	first_raw = numpy_helper.from_array(numpy.array([ 1.0, 2.0 ], dtype = numpy.float32), name = 'first')
	middle_typed = helper.make_tensor('middle', onnx.TensorProto.INT32, [ 2 ], [ 7, 8 ], raw = False)
	last_raw = numpy_helper.from_array(numpy.array([ [ 9, -9 ], [ 8, -8 ] ], dtype = numpy.int64), name = 'last')
	multi_raw_model = make_model_with_initializers([ first_raw, middle_typed, last_raw ])
	cases.append(('multi_last_is_int64_raw', multi_raw_model, last_raw))

	first_typed = helper.make_tensor('first', onnx.TensorProto.UINT8, [ 2 ], [ 1, 2 ], raw = False)
	middle_raw = numpy_helper.from_array(numpy.array([ 1.0, 2.0, 3.0 ], dtype = numpy.float64), name = 'middle')
	last_typed = helper.make_tensor('last', onnx.TensorProto.FLOAT, [ 2, 2 ], [ 1.25, -2.5, 3.75, -4.0 ], raw = False)
	multi_typed_model = make_model_with_initializers([ first_typed, middle_raw, last_typed ])
	cases.append(('multi_last_is_float32_typed', multi_typed_model, last_typed))

	# A model_helper.py-shaped case: several initializers where the last is a
	# realistic-looking "conversion matrix" (float32, non-square, raw_data), the shape
	# of thing face_swapper actually pulls out.
	embedding_a = numpy_helper.from_array(numpy.arange(4, dtype = numpy.float32), name = 'unrelated_a')
	embedding_b = helper.make_tensor('unrelated_b', onnx.TensorProto.INT64, [ 3 ], [ 1, 2, 3 ], raw = False)
	conversion_matrix = numpy_helper.from_array(
		(numpy.arange(24, dtype = numpy.float32).reshape(4, 6) * 0.1) - 1.0,
		name = 'conversion_matrix')
	conversion_model = make_model_with_initializers([ embedding_a, embedding_b, conversion_matrix ])
	cases.append(('face_swapper_like_conversion_matrix', conversion_model, conversion_matrix))

	return cases


def tensor_to_manifest_entry(reference_tensor):
	array = numpy_helper.to_array(reference_tensor)
	data_type = reference_tensor.data_type
	dtype_name = DTYPE_NAMES[data_type]

	if dtype_name in ('float32', 'double', 'float16'):
		values = [ float(v) for v in array.reshape(-1).tolist() ]
	else:
		values = [ int(v) for v in array.reshape(-1).tolist() ]

	return {
		'shape': list(array.shape),
		'dtype': dtype_name,
		'values': values,
	}


def main():
	if os.path.exists(FIXTURES_DIRECTORY):
		shutil.rmtree(FIXTURES_DIRECTORY)
	os.makedirs(FIXTURES_DIRECTORY)

	manifest = {}

	for name, model, reference_tensor in build_cases():
		file_path = os.path.join(FIXTURES_DIRECTORY, name + '.onnx')
		onnx.save(model, file_path)

		# Round-trip through onnx.load, exactly like model_helper.py does, so the
		# manifest reflects what a real load-and-decode produces rather than the
		# in-memory tensor we built.
		loaded = onnx.load(file_path)
		loaded_last = loaded.graph.initializer[-1]
		manifest[name] = tensor_to_manifest_entry(loaded_last)

	manifest_path = os.path.join(FIXTURES_DIRECTORY, 'manifest.json')
	with open(manifest_path, 'w') as manifest_file:
		json.dump(manifest, manifest_file, indent = '\t', sort_keys = True)
		manifest_file.write('\n')

	print(f'Wrote {len(manifest)} ONNX fixtures + manifest to {FIXTURES_DIRECTORY}')


if __name__ == '__main__':
	main()
