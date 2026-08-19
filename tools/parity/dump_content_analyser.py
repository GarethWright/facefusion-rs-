"""
Dump ground truth from the real `facefusion.content_analyser` for the C# ContentAnalyser
parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/content_analyser` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/content_analyser \
        python3 tools/parity/dump_content_analyser.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg) and a working internet connection the first
time it runs, since `pre_check()` downloads the nsfw_1/nsfw_2/nsfw_3 ONNX models (~460 MB
total) into `.assets/models/` if they are not already present.

For each of the three sub-models this dumps:
  - <model>/input.npy       - prepare_detect_frame() output, the exact tensor handed to
                               session.run. If this matches the C# port's PrepareDetectFrame
                               output, any further mismatch against the raw-output/score
                               fixtures below is ONNX Runtime's own arithmetic (expect ~0
                               divergence there per PARITY_HARNESS.md); if it does not match,
                               the bug is in .NET preprocessing.
  - <model>/raw_output.npy  - forward_nsfw() output, unmodified (batch dim included).
  - <model>/score.json      - the scalar detection_score each detect_with_nsfw_N derives from
                               the raw output (numpy.max/amax for nsfw_1, a difference of
                               components for nsfw_2/nsfw_3).
  - <model>/is_nsfw.json    - the boolean each detect_with_nsfw_N returns (score > threshold).

Plus, once per image:
  - overall/detect_nsfw.json  - detect_nsfw()'s combined majority-vote boolean.
  - overall/analyse_frame.json - analyse_frame()'s boolean (identical to detect_nsfw() today,
                                  dumped separately in case that changes).

Kept lean per the assignment: one image (source.jpg) drives every case — content_analyser has
no per-family branching the way face_detector does, so there is no equivalent need for a
second case.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy

from facefusion import content_analyser, state_manager
from facefusion.vision import read_static_image
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')

MODEL_NAMES = ('nsfw_1', 'nsfw_2', 'nsfw_3')


def init_state() -> None:
	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
	]:
		state_manager.init_item(key, value)


def compute_score(model_name: str, detection) -> float:
	detection = numpy.asarray(detection)

	if model_name == 'nsfw_1':
		return float(numpy.max(numpy.amax(detection[:, 4:], axis=1)))
	if model_name == 'nsfw_2':
		return float(detection[0] - detection[1])
	if model_name == 'nsfw_3':
		return float((detection[2] + detection[3]) - (detection[0] + detection[1]))
	raise ValueError(model_name)


def compute_is_nsfw(model_name: str, score: float) -> bool:
	threshold = {'nsfw_1': 0.2, 'nsfw_2': 0.25, 'nsfw_3': 10.5}[model_name]
	return bool(score > threshold)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	if not os.path.isfile(SOURCE_IMAGE):
		raise SystemExit(f'Missing example image: {SOURCE_IMAGE} (run tools/parity/fetch_examples.sh)')

	init_state()

	if not content_analyser.pre_check():
		raise SystemExit('content_analyser.pre_check() failed to fetch the nsfw models.')

	source_frame = read_static_image(SOURCE_IMAGE)

	for model_name in MODEL_NAMES:
		detect_frame = content_analyser.prepare_detect_frame(source_frame, model_name)
		dump(f'{model_name}/input', detect_frame)

		raw_detection = content_analyser.forward_nsfw(detect_frame, model_name)
		dump(f'{model_name}/raw_output', raw_detection)

		score = compute_score(model_name, raw_detection)
		dump_scalar(f'{model_name}/score', score)
		dump_scalar(f'{model_name}/is_nsfw', compute_is_nsfw(model_name, score))

	dump_scalar('overall/detect_nsfw', content_analyser.detect_nsfw(source_frame))
	dump_scalar('overall/analyse_frame', content_analyser.analyse_frame(source_frame))


if __name__ == '__main__':
	main()
