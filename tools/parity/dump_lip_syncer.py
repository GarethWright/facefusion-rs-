"""
Dump ground truth from `facefusion.processors.modules.lip_syncer.core` for the C#
`FaceFusion.Processors.LipSyncer` parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/lip_syncer` (or a scratch directory to inspect
before committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/lip_syncer \
        python3 tools/parity/dump_lip_syncer.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg, source.mp3) and a working internet
connection the first time it runs, since `pre_check()` downloads the lip_syncer/detector/
landmarker models into `.assets/models/` if not already present (edtalk_256 + wav2lip_gan_96
together are well under 100MB).

Kept lean per the assignment: two model families exercise the two distinct lip_syncer
branches `sync_lip` takes (`edtalk` -> a single warped 512x512 crop plus a scalar `weight`
model input; `wav2lip` -> a masked/warped lower-face-bounding-box crop concatenated with
itself into a 6-channel input, no `weight` input, pasted back via a second `warpAffine`).
One image (source.jpg, used as both the "target video frame" and the face-detection source)
and one audio file (source.mp3) keeps this a single-frame, single-face fixture (no video
decode needed to reproduce it) while still exercising the exact same
`select_faces()`/`scale_face()`/`sync_lip()` call sequence `process_frame()` uses.

For each family the highest-value dumps are, in this order, so a mismatch can be localized
to the exact stage that caused it:
  1. `source_voice_frame` - the raw mel-spectrogram slice from `audio.get_voice_frame()`,
     *before* `prepare_audio_frame` touches it. If this alone diverges from the C# port's
     `FaceFusion.Media.Audio.GetVoiceFrame` output, the bug is upstream in the audio layer,
     not in lip_syncer.
  2. `prepared_audio_frame` - `prepare_audio_frame()`'s output, the exact `'source'` model
     input tensor.
  3. `crop_vision_frame` / (`wav2lip` only) `area_vision_frame` - the exact `'target'` model
     input tensor's pre-model-preprocessing source, plus the fully prepared `target_input`
     tensor itself.
  4. `raw_model_output` / `normalized_crop_frame` (or `_area` for wav2lip) - a real ONNX
     Runtime forward pass, checked with a tight tolerance per PARITY_HARNESS.md's "expect ~0"
     guidance for a stage ONNX Runtime's own arithmetic dominates.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import cv2
import numpy

from facefusion import face_classifier, face_detector, face_landmarker, face_masker, face_recognizer, state_manager
from facefusion.audio import get_voice_frame
from facefusion.face_creator import get_static_faces, scale_face
from facefusion.face_helper import create_bounding_box, warp_face_by_bounding_box, warp_face_by_face_landmark_5
from facefusion.face_masker import create_box_mask
from facefusion.face_selector import select_faces
from facefusion.processors.modules.lip_syncer import core as lip_syncer
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')
SOURCE_AUDIO = os.path.join(EXAMPLES_DIR, 'source.mp3')
FPS = 25.0


def init_state(lip_syncer_model: str, lip_syncer_weight: float) -> None:
	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('face_detector_model', 'yolo_face'),
		('face_detector_size', '640x640'),
		('face_detector_score', 0.5),
		('face_detector_angles', [0]),
		('face_detector_margin', [0, 0, 0, 0]),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('face_landmarker_model', 'many'),
		('face_landmarker_score', 0.0),
		('face_selector_mode', 'reference'),
		('face_selector_order', 'large-small'),
		('face_selector_gender', None),
		('face_selector_race', None),
		('face_selector_age_start', None),
		('face_selector_age_end', None),
		('reference_face_position', 0),
		('reference_face_distance', 0.6),
		('face_tracker_score', 0.0),
		('face_mask_types', ['box']),
		('face_mask_blur', 0.3),
		('face_mask_padding', (0, 0, 0, 0)),
		('lip_syncer_model', lip_syncer_model),
		('lip_syncer_weight', lip_syncer_weight),
		('voice_extractor_model', 'kim_vocal_2'),
	]:
		state_manager.init_item(key, value)


def dump_family(family_name: str, lip_syncer_model: str, lip_syncer_weight: float) -> None:
	init_state(lip_syncer_model, lip_syncer_weight)

	if not lip_syncer.pre_check():
		raise SystemExit(f'lip_syncer.pre_check() failed to fetch models for {lip_syncer_model}.')

	temp_vision_frame = cv2.imread(SOURCE_IMAGE)

	# reference frame == source frame == target frame keeps this a single-image fixture (no
	# video decode needed to reproduce it) while still exercising the exact same
	# select_faces()/scale_face()/sync_lip() call sequence process_frame() uses.
	target_faces = select_faces(temp_vision_frame, [temp_vision_frame], [temp_vision_frame])
	target_face = target_faces[0]
	target_face = scale_face(target_face, temp_vision_frame, temp_vision_frame)
	dump(f'{family_name}/target_landmark_5_of_68', target_face.landmark_set.get('5/68'))
	dump(f'{family_name}/target_landmark_68', target_face.landmark_set.get('68'))

	source_voice_frame = get_voice_frame(SOURCE_AUDIO, FPS, frame_number=10)
	dump(f'{family_name}/source_voice_frame', source_voice_frame)

	model_options = lip_syncer.get_model_options()
	model_type = model_options.get('type')
	model_size = model_options.get('size')
	dump_scalar(f'{family_name}/model_type', model_type)
	dump_scalar(f'{family_name}/model_size', list(model_size))

	prepared_audio_frame = lip_syncer.prepare_audio_frame(source_voice_frame)
	dump(f'{family_name}/prepared_audio_frame', prepared_audio_frame)

	crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(temp_vision_frame, target_face.landmark_set.get('5/68'), 'ffhq_512', (512, 512))
	dump(f'{family_name}/crop_vision_frame', crop_vision_frame)

	box_mask = create_box_mask(crop_vision_frame, 0.3, (0, 0, 0, 0))
	dump(f'{family_name}/box_mask', box_mask)

	inference_pool = lip_syncer.get_inference_pool()
	lip_syncer_session = inference_pool.get('lip_syncer')

	if model_type == 'edtalk':
		lip_syncer_weight_array = numpy.array([lip_syncer_weight]).astype(numpy.float32)
		dump(f'{family_name}/weight_input', lip_syncer_weight_array)

		target_input = lip_syncer.prepare_crop_frame(crop_vision_frame)
		dump(f'{family_name}/target_input', target_input)

		session_inputs = {
			'source': prepared_audio_frame,
			'target': target_input,
			'weight': lip_syncer_weight_array,
		}
		raw_output = lip_syncer_session.run(None, session_inputs)[0]
		dump(f'{family_name}/raw_model_output', raw_output[0])

		normalized = lip_syncer.normalize_crop_frame(raw_output)
		dump(f'{family_name}/normalized_crop_frame', normalized)

	elif model_type == 'wav2lip':
		face_landmark_68 = cv2.transform(target_face.landmark_set.get('68').reshape(1, -1, 2), affine_matrix).reshape(-1, 2)
		dump(f'{family_name}/transformed_landmark_68', face_landmark_68)

		bounding_box = create_bounding_box(face_landmark_68)
		dump(f'{family_name}/bounding_box', bounding_box)

		area_vision_frame, area_matrix = warp_face_by_bounding_box(crop_vision_frame, bounding_box, model_size)
		dump(f'{family_name}/area_vision_frame', area_vision_frame)

		target_input = lip_syncer.prepare_crop_frame(area_vision_frame)
		dump(f'{family_name}/target_input', target_input)

		session_inputs = {
			'source': prepared_audio_frame,
			'target': target_input,
		}
		raw_output = lip_syncer_session.run(None, session_inputs)[0]
		dump(f'{family_name}/raw_model_output', raw_output[0])

		normalized = lip_syncer.normalize_crop_frame(raw_output)
		dump(f'{family_name}/normalized_area_vision_frame', normalized)

	else:
		raise SystemExit(f'dump_lip_syncer.py only supports edtalk/wav2lip, got {model_type}')


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	dump_family('edtalk_256', 'edtalk_256', 0.5)
	dump_family('wav2lip_gan_96', 'wav2lip_gan_96', 0.3)


if __name__ == '__main__':
	main()
