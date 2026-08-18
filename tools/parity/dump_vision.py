"""
Dump ground truth from the real `facefusion.vision` for the C# Vision parity tests.

See docs/PARITY_HARNESS.md and docs/IMPLEMENTATION_STATUS.md ("Open divergences in the
Vision port"). Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/vision` (or a scratch directory to inspect before
committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/vision \
        python3 tools/parity/dump_vision.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.jpg, target-240p.mp4, target-1080p.mp4).
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from facefusion import vision
from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_IMAGE = os.path.join(EXAMPLES_DIR, 'source.jpg')
TARGET_240P = os.path.join(EXAMPLES_DIR, 'target-240p.mp4')
TARGET_1080P = os.path.join(EXAMPLES_DIR, 'target-1080p.mp4')


def dump_video_metadata() -> None:
	for name, video_path in (('target_240p', TARGET_240P), ('target_1080p', TARGET_1080P)):
		resolution = vision.detect_video_resolution(video_path)
		fps = vision.detect_video_fps(video_path)
		duration = vision.detect_video_duration(video_path)
		frame_total = vision.count_video_frame_total(video_path)

		dump_scalar(f'video/{name}/resolution', list(resolution))
		dump_scalar(f'video/{name}/fps', fps)
		dump_scalar(f'video/{name}/duration', duration)
		dump_scalar(f'video/{name}/frame_total', frame_total)


def dump_predict_video_frame_total() -> None:
	cases = [
		{ 'fps': 25.0, 'trim_frame_start': 0, 'trim_frame_end': 270 },
		{ 'fps': 30.0, 'trim_frame_start': 0, 'trim_frame_end': 270 },
		{ 'fps': 12.5, 'trim_frame_start': 10, 'trim_frame_end': 100 },
		{ 'fps': 25.0, 'trim_frame_start': 50, 'trim_frame_end': 60 },
		{ 'fps': 60.0, 'trim_frame_start': 0, 'trim_frame_end': 1 },
	]
	results = []

	for case in cases:
		frame_total = vision.predict_video_frame_total(TARGET_240P, case['fps'], case['trim_frame_start'], case['trim_frame_end'])
		results.append({ **case, 'result': frame_total })

	dump_scalar('video/predict_video_frame_total', results)


def dump_restrict_trim_frame() -> None:
	cases = [
		{ 'trim_frame_start': None, 'trim_frame_end': None },
		{ 'trim_frame_start': 10, 'trim_frame_end': None },
		{ 'trim_frame_start': None, 'trim_frame_end': 50 },
		{ 'trim_frame_start': 10, 'trim_frame_end': 50 },
		{ 'trim_frame_start': -5, 'trim_frame_end': 1000 },
		{ 'trim_frame_start': 100, 'trim_frame_end': 50 },
	]
	results = []

	for case in cases:
		trim_frame_start, trim_frame_end = vision.restrict_trim_frame(TARGET_240P, case['trim_frame_start'], case['trim_frame_end'])
		results.append({ **case, 'result': [ trim_frame_start, trim_frame_end ] })

	dump_scalar('video/restrict_trim_frame', results)


def dump_image_resolution() -> None:
	resolution = vision.detect_image_resolution(SOURCE_IMAGE)
	dump_scalar('image/source_resolution', list(resolution))


def dump_resolution_pack_unpack() -> None:
	pack_cases = [ (0, 0), (1, 1), (426, 226), (427, 227), (1920, 1081), (2048, 1080) ]
	pack_results = [ { 'input': list(case), 'packed': vision.pack_resolution(case) } for case in pack_cases ]
	dump_scalar('resolution/pack_resolution', pack_results)

	unpack_cases = [ '426x226', '1x1', '2048x1080', '0x0' ]
	unpack_results = [ { 'input': case, 'output': list(vision.unpack_resolution(case)) } for case in unpack_cases ]
	dump_scalar('resolution/unpack_resolution', unpack_results)


def dump_read_static_image() -> None:
	image = vision.read_static_image(SOURCE_IMAGE)
	dump('image/source_pixels', image)


FRAME_NUMBERS = (0, 1, 50, 150, 269)


def dump_read_static_video_frame() -> None:
	for frame_number in FRAME_NUMBERS:
		frame = vision.read_static_video_frame(TARGET_240P, frame_number)
		dump(f'video/frames/target_240p_frame_{frame_number}', frame)


def dump_frame_fit_and_restrict() -> None:
	frame = vision.read_static_video_frame(TARGET_240P, 0)

	for width, height in ((300, 300), (600, 200)):
		dump(f'frame/fit_contain_{width}x{height}', vision.fit_contain_frame(frame, (width, height)))
		dump(f'frame/fit_cover_{width}x{height}', vision.fit_cover_frame(frame, (width, height)))

	# 426x226 source: (200, 100) forces a downscale, (800, 800) is a no-op (frame already fits).
	dump('frame/restrict_200x100', vision.restrict_frame(frame, (200, 100)))
	dump('frame/restrict_800x800', vision.restrict_frame(frame, (800, 800)))


def dump_color_helpers() -> None:
	source_frame = vision.read_static_video_frame(TARGET_240P, 0)
	target_frame = vision.read_static_video_frame(TARGET_240P, 150)

	dump('color/match_frame_color', vision.match_frame_color(source_frame, target_frame))
	dump('color/equalize_frame_color_small', vision.equalize_frame_color(source_frame, target_frame, (16, 16)))
	dump('color/equalize_frame_color_full', vision.equalize_frame_color(source_frame, target_frame, target_frame.shape[:2][::-1]))

	for blend_factor in (0.0, 0.25, 0.75, 1.0):
		dump(f'color/blend_frame_{blend_factor}', vision.blend_frame(source_frame, target_frame, blend_factor))


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	dump_video_metadata()
	dump_predict_video_frame_total()
	dump_restrict_trim_frame()
	dump_image_resolution()
	dump_resolution_pack_unpack()
	dump_read_static_image()
	dump_read_static_video_frame()
	dump_frame_fit_and_restrict()
	dump_color_helpers()


if __name__ == '__main__':
	main()
