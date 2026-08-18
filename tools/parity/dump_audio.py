"""
Dump ground truth from the real `scipy.signal` primitives, `facefusion.audio`, and
`facefusion.voice_extractor` for the C# Audio/VoiceExtractor parity tests.

See docs/PARITY_HARNESS.md. Run with `FACEFUSION_PARITY_DIR` pointing at
`tests/FaceFusion.ParityTests/fixtures/audio` (or a scratch directory to inspect before
committing):

    FACEFUSION_PARITY_DIR=tests/FaceFusion.ParityTests/fixtures/audio \
        python3 tools/parity/dump_audio.py

Requires the example media fetched by `tools/parity/fetch_examples.sh` into
`/tmp/facefusion-test-examples/` (source.mp3 / source.wav) and, for the voice_extractor
section, a working internet connection the first time it runs (pre_check() downloads the
~64 MB kim_vocal_2 ONNX model into `.assets/models/` if not already present) plus ffmpeg on
PATH.

Sections:
  - primitives/*        - scipy.signal building blocks (lfilter, hann sym/periodic, triang,
                           resample, stft, istft) on small seeded synthetic arrays, checked
                           with rtol=atol=0 against the hand-written C# FFT/STFT/ISTFT/lfilter.
  - audio/*              - facefusion.audio end-to-end: read_static_audio()'s mel-spectrogram
                            frames for source.wav at fps=25, plus the raw prepare_audio() /
                            create_spectrogram() intermediate stages so a mismatch can be
                            localized to a specific stage.
  - voice_extractor/*    - facefusion.voice_extractor end-to-end on a short slice of
                            source.wav: the exact ONNX model input/output tensor pair (checked
                            with rtol=atol=0 against C#, since ORT does the arithmetic there)
                            plus the final extracted voice waveform.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy
import scipy.signal

from tools.parity.parity_dump import dump, dump_enabled, dump_scalar

EXAMPLES_DIR = '/tmp/facefusion-test-examples'
SOURCE_WAV = os.path.join(EXAMPLES_DIR, 'source.wav')
SOURCE_MP3 = os.path.join(EXAMPLES_DIR, 'source.mp3')


def dump_primitives() -> None:
	rng = numpy.random.default_rng(20260818)

	# lfilter (pre-emphasis coefficients, the only ones actually used)
	x64 = rng.standard_normal(4096)
	y = scipy.signal.lfilter([1.0, -0.97], [1.0], x64)
	dump('primitives/lfilter/x', x64)
	dump('primitives/lfilter/y', y)

	# hann windows: symmetric (voice_extractor.py's explicit array) and periodic (audio.py's
	# implicit default via the 'hann' string passed to stft/istft)
	dump('primitives/hann_sym_8', scipy.signal.windows.hann(8))
	dump('primitives/hann_sym_7680', scipy.signal.windows.hann(7680))
	dump('primitives/hann_periodic_8', scipy.signal.windows.hann(8, sym=False))
	dump('primitives/hann_periodic_800', scipy.signal.get_window('hann', 800))

	# triang (mel filter bank), both parities
	dump('primitives/triang_5', scipy.signal.windows.triang(5))
	dump('primitives/triang_4', scipy.signal.windows.triang(4))
	dump('primitives/triang_37', scipy.signal.windows.triang(37))

	# resample (up- and down-sampling, odd and even lengths)
	x_small = rng.standard_normal(3000)
	dump('primitives/resample/x_3000', x_small)
	dump('primitives/resample/up_1000', scipy.signal.resample(x_small, 1000))
	dump('primitives/resample/down_2731', scipy.signal.resample(x_small, 2731))
	dump('primitives/resample/factor_16k', scipy.signal.resample(x_small, round(len(x_small) * 16000 / 48000)))

	# stft (audio.py convention: default periodic hann, nperseg=nfft=800, noverlap=600)
	x_stft = rng.standard_normal(3000)
	dump('primitives/stft/x', x_stft)
	_, _, z = scipy.signal.stft(x_stft, nperseg=800, nfft=800, noverlap=600)
	dump('primitives/stft/real', numpy.real(z))
	dump('primitives/stft/imag', numpy.imag(z))

	# stft/istft (voice_extractor.py convention: explicit symmetric hann, nperseg=7680,
	# noverlap=6656, round-tripped to check the normalisation/scaling)
	x_roundtrip = rng.standard_normal(23040).astype(numpy.float32)
	window = scipy.signal.windows.hann(7680)
	_, _, z2 = scipy.signal.stft(x_roundtrip, nperseg=7680, noverlap=6656, window=window)
	dump('primitives/stft_voice/x', x_roundtrip)
	dump('primitives/stft_voice/real', numpy.real(z2))
	dump('primitives/stft_voice/imag', numpy.imag(z2))
	_, xr = scipy.signal.istft(z2, nperseg=7680, noverlap=6656, window=window)
	dump('primitives/istft_voice/x', xr)


def init_state() -> None:
	from facefusion import state_manager

	for key, value in [
		('download_providers', ['github']),
		('execution_providers', ['cpu']),
		('execution_device_ids', [0]),
		('log_level', 'error'),
		('video_memory_strategy', 'tolerant'),
		('voice_extractor_model', 'kim_vocal_2'),
	]:
		state_manager.init_item(key, value)


def dump_audio_pipeline() -> None:
	from facefusion import ffmpeg
	from facefusion.audio import create_mel_filter_bank, create_spectrogram, extract_audio_frames, prepare_audio, read_static_audio

	audio_buffer = ffmpeg.read_audio_buffer(SOURCE_WAV, 48000, 16, 2)
	raw_audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)
	dump('audio/raw_int16', raw_audio)

	prepared = prepare_audio(raw_audio)
	dump('audio/prepared', prepared)

	mel_filter_bank = create_mel_filter_bank()
	dump('audio/mel_filter_bank', mel_filter_bank)

	spectrogram = create_spectrogram(prepared)
	dump('audio/spectrogram', spectrogram)

	audio_frames = extract_audio_frames(spectrogram, 25)
	dump_scalar('audio/frame_total', len(audio_frames))
	dump('audio/frame_0', audio_frames[0])
	dump('audio/frame_10', audio_frames[10])
	dump('audio/frame_last', audio_frames[-1])

	frames = read_static_audio(SOURCE_WAV, 25)
	dump_scalar('audio/read_static_audio_frame_total', len(frames))
	dump_scalar('audio/read_static_audio_frame_shape', list(frames[0].shape))


def dump_voice_extractor_pipeline() -> None:
	from facefusion import ffmpeg, voice_extractor

	if not voice_extractor.pre_check():
		print('voice_extractor.pre_check() failed to fetch the kim_vocal_2 model; skipping voice_extractor dumps.')
		return

	audio_buffer = ffmpeg.read_audio_buffer(SOURCE_WAV, 48000, 16, 2)
	raw_audio = numpy.frombuffer(audio_buffer, dtype=numpy.int16).reshape(-1, 2)

	# A short slice keeps the model-input dump small (one chunk only) while still exercising
	# the full prepare/decompose/forward/compose/normalize pipeline end to end.
	short_audio = raw_audio[:48000 * 3, ...]
	dump('voice_extractor/input_int16', short_audio)

	temp_audio_chunk = short_audio.T
	voice_chunk_size = (voice_extractor.get_inference_pool().get('kim_vocal_2').get_inputs()[0].shape[3] - 1) * 1024
	voice_trim_size = 3840

	prepared_chunk, audio_pad_size = voice_extractor.prepare_audio_chunk(temp_audio_chunk, voice_chunk_size, voice_trim_size)
	dump('voice_extractor/prepared_chunk', prepared_chunk)
	dump_scalar('voice_extractor/audio_pad_size', int(audio_pad_size))

	decomposed = voice_extractor.decompose_audio_chunk(prepared_chunk, voice_trim_size)
	dump('voice_extractor/model_input', decomposed)

	model_output = voice_extractor.forward(decomposed)
	# Full tensor is ~13 MB uncompressed (~11 MB gzipped, float32 audio compresses poorly) —
	# a small representative slice is enough to prove the OrtValue/RunSession wiring parses
	# ONNX Runtime's output correctly; the *composed*/*normalized*/*extract_voice_output*
	# dumps below exercise the full tensor's values indirectly (istft/normalize run over
	# all of it), so nothing downstream loses coverage by slicing this one.
	dump('voice_extractor/model_output_slice', model_output[:, :, :8, :])
	dump_scalar('voice_extractor/model_output_shape', list(model_output.shape))

	composed = voice_extractor.compose_audio_chunk(model_output, voice_trim_size)
	dump('voice_extractor/composed', composed)

	normalized = voice_extractor.normalize_audio_chunk(composed, voice_chunk_size, voice_trim_size, audio_pad_size)
	dump('voice_extractor/normalized', normalized)

	extracted = voice_extractor.extract_voice(short_audio)
	dump('voice_extractor/extract_voice_output', extracted)

	batch_extracted = voice_extractor.batch_extract_voice(short_audio, 240 * 1024, 180 * 1024)
	dump('voice_extractor/batch_extract_voice_output', batch_extracted)


def main() -> None:
	if not dump_enabled():
		raise SystemExit('Set FACEFUSION_PARITY_DIR before running this script.')

	dump_primitives()

	if os.path.exists(SOURCE_WAV):
		init_state()
		dump_audio_pipeline()
		dump_voice_extractor_pipeline()
	else:
		print(f'{SOURCE_WAV} not found; skipping audio/voice_extractor pipeline dumps (primitives dumped anyway).')


if __name__ == '__main__':
	main()
