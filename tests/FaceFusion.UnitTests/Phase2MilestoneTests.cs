using FaceFusion.Core;
using FaceFusion.Media;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// End-to-end test for the docs/DOTNET_PORT_PLAN.md Phase 2 milestone: the port can
/// extract frames from a video, write them back, and remerge with audio — the same round
/// trip <c>facefusion.core</c>'s <c>process_video</c> drives target frame-by-frame, exercised
/// here purely through <see cref="Ffmpeg"/>/<see cref="Ffprobe"/> (no face-swap in between,
/// since that is out of this module's scope) against the real <c>target-240p.mp4</c> example.
///
/// <para>
/// Unlike <c>FfmpegTests</c>/<c>FfprobeTests</c> (which port individual Python test cases
/// 1:1), this test has no Python counterpart — it is a port-only addition pinning the
/// milestone itself: extract → verify frame count/resolution → merge → verify the merged
/// video's own metadata → restore audio → verify the final output is a playable video with
/// an audio stream attached.
/// </para>
/// </summary>
[Collection("MediaOutput")]
public sealed class Phase2MilestoneTests
{
	private static readonly string TempPath = Path.GetTempPath();
	private const string TempFrameFormat = "png";
	private static readonly Resolution ExpectedResolution = new(426, 226);

	public Phase2MilestoneTests()
	{
		MediaFixtures.Ensure();
		TestHelper.PrepareTestOutputDirectory();
	}

	[MediaFact]
	public void ExtractMergeAndRestoreAudioRoundTrip()
	{
		// target-240p.mp4 itself has no audio stream (confirmed via ffprobe — the
		// downloaded example is video-only), so restoring audio needs a target that
		// actually carries one: target-240p-16khz.mp4, a MediaFixtures-generated variant
		// with source.mp3's audio muxed in at the same resolution/fps/duration — this is
		// also the realistic shape of the real pipeline (target_path arrives with its own
		// audio track, which restore_audio reattaches after the video is re-encoded).
		var targetPath = TestHelper.GetTestExampleFile("target-240p-16khz.mp4");
		const int trimFrameStart = 0;
		const int trimFrameEnd = 50; // 2.0s at the source's native 25 fps.
		const double videoFps = 25.0;

		TempHelper.ClearTempDirectory(targetPath, TempPath);
		TempHelper.CreateTempDirectory(targetPath, TempPath);
		try
		{
			// --- 1. Extract frames -----------------------------------------------------

			var extracted = Ffmpeg.ExtractFrames(targetPath, ExpectedResolution, videoFps, trimFrameStart, trimFrameEnd, TempPath, TempFrameFormat, _ => { });
			Assert.True(extracted, "extract_frames failed for the Phase 2 milestone round trip");

			var tempFrameSet = TempHelper.ResolveTempFrameSet(targetPath, TempPath, TempFrameFormat);
			Assert.Equal(trimFrameEnd - trimFrameStart, tempFrameSet.Count);

			using (var firstFrame = Vision.Vision.ReadImage(tempFrameSet[trimFrameStart]))
			{
				Assert.NotNull(firstFrame);
				Assert.Equal(ExpectedResolution.Width, firstFrame!.Cols);
				Assert.Equal(ExpectedResolution.Height, firstFrame.Rows);
			}

			// --- 2. Merge the extracted frames back into a video ------------------------

			var merged = Ffmpeg.MergeVideo(targetPath, videoFps, ExpectedResolution, videoFps, trimFrameStart, trimFrameEnd, VideoEncoder.Libx264, 100, VideoPreset.Ultrafast, TempPath, TempFrameFormat, _ => { });
			Assert.True(merged, "merge_video failed for the Phase 2 milestone round trip");

			var mergedVideoPath = TempHelper.GetTempFilePath(targetPath, TempPath);
			Assert.True(FileSystem.IsFile(mergedVideoPath));

			var mergedMetadata = Ffprobe.ExtractVideoMetadata(mergedVideoPath);
			Assert.Equal(videoFps, mergedMetadata.Fps);
			Assert.Equal(ExpectedResolution, mergedMetadata.Resolution);
			Assert.Equal(trimFrameEnd - trimFrameStart, mergedMetadata.FrameTotal);
			var expectedDuration = (trimFrameEnd - trimFrameStart) / videoFps;
			Assert.True(Math.Abs(mergedMetadata.Duration - expectedDuration) < 0.2, $"expected duration near {expectedDuration}s, got {mergedMetadata.Duration}s");

			// --- 3. Restore audio from the original target into the final output --------

			var outputPath = TestHelper.GetTestOutputFile("phase2-milestone.mp4");
			var restored = Ffmpeg.RestoreAudio(targetPath, outputPath, trimFrameStart, trimFrameEnd, AudioEncoder.Aac, 100, 100, TempPath);
			Assert.True(restored, "restore_audio failed for the Phase 2 milestone round trip");
			Assert.True(FileSystem.IsFile(outputPath));

			var outputVideoMetadata = Ffprobe.ExtractVideoMetadata(outputPath);
			Assert.Equal(ExpectedResolution, outputVideoMetadata.Resolution);
			Assert.True(Math.Abs(outputVideoMetadata.Duration - expectedDuration) < 0.2, $"expected output duration near {expectedDuration}s, got {outputVideoMetadata.Duration}s");

			// The output has an actual audio stream re-attached from the original target
			// (not just a silent/video-only container) — confirmed by ffprobe reporting a
			// real sample_rate for stream a:0.
			var audioEntries = Ffprobe.ProbeAudioEntries(outputPath, new[] { "sample_rate" });
			Assert.True(audioEntries.TryGetValue("sample_rate", out var sampleRate) && int.Parse(sampleRate, System.Globalization.CultureInfo.InvariantCulture) > 0,
				"expected the restored output to carry a real audio stream");
		}
		finally
		{
			TempHelper.ClearTempDirectory(targetPath, TempPath);
		}
	}
}
