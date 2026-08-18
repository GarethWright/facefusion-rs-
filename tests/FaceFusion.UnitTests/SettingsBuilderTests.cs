using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Smoke coverage for SettingsBuilder / Settings — there is no Python equivalent test file
/// (state_manager.py has no tests/test_state_manager.py), so these are new tests written for
/// the .NET-only immutable-record replacement described in DOTNET_PORT_PLAN.md §3.
/// </summary>
public class SettingsBuilderTests
{
	[Fact]
	public void TestBuildAppliesHardcodedDefaultsWhenIniIsBlank()
	{
		var config = Config.FromText(
			"[face_detector]\n" +
			"face_detector_model =\n");

		var state = SettingsBuilder.Build(config);

		Assert.Equal(FaceDetectorModel.YoloFace, state.FaceDetectorModel);
		Assert.Equal("640x640", state.FaceDetectorSize);
		Assert.Equal(0.5, state.FaceDetectorScore);
		Assert.Equal(new Margin(0, 0, 0, 0), state.FaceDetectorMargin);
		Assert.Equal(VideoPreset.Veryfast, state.OutputVideoPreset);
		Assert.Equal(WorkflowMode.Auto, state.WorkflowMode);
		Assert.Equal(new[] { "face_swapper" }, state.Processors);
	}

	[Fact]
	public void TestBuildPrefersIniValueOverHardcodedDefault()
	{
		var config = Config.FromText(
			"[face_detector]\n" +
			"face_detector_model = scrfd\n" +
			"face_detector_score = 0.75\n" +
			"\n" +
			"[output_creation]\n" +
			"output_video_preset = slow\n");

		var state = SettingsBuilder.Build(config);

		Assert.Equal(FaceDetectorModel.Scrfd, state.FaceDetectorModel);
		Assert.Equal(0.75, state.FaceDetectorScore);
		Assert.Equal(VideoPreset.Slow, state.OutputVideoPreset);
	}

	[Fact]
	public void TestFaceDetectorSizeFallbackFollowsIniOverriddenModel()
	{
		// Python: face_detector_size's default is get_last(face_detector_set[face_detector_model]),
		// evaluated against the (possibly ini-overridden) model, not the hardcoded one.
		var config = Config.FromText(
			"[face_detector]\n" +
			"face_detector_model = retinaface\n");

		var state = SettingsBuilder.Build(config);

		Assert.Equal(FaceDetectorModel.Retinaface, state.FaceDetectorModel);
		Assert.Equal("640x640", state.FaceDetectorSize);
	}

	[Fact]
	public void TestSettingsGetItemAndWithItemRoundTrip()
	{
		var state = SettingsBuilder.Build(Config.FromText(string.Empty));

		Assert.Equal(state.FaceDetectorScore, Settings.GetItem(state, StateKey.FaceDetectorScore));

		var updated = Settings.WithItem(state, StateKey.FaceDetectorScore, 0.9);

		Assert.Equal(0.9, updated.FaceDetectorScore);
		Assert.Equal(0.5, state.FaceDetectorScore); // original is untouched (immutability)
	}
}
