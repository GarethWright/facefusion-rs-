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

	/// <summary>
	/// Regression coverage for the nullability defect: facefusion/program.py defines several
	/// state keys via `config.get_*_value(section, option)` with NO third (fallback) argument,
	/// which returns None when the ini key is absent. An unset key must read back as null, not
	/// as that type's zero value (0, "", 0.0, first enum member) — the earlier, non-nullable
	/// State shape could not express "unset" and silently substituted a zero value that a
	/// consumer could not distinguish from the user genuinely choosing zero (e.g.
	/// face_selector_age_start = 0, trim_frame_start = 0, output_video_fps = 0.0) or genuinely
	/// choosing 'auto' (face_selector_gender/race, whose first declared member IS 'auto').
	/// </summary>
	[Fact]
	public void TestUnsetOptionalKeysReadBackAsNullNotZero()
	{
		var state = SettingsBuilder.Build(Config.FromText(string.Empty));

		Assert.Null(state.SourcePaths);
		Assert.Null(state.TargetPath);
		Assert.Null(state.OutputPath);
		Assert.Null(state.SourcePattern);
		Assert.Null(state.TargetPattern);
		Assert.Null(state.OutputPattern);
		Assert.Null(state.FaceSelectorGender);
		Assert.Null(state.FaceSelectorRace);
		Assert.Null(state.FaceSelectorAgeStart);
		Assert.Null(state.FaceSelectorAgeEnd);
		Assert.Null(state.TrimFrameStart);
		Assert.Null(state.TrimFrameEnd);
		Assert.Null(state.OutputVideoFps);
	}

	[Fact]
	public void TestOptionalKeysSetToZeroLikeValuesAreDistinguishableFromUnset()
	{
		// Python: state_manager.get_item('face_selector_age_start') or state_manager.get_item('face_selector_age_end')
		// (facefusion/face_selector.py:88) relies on None vs 0 both being distinguishable from a
		// genuine, non-zero setting. Ini-set zero must survive as 0, not collapse to "unset".
		var config = Config.FromText(
			"[face_selector]\n" +
			"face_selector_age_start = 0\n" +
			"\n" +
			"[frame_extraction]\n" +
			"trim_frame_start = 0\n" +
			"\n" +
			"[output_creation]\n" +
			"output_video_fps = 0\n");

		var state = SettingsBuilder.Build(config);

		Assert.NotNull(state.FaceSelectorAgeStart);
		Assert.Equal(0, state.FaceSelectorAgeStart);
		Assert.Null(state.FaceSelectorAgeEnd); // still unset

		Assert.NotNull(state.TrimFrameStart);
		Assert.Equal(0, state.TrimFrameStart);
		Assert.Null(state.TrimFrameEnd); // still unset

		Assert.NotNull(state.OutputVideoFps);
		Assert.Equal(0.0, state.OutputVideoFps);
	}

	[Fact]
	public void TestFaceSelectorGenderUnsetDiffersFromExplicitAuto()
	{
		// Python: facefusion/face_selector.py compares face_selector_gender == 'auto' to decide
		// whether to infer gender from the source face. 'auto' is a real, distinct value from
		// "unset" — unset must not silently behave as if 'auto' had been chosen.
		var unset = SettingsBuilder.Build(Config.FromText(string.Empty));
		var explicitAuto = SettingsBuilder.Build(Config.FromText(
			"[face_selector]\n" +
			"face_selector_gender = auto\n"));

		Assert.Null(unset.FaceSelectorGender);
		Assert.Equal(FaceSelectorGender.Auto, explicitAuto.FaceSelectorGender);
		Assert.NotEqual(unset.FaceSelectorGender, explicitAuto.FaceSelectorGender);
	}

	[Fact]
	public void TestSettingsWithItemAcceptsNullForOptionalKeys()
	{
		var config = Config.FromText(
			"[face_selector]\n" +
			"face_selector_age_start = 21\n");
		var state = SettingsBuilder.Build(config);

		Assert.Equal(21, state.FaceSelectorAgeStart);
		Assert.Equal(21, Settings.GetItem(state, StateKey.FaceSelectorAgeStart));

		var cleared = Settings.WithItem(state, StateKey.FaceSelectorAgeStart, null);

		Assert.Null(cleared.FaceSelectorAgeStart);
		Assert.Null(Settings.GetItem(cleared, StateKey.FaceSelectorAgeStart));
		Assert.Equal(21, state.FaceSelectorAgeStart); // original is untouched (immutability)
	}
}
