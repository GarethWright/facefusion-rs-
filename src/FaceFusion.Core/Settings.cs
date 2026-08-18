using System;
using FaceFusion.Types;

namespace FaceFusion.Core;

/// <summary>
/// Ported from facefusion/state_manager.py.
///
/// Python keeps a module-level mutable dict (<c>STATE_SET</c>) split into 'cli' and 'ui'
/// contexts, selected at read/write time by <c>app_context.detect_app_context()</c>, and
/// exposes <c>get_item</c>/<c>set_item</c>/<c>init_item</c>/<c>sync_item</c>/<c>clear_item</c>
/// free functions that mutate it. Per DOTNET_PORT_PLAN.md §3 ("No global mutable state") that
/// whole design is dropped: <see cref="FaceFusion.Types.State"/> (built by an earlier agent) is
/// already the immutable, DI-friendly replacement — one record with every state key as a
/// property, no global, no cli/ui split (that split exists only to work around Gradio's
/// threading model and has no equivalent need in Blazor's scoped DI).
///
/// This class does not redeclare any of <see cref="State"/>'s fields. It only adds the two
/// operations state_manager.py provided that <see cref="State"/> by itself does not: reading
/// and producing an updated copy of a field by its <see cref="StateKey"/>, for callers that
/// hold a key dynamically (e.g. a future ported <c>job_store</c>/<c>args.py</c> that iterates
/// over a list of keys) rather than a field name known at compile time. Callers that already
/// know the field statically should just use the record's property / a normal
/// <c>state with { Field = value }</c> expression instead of going through this class.
///
/// <c>sync_state</c>/<c>sync_item</c> (copying the 'ui' dict over the 'cli' one) has no
/// equivalent here: it existed solely to reconcile the two contexts, which no longer exist.
/// <c>clear_item</c> (Python: <c>set_item(key, None)</c>) is representable directly for the
/// fields <see cref="State"/> declares nullable (see the nullability note on that record) —
/// callers can pass <c>null</c> to <see cref="WithItem"/> for those keys. It has no equivalent
/// for the remaining, non-nullable fields, matching Python's own consumers, none of which treat
/// <c>None</c> as meaningful for those keys either.
/// </summary>
public static class Settings
{
	/// <summary>
	/// Ported from state_manager.py's <c>get_item</c>. Returns the value stored at
	/// <paramref name="key"/>, boxed as <see cref="object"/> since the value type varies by key
	/// (this mirrors the Python <c>Any</c> return type).
	/// </summary>
	public static object? GetItem(State state, StateKey key)
	{
		return key switch
		{
			StateKey.Command => state.Command,
			StateKey.ConfigPath => state.ConfigPath,
			StateKey.TempPath => state.TempPath,
			StateKey.JobsPath => state.JobsPath,
			StateKey.SourcePaths => state.SourcePaths,
			StateKey.TargetPath => state.TargetPath,
			StateKey.OutputPath => state.OutputPath,
			StateKey.SourcePattern => state.SourcePattern,
			StateKey.TargetPattern => state.TargetPattern,
			StateKey.OutputPattern => state.OutputPattern,
			StateKey.DownloadProviders => state.DownloadProviders,
			StateKey.DownloadScope => state.DownloadScope,
			StateKey.BenchmarkMode => state.BenchmarkMode,
			StateKey.BenchmarkResolutions => state.BenchmarkResolutions,
			StateKey.BenchmarkCycleCount => state.BenchmarkCycleCount,
			StateKey.FaceDetectorModel => state.FaceDetectorModel,
			StateKey.FaceDetectorSize => state.FaceDetectorSize,
			StateKey.FaceDetectorMargin => state.FaceDetectorMargin,
			StateKey.FaceDetectorAngles => state.FaceDetectorAngles,
			StateKey.FaceDetectorScore => state.FaceDetectorScore,
			StateKey.FaceLandmarkerModel => state.FaceLandmarkerModel,
			StateKey.FaceLandmarkerScore => state.FaceLandmarkerScore,
			StateKey.FaceSelectorMode => state.FaceSelectorMode,
			StateKey.FaceSelectorOrder => state.FaceSelectorOrder,
			StateKey.FaceSelectorGender => state.FaceSelectorGender,
			StateKey.FaceSelectorRace => state.FaceSelectorRace,
			StateKey.FaceSelectorAgeStart => state.FaceSelectorAgeStart,
			StateKey.FaceSelectorAgeEnd => state.FaceSelectorAgeEnd,
			StateKey.ReferenceFacePosition => state.ReferenceFacePosition,
			StateKey.ReferenceFaceDistance => state.ReferenceFaceDistance,
			StateKey.ReferenceFrameNumber => state.ReferenceFrameNumber,
			StateKey.FaceTrackerScore => state.FaceTrackerScore,
			StateKey.FaceOccluderModel => state.FaceOccluderModel,
			StateKey.FaceParserModel => state.FaceParserModel,
			StateKey.FaceMaskTypes => state.FaceMaskTypes,
			StateKey.FaceMaskAreas => state.FaceMaskAreas,
			StateKey.FaceMaskRegions => state.FaceMaskRegions,
			StateKey.FaceMaskBlur => state.FaceMaskBlur,
			StateKey.FaceMaskPadding => state.FaceMaskPadding,
			StateKey.VoiceExtractorModel => state.VoiceExtractorModel,
			StateKey.TrimFrameStart => state.TrimFrameStart,
			StateKey.TrimFrameEnd => state.TrimFrameEnd,
			StateKey.TempFrameFormat => state.TempFrameFormat,
			StateKey.TempPixelFormat => state.TempPixelFormat,
			StateKey.TargetFrameAmount => state.TargetFrameAmount,
			StateKey.OutputImageQuality => state.OutputImageQuality,
			StateKey.OutputImageScale => state.OutputImageScale,
			StateKey.OutputAudioEncoder => state.OutputAudioEncoder,
			StateKey.OutputAudioQuality => state.OutputAudioQuality,
			StateKey.OutputAudioVolume => state.OutputAudioVolume,
			StateKey.OutputVideoEncoder => state.OutputVideoEncoder,
			StateKey.OutputVideoPreset => state.OutputVideoPreset,
			StateKey.OutputVideoQuality => state.OutputVideoQuality,
			StateKey.OutputVideoScale => state.OutputVideoScale,
			StateKey.OutputVideoFps => state.OutputVideoFps,
			StateKey.WorkflowMode => state.WorkflowMode,
			StateKey.WorkflowStrategy => state.WorkflowStrategy,
			StateKey.Processors => state.Processors,
			StateKey.OpenBrowser => state.OpenBrowser,
			StateKey.UiLayouts => state.UiLayouts,
			StateKey.UiWorkflow => state.UiWorkflow,
			StateKey.ExecutionDeviceIds => state.ExecutionDeviceIds,
			StateKey.ExecutionProviders => state.ExecutionProviders,
			StateKey.ExecutionThreadCount => state.ExecutionThreadCount,
			StateKey.VideoMemoryStrategy => state.VideoMemoryStrategy,
			StateKey.LogLevel => state.LogLevel,
			StateKey.HaltOnError => state.HaltOnError,
			StateKey.JobId => state.JobId,
			StateKey.JobStatus => state.JobStatus,
			StateKey.StepIndex => state.StepIndex,
			_ => throw new ArgumentOutOfRangeException(nameof(key), key, "Unknown StateKey.")
		};
	}

	/// <summary>
	/// Ported from state_manager.py's <c>set_item</c>/<c>init_item</c>, reshaped for
	/// immutability: instead of mutating a global dict in place, returns a new
	/// <see cref="State"/> with the field named by <paramref name="key"/> replaced by
	/// <paramref name="value"/> (a <c>with</c>-expression per field). <paramref name="value"/>
	/// must be assignable to that field's type or an <see cref="InvalidCastException"/> is
	/// thrown, mirroring the way Python would raise deep in the caller when a wrongly-typed
	/// value reached, say, a string method.
	/// </summary>
	public static State WithItem(State state, StateKey key, object? value)
	{
		return key switch
		{
			StateKey.Command => state with { Command = (string)value! },
			StateKey.ConfigPath => state with { ConfigPath = (string)value! },
			StateKey.TempPath => state with { TempPath = (string)value! },
			StateKey.JobsPath => state with { JobsPath = (string)value! },
			StateKey.SourcePaths => state with { SourcePaths = (System.Collections.Generic.IReadOnlyList<string>?)value },
			StateKey.TargetPath => state with { TargetPath = (string?)value },
			StateKey.OutputPath => state with { OutputPath = (string?)value },
			StateKey.SourcePattern => state with { SourcePattern = (string?)value },
			StateKey.TargetPattern => state with { TargetPattern = (string?)value },
			StateKey.OutputPattern => state with { OutputPattern = (string?)value },
			StateKey.DownloadProviders => state with { DownloadProviders = (System.Collections.Generic.IReadOnlyList<DownloadProvider>)value! },
			StateKey.DownloadScope => state with { DownloadScope = (DownloadScope)value! },
			StateKey.BenchmarkMode => state with { BenchmarkMode = (BenchmarkMode)value! },
			StateKey.BenchmarkResolutions => state with { BenchmarkResolutions = (System.Collections.Generic.IReadOnlyList<BenchmarkResolution>)value! },
			StateKey.BenchmarkCycleCount => state with { BenchmarkCycleCount = (int)value! },
			StateKey.FaceDetectorModel => state with { FaceDetectorModel = (FaceDetectorModel)value! },
			StateKey.FaceDetectorSize => state with { FaceDetectorSize = (string)value! },
			StateKey.FaceDetectorMargin => state with { FaceDetectorMargin = (Margin)value! },
			StateKey.FaceDetectorAngles => state with { FaceDetectorAngles = (System.Collections.Generic.IReadOnlyList<int>)value! },
			StateKey.FaceDetectorScore => state with { FaceDetectorScore = (double)value! },
			StateKey.FaceLandmarkerModel => state with { FaceLandmarkerModel = (FaceLandmarkerModel)value! },
			StateKey.FaceLandmarkerScore => state with { FaceLandmarkerScore = (double)value! },
			StateKey.FaceSelectorMode => state with { FaceSelectorMode = (FaceSelectorMode)value! },
			StateKey.FaceSelectorOrder => state with { FaceSelectorOrder = (FaceSelectorOrder)value! },
			StateKey.FaceSelectorGender => state with { FaceSelectorGender = (FaceSelectorGender?)value },
			StateKey.FaceSelectorRace => state with { FaceSelectorRace = (FaceSelectorRace?)value },
			StateKey.FaceSelectorAgeStart => state with { FaceSelectorAgeStart = (int?)value },
			StateKey.FaceSelectorAgeEnd => state with { FaceSelectorAgeEnd = (int?)value },
			StateKey.ReferenceFacePosition => state with { ReferenceFacePosition = (int)value! },
			StateKey.ReferenceFaceDistance => state with { ReferenceFaceDistance = (double)value! },
			StateKey.ReferenceFrameNumber => state with { ReferenceFrameNumber = (int)value! },
			StateKey.FaceTrackerScore => state with { FaceTrackerScore = (double)value! },
			StateKey.FaceOccluderModel => state with { FaceOccluderModel = (FaceOccluderModel)value! },
			StateKey.FaceParserModel => state with { FaceParserModel = (FaceParserModel)value! },
			StateKey.FaceMaskTypes => state with { FaceMaskTypes = (System.Collections.Generic.IReadOnlyList<FaceMaskType>)value! },
			StateKey.FaceMaskAreas => state with { FaceMaskAreas = (System.Collections.Generic.IReadOnlyList<FaceMaskArea>)value! },
			StateKey.FaceMaskRegions => state with { FaceMaskRegions = (System.Collections.Generic.IReadOnlyList<FaceMaskRegion>)value! },
			StateKey.FaceMaskBlur => state with { FaceMaskBlur = (double)value! },
			StateKey.FaceMaskPadding => state with { FaceMaskPadding = (Padding)value! },
			StateKey.VoiceExtractorModel => state with { VoiceExtractorModel = (VoiceExtractorModel)value! },
			StateKey.TrimFrameStart => state with { TrimFrameStart = (int?)value },
			StateKey.TrimFrameEnd => state with { TrimFrameEnd = (int?)value },
			StateKey.TempFrameFormat => state with { TempFrameFormat = (TempFrameFormat)value! },
			StateKey.TempPixelFormat => state with { TempPixelFormat = (TempPixelFormat)value! },
			StateKey.TargetFrameAmount => state with { TargetFrameAmount = (int)value! },
			StateKey.OutputImageQuality => state with { OutputImageQuality = (int)value! },
			StateKey.OutputImageScale => state with { OutputImageScale = (double)value! },
			StateKey.OutputAudioEncoder => state with { OutputAudioEncoder = (AudioEncoder)value! },
			StateKey.OutputAudioQuality => state with { OutputAudioQuality = (int)value! },
			StateKey.OutputAudioVolume => state with { OutputAudioVolume = (int)value! },
			StateKey.OutputVideoEncoder => state with { OutputVideoEncoder = (VideoEncoder)value! },
			StateKey.OutputVideoPreset => state with { OutputVideoPreset = (VideoPreset)value! },
			StateKey.OutputVideoQuality => state with { OutputVideoQuality = (int)value! },
			StateKey.OutputVideoScale => state with { OutputVideoScale = (double)value! },
			StateKey.OutputVideoFps => state with { OutputVideoFps = (double?)value },
			StateKey.WorkflowMode => state with { WorkflowMode = (WorkflowMode)value! },
			StateKey.WorkflowStrategy => state with { WorkflowStrategy = (WorkflowStrategy)value! },
			StateKey.Processors => state with { Processors = (System.Collections.Generic.IReadOnlyList<string>)value! },
			StateKey.OpenBrowser => state with { OpenBrowser = (bool)value! },
			StateKey.UiLayouts => state with { UiLayouts = (System.Collections.Generic.IReadOnlyList<string>)value! },
			StateKey.UiWorkflow => state with { UiWorkflow = (UiWorkflow)value! },
			StateKey.ExecutionDeviceIds => state with { ExecutionDeviceIds = (System.Collections.Generic.IReadOnlyList<int>)value! },
			StateKey.ExecutionProviders => state with { ExecutionProviders = (System.Collections.Generic.IReadOnlyList<ExecutionProvider>)value! },
			StateKey.ExecutionThreadCount => state with { ExecutionThreadCount = (int)value! },
			StateKey.VideoMemoryStrategy => state with { VideoMemoryStrategy = (VideoMemoryStrategy)value! },
			StateKey.LogLevel => state with { LogLevel = (LogLevel)value! },
			StateKey.HaltOnError => state with { HaltOnError = (bool)value! },
			StateKey.JobId => state with { JobId = (string)value! },
			StateKey.JobStatus => state with { JobStatus = (JobStatus)value! },
			StateKey.StepIndex => state with { StepIndex = (int)value! },
			_ => throw new ArgumentOutOfRangeException(nameof(key), key, "Unknown StateKey.")
		};
	}
}
