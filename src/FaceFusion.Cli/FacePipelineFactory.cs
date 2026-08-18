using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Cli;

/// <summary>
/// Builds the shared detector/landmarker/recognizer/classifier session set every face-pipeline
/// processor (<c>face_debugger</c>, <c>face_swapper</c>, ...) needs to turn a raw
/// <see cref="Mat"/> into <see cref="Types.Face"/> records, matching Python's own
/// <c>get_many_faces</c>/<c>get_static_faces</c> (each processor module calls the *same*
/// <c>facefusion.face_store</c>-backed cache and the *same* <c>InferencePool</c>s — there is
/// exactly one detector/landmarker/recognizer/classifier session set per process, not one per
/// processor). <see cref="FaceCreator"/> holds no state of its own (see its class remarks), so
/// this class is the CLI-layer owner of that state: it reads every relevant
/// <c>face_detector_*</c>/<c>face_landmarker_*</c> step arg once, opens the sessions those
/// settings need, and hands back delegates (<see cref="Resources.GetStaticFaces"/>/
/// <see cref="Resources.RefillFaces"/>) shaped exactly like the ones
/// <c>FaceSelector.SelectFaces</c>/every processor's <c>*Inputs</c> record already expects —
/// see e.g. <c>FaceDebugger.FaceDebuggerInputs.GetStaticFaces</c>.
///
/// <para>
/// <b>Built once per run, shared across every processor in the step's <c>--processors</c>
/// list</b> — <see cref="ProcessorStepFactory.Build"/> takes an already-built
/// <see cref="Resources"/> instance rather than creating its own, so requesting e.g.
/// <c>face_debugger face_swapper</c> together opens one detector session, not two. Python
/// achieves the same sharing implicitly, via <c>inference_manager</c>'s process-wide pool cache
/// keyed by model name.
/// </para>
/// </summary>
public static class FacePipelineFactory
{
	/// <summary>One built session set plus the config values it was built from. Caller (
	/// <see cref="HeadlessRunner"/>) owns the returned instance and must dispose it once every
	/// processor step built against it has finished running.</summary>
	public sealed class Resources : IDisposable
	{
		public required IReadOnlyDictionary<string, InferenceSession> FaceDetectorSessions { get; init; }

		public required InferenceSession Fan685Session { get; init; }

		public InferenceSession? TwoDFan4Session { get; init; }

		public InferenceSession? PeppaWutzSession { get; init; }

		public required InferenceSession FaceRecognizerSession { get; init; }

		public required InferenceSession FaceClassifierSession { get; init; }

		public required FaceDetectorModel FaceDetectorModel { get; init; }

		public required string FaceDetectorSize { get; init; }

		public required double FaceDetectorScoreThreshold { get; init; }

		public required IReadOnlyList<int> FaceDetectorMargin { get; init; }

		public required IReadOnlyList<int> FaceDetectorAngles { get; init; }

		public required double FaceLandmarkerScoreThreshold { get; init; }

		public required FaceLandmarkerModel FaceLandmarkerModel { get; init; }

		private readonly FaceStore _faceStore = new();

		/// <summary>Python: <c>get_static_faces</c>, bound to this run's session set and its own
		/// process-wide <see cref="FaceStore"/> cache. Matches the
		/// <c>Func&lt;IReadOnlyList&lt;Mat&gt;, IReadOnlyList&lt;Face&gt;&gt;</c> shape every
		/// processor's <c>*Inputs</c> record (e.g. <c>FaceDebuggerInputs.GetStaticFaces</c>)
		/// expects.</summary>
		public IReadOnlyList<Types.Face> GetStaticFaces(IReadOnlyList<Mat> visionFrames)
			=> FaceCreator.GetStaticFaces(
				visionFrames,
				_faceStore,
				FaceDetectorModel,
				FaceDetectorSize,
				FaceDetectorScoreThreshold,
				FaceDetectorMargin,
				FaceDetectorAngles,
				FaceLandmarkerScoreThreshold,
				FaceLandmarkerModel,
				FaceDetectorSessions,
				Fan685Session,
				TwoDFan4Session,
				PeppaWutzSession,
				FaceRecognizerSession,
				FaceClassifierSession);

		/// <summary>Python: <c>refill_faces</c> — stateless, so this just forwards to
		/// <see cref="FaceCreator.RefillFaces"/>.</summary>
		public IReadOnlyList<Types.Face> RefillFaces(IReadOnlyList<Types.Face?> faces)
			=> FaceCreator.RefillFaces(faces);

		public void Dispose()
		{
			foreach (var session in FaceDetectorSessions.Values)
			{
				session.Dispose();
			}

			Fan685Session.Dispose();
			TwoDFan4Session?.Dispose();
			PeppaWutzSession?.Dispose();
			FaceRecognizerSession.Dispose();
			FaceClassifierSession.Dispose();
		}
	}

	/// <summary>True for every processor whose <c>*Inputs</c> record needs
	/// <see cref="Resources"/> — the set this phase wires (see
	/// <see cref="ProcessorStepFactory"/>'s class remarks for which of these are actually built
	/// vs. still <see cref="NotSupportedException"/>).</summary>
	public static bool Requires(string processorName) => processorName switch
	{
		"face_debugger" or "face_swapper" or "age_modifier" or "expression_restorer"
			or "deep_swapper" or "face_editor" or "lip_syncer" or "face_enhancer" => true,
		_ => false,
	};

	/// <summary>Python: the <c>face_detector</c>/<c>face_landmarker</c>-related half of
	/// <c>apply_args</c> plus every <c>get_inference_pool()</c> call those two modules'
	/// <c>pre_process</c> makes — opens every session the chosen models need. Only call once
	/// <see cref="PreCheck"/> has passed for every processor that will use it.</summary>
	public static Resources Build(IReadOnlyDictionary<string, object?> args)
	{
		var modelsDirectory = HeadlessRunner.ResolveModelsDirectory();

		var faceDetectorModel = EnumNames.FromWireName<FaceDetectorModel>(StepArgsReader.GetString(args, "face_detector_model", "yolo_face"));
		var faceDetectorSize = StepArgsReader.GetString(args, "face_detector_size", "640x640");
		var faceDetectorScoreThreshold = StepArgsReader.GetDouble(args, "face_detector_score", 0.5);
		var faceDetectorMargin = StepArgsReader.GetIntList(args, "face_detector_margin", new[] { 0, 0, 0, 0 });
		var faceDetectorAngles = StepArgsReader.GetIntList(args, "face_detector_angles", new[] { 0 });
		var faceLandmarkerScoreThreshold = StepArgsReader.GetDouble(args, "face_landmarker_score", 0.5);
		var faceLandmarkerModel = EnumNames.FromWireName<FaceLandmarkerModel>(StepArgsReader.GetString(args, "face_landmarker_model", "2dfan4"));

		var faceDetectorSessions = FaceDetector.CreateStaticModelSet(DownloadScope.Full);
		var detectorSessions = new Dictionary<string, InferenceSession>();

		foreach (var family in RequiredDetectorFamilies(faceDetectorModel))
		{
			detectorSessions[family.ToWireName()] = new InferenceSession(faceDetectorSessions[family].Source.Path);
		}

		InferenceSession? twoDFan4Session = null;
		InferenceSession? peppaWutzSession = null;

		if (faceLandmarkerScoreThreshold > 0)
		{
			if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.TwoDFan4)
			{
				twoDFan4Session = new InferenceSession(Path.Combine(modelsDirectory, "2dfan4.onnx"));
			}

			if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.PeppaWutz)
			{
				peppaWutzSession = new InferenceSession(Path.Combine(modelsDirectory, "peppa_wutz.onnx"));
			}
		}

		return new Resources
		{
			FaceDetectorSessions = detectorSessions,
			Fan685Session = new InferenceSession(Path.Combine(modelsDirectory, "fan_68_5.onnx")),
			TwoDFan4Session = twoDFan4Session,
			PeppaWutzSession = peppaWutzSession,
			FaceRecognizerSession = new InferenceSession(Path.Combine(modelsDirectory, "arcface_w600k_r50.onnx")),
			FaceClassifierSession = new InferenceSession(Path.Combine(modelsDirectory, "fairface.onnx")),
			FaceDetectorModel = faceDetectorModel,
			FaceDetectorSize = faceDetectorSize,
			FaceDetectorScoreThreshold = faceDetectorScoreThreshold,
			FaceDetectorMargin = faceDetectorMargin,
			FaceDetectorAngles = faceDetectorAngles,
			FaceLandmarkerScoreThreshold = faceLandmarkerScoreThreshold,
			FaceLandmarkerModel = faceLandmarkerModel,
		};
	}

	/// <summary>Cheap, allocation-free presence check for every model file
	/// <see cref="Build"/> would open for the given args — mirrors
	/// <c>ProcessorStepFactory.PreCheck</c>'s own reduced-scope shape.</summary>
	public static bool PreCheck(IReadOnlyDictionary<string, object?> args)
	{
		var modelsDirectory = HeadlessRunner.ResolveModelsDirectory();
		var faceDetectorModel = EnumNames.FromWireName<FaceDetectorModel>(StepArgsReader.GetString(args, "face_detector_model", "yolo_face"));
		var faceLandmarkerScoreThreshold = StepArgsReader.GetDouble(args, "face_landmarker_score", 0.5);
		var faceLandmarkerModel = EnumNames.FromWireName<FaceLandmarkerModel>(StepArgsReader.GetString(args, "face_landmarker_model", "2dfan4"));

		var faceDetectorModelSet = FaceDetector.CreateStaticModelSet(DownloadScope.Full);

		foreach (var family in RequiredDetectorFamilies(faceDetectorModel))
		{
			var (hash, source) = faceDetectorModelSet[family];
			if (!FaceFusion.Core.FileSystem.IsFile(hash.Path) || !FaceFusion.Core.FileSystem.IsFile(source.Path))
			{
				return false;
			}
		}

		var required = new List<string> { "fan_68_5.onnx", "arcface_w600k_r50.onnx", "fairface.onnx" };

		if (faceLandmarkerScoreThreshold > 0)
		{
			if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.TwoDFan4)
			{
				required.Add("2dfan4.onnx");
			}

			if (faceLandmarkerModel is FaceLandmarkerModel.Many or FaceLandmarkerModel.PeppaWutz)
			{
				required.Add("peppa_wutz.onnx");
			}
		}

		return required.All(fileName => FaceFusion.Core.FileSystem.IsFile(Path.Combine(modelsDirectory, fileName)));
	}

	private static IReadOnlyList<FaceDetectorModel> RequiredDetectorFamilies(FaceDetectorModel faceDetectorModel)
		=> faceDetectorModel == FaceDetectorModel.Many
			? new[] { FaceDetectorModel.Retinaface, FaceDetectorModel.Scrfd, FaceDetectorModel.YoloFace }
			: new[] { faceDetectorModel };
}
