using System.Linq;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_detector.py</c> — the four detector families
/// (<c>retinaface</c>, <c>scrfd</c>, <c>yolo_face</c>, <c>yunet</c>) plus <c>many</c> (run
/// several and merge), and the shared preprocessing/margin/angle-rotation plumbing around
/// them.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python function here reads
/// <c>state_manager.get_item(...)</c> for <c>face_detector_model</c> /
/// <c>face_detector_size</c> / <c>face_detector_score</c> / <c>face_detector_margin</c> /
/// <c>execution_device_ids</c> / <c>execution_providers</c>. Every one of those becomes an
/// explicit parameter here instead — <see cref="DetectFaces"/>,
/// <see cref="DetectFacesByAngle"/> and the four <c>DetectWith*</c> methods all take the
/// already-resolved <see cref="InferenceSession"/>(s) rather than reaching into a pool
/// keyed by ambient state, and <see cref="GetInferencePool"/> below takes the
/// <see cref="InferenceManager"/> instance and the execution settings as parameters instead
/// of calling a module-level singleton.
/// </para>
///
/// <para>
/// <b>Representation choices</b> (per docs/DOTNET_PORT_PLAN.md §4, and matching the
/// conventions already established by <see cref="FaceHelper"/>): <c>VisionFrame</c> stays a
/// caller-owned <see cref="Mat"/> (BGR, <c>CV_8UC3</c> — Python's <c>color_mode = 'rgb'</c> is
/// a misleading name for "not RGBA"; <c>cv2.imread</c>/<c>cv2.IMREAD_COLOR</c> decodes BGR
/// either way, so no channel swap happens anywhere in this file, matching Python exactly). A
/// single <c>BoundingBox</c> is <c>float[]</c> of length 4 (<c>[x1, y1, x2, y2]</c>), a single
/// <c>FaceLandmark5</c> is <c>float[,]</c> of shape <c>(5, 2)</c> — both matching
/// <see cref="FaceHelper"/>'s existing public surface exactly, so results from this class
/// plug straight into <see cref="FaceHelper.NormalizeBoundingBox"/>,
/// <see cref="FaceHelper.TransformBoundingBox"/>, etc. without conversion. <c>Score</c> is
/// <see cref="double"/> per <c>FaceFusion.Types.TypeAliases</c> (widened from the model's
/// float32 output; a plain widening assignment, no precision is manufactured).
/// </para>
///
/// <para>
/// <b>The float32-vs-float64 divergence this file inherits from <see cref="FaceHelper"/>
/// (documented, not introduced here).</b> Python's anchor grid
/// (<c>face_helper.create_static_anchors</c>, built from <c>numpy.mgrid</c>) is int64;
/// combining it with the float32 model output via <c>distance_to_bounding_box</c> /
/// <c>distance_to_face_landmark_5</c> upcasts the result to float64 (mixed-width array/array
/// arithmetic always upcasts, unlike the array/python-scalar case below). This is why the
/// Python fixtures dumped by <c>tools/parity/dump_face_detector.py</c> show float64 bounding
/// boxes/landmarks for retinaface/scrfd/yunet but float32 for yolo_face (which has no anchor
/// step — its box math is ORT float32 output combined only with plain Python float scalars,
/// which numpy's NEP 50 scalar-promotion rules keep at float32). <see cref="FaceHelper.CreateStaticAnchors"/>
/// already returns <c>float[,]</c> (single precision) rather than reproducing the int64
/// anchor grid, so every family's box/landmark decode in this port stays float32 throughout —
/// this is <see cref="FaceHelper"/>'s established, already-tested representation choice, not
/// a new divergence introduced here. Per PARITY_HARNESS.md this lands in the "managed float
/// math ... a real epsilon belongs here" bucket for the anchor-decode families (not the
/// "ORT does the arithmetic, expect ~0" bucket) — see FaceDetectorParityTests for the
/// measured tolerance.
/// </para>
///
/// <para>
/// <b>Model set / <c>pre_check</c> — a deliberately reduced-scope port.</b> Python's
/// <c>pre_check</c> calls <c>conditional_download_hashes</c> / <c>conditional_download_sources</c>
/// (facefusion/download.py), which shell out to <c>curl</c>, ping candidate download
/// providers, and validate a SHA-1 hash file (facefusion/hash_helper.py) against the
/// downloaded model — none of which is ported (out of this module's assignment; download.py
/// and hash_helper.py are substantial modules of their own, and this container's tests must
/// run with no network per PARITY_HARNESS.md). <see cref="PreCheck"/> here instead checks that
/// every hash/source file the requested model needs is already present on disk — the actual
/// download/hash-verification step is left for a future port of <c>download.py</c>. This
/// means <see cref="PreCheck"/> can return <see langword="true"/> for a corrupted or
/// truncated model file where Python's hash check would have caught it and re-downloaded;
/// that gap is accepted and documented rather than silently narrowed. <see cref="CreateStaticModelSet"/>
/// still reproduces the file layout exactly (same file names, same <c>.assets/models</c>
/// directory, same GitHub release URL shape via <c>Choices.DownloadProviderSet</c>) so a
/// caller that does want to add real fetching later has the right paths/URLs to hand.
/// </para>
///
/// <para>
/// <b>The <c>yunet</c>-and-<c>many</c> quirk (reproduced deliberately, per PORT_CONVENTIONS.md
/// rule 1).</b> Python's <c>collect_model_downloads</c> loads/pools <c>yunet</c> whenever
/// <c>face_detector_model</c> is <c>'many'</c> (its loop checks
/// <c>state_manager.get_item('face_detector_model') in ['many', face_detector_model]</c> for
/// all four families uniformly), but <c>detect_faces</c>'s dispatch checks
/// <c>== 'yunet'</c> for the yunet branch specifically (not <c>in ['many', 'yunet']</c> like
/// the other three) — so under <c>'many'</c> the yunet model is downloaded and loaded into
/// the inference pool but never actually run. <see cref="CollectModelDownloads"/> and
/// <see cref="GetInferencePool"/> include yunet under <c>Many</c>; <see cref="DetectFaces"/>'s
/// dispatch excludes it, exactly matching this asymmetry.
/// </para>
/// </summary>
public static class FaceDetector
{
    private const int FeatureMapChannel = 3;
    private static readonly int[] FeatureStrides = { 8, 16, 32 };

    private static readonly IReadOnlyList<FaceDetectorModel> AllFamilies = new[]
    {
        FaceDetectorModel.Retinaface, FaceDetectorModel.Scrfd, FaceDetectorModel.YoloFace, FaceDetectorModel.Yunet,
    };

    private static readonly IReadOnlyDictionary<FaceDetectorModel, string> ModelFileNames = new Dictionary<FaceDetectorModel, string>
    {
        [FaceDetectorModel.Retinaface] = "retinaface_10g",
        [FaceDetectorModel.Scrfd] = "scrfd_2.5g",
        [FaceDetectorModel.YoloFace] = "yoloface_8n",
        [FaceDetectorModel.Yunet] = "yunet_2023_mar",
    };

    // Python: create_static_model_set's 'models-3.0.0' / 'models-3.4.0' resolve_download_url
    // base_name argument, per family.
    private static readonly IReadOnlyDictionary<FaceDetectorModel, string> ModelBaseNames = new Dictionary<FaceDetectorModel, string>
    {
        [FaceDetectorModel.Retinaface] = "models-3.0.0",
        [FaceDetectorModel.Scrfd] = "models-3.0.0",
        [FaceDetectorModel.YoloFace] = "models-3.0.0",
        [FaceDetectorModel.Yunet] = "models-3.4.0",
    };

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>). See the class remarks for
    /// why the download URL is built directly from <see cref="Choices.DownloadProviderSet"/>'s
    /// github entry rather than through the unported <c>resolve_download_url</c> (which pings
    /// each configured provider in turn and picks the first reachable one).
    /// <paramref name="downloadScope"/> is accepted for signature parity with Python — the
    /// dict body there does not vary by scope either, every family has exactly one variant.
    /// </summary>
    public static IReadOnlyDictionary<FaceDetectorModel, (Download Hash, Download Source)> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<FaceDetectorModel, (Download, Download)>();

        foreach (var family in AllFamilies)
        {
            var fileName = ModelFileNames[family];
            var baseName = ModelBaseNames[family];

            var hash = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".hash"),
                Path.Combine(modelsDirectory, fileName + ".hash"));
            var source = new Download(
                BuildDownloadUrl(githubProvider, baseName, fileName + ".onnx"),
                Path.Combine(modelsDirectory, fileName + ".onnx"));

            result[family] = (hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>
    /// Python: <c>resolve_relative_path('../.assets/models')</c> as called from
    /// facefusion/download.py, which resolves relative to the <c>facefusion</c> package
    /// directory's parent (the repository root). <see cref="FaceFusion.Core.FileSystem.ResolveRelativePath"/>
    /// resolves against the .NET build output directory instead, which would not reach the
    /// real <c>.assets/models</c> at the repo root from a test assembly's bin folder — so this
    /// walks up from <see cref="System.AppContext.BaseDirectory"/> looking for the solution
    /// file as the repo-root marker, the same directory that holds <c>.assets</c> in this repo
    /// layout.
    /// </summary>
    private static string ResolveModelsDirectory()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return Path.Combine(directory.FullName, ".assets", "models");
            }

            directory = directory.Parent;
        }

        throw new InvalidOperationException("Could not locate the repository root (FaceFusion.sln) to resolve .assets/models.");
    }

    /// <summary>
    /// Python: <c>collect_model_downloads</c>. Keyed by each family's wire name (e.g.
    /// <c>"retinaface"</c>), matching <see cref="GetInferencePool"/> / the model-name keys
    /// Python's <c>get_inference_pool().get(...)</c> uses. See the class remarks for the
    /// <c>yunet</c>-under-<c>many</c> asymmetry this reproduces.
    /// </summary>
    public static (IReadOnlyDictionary<string, Download> Hashes, IReadOnlyDictionary<string, Download> Sources) CollectModelDownloads(FaceDetectorModel faceDetectorModel)
    {
        var modelSet = CreateStaticModelSet(DownloadScope.Full);
        var hashes = new Dictionary<string, Download>();
        var sources = new Dictionary<string, Download>();

        foreach (var family in AllFamilies)
        {
            if (faceDetectorModel == FaceDetectorModel.Many || faceDetectorModel == family)
            {
                var (hash, source) = modelSet[family];
                hashes[family.ToWireName()] = hash;
                sources[family.ToWireName()] = source;
            }
        }

        return (hashes, sources);
    }

    /// <summary>
    /// Python: <c>pre_check</c>. See the class remarks — this checks file presence only; it
    /// does not download or validate hashes (download.py/hash_helper.py are out of this
    /// module's scope).
    /// </summary>
    public static bool PreCheck(FaceDetectorModel faceDetectorModel)
    {
        var (hashes, sources) = CollectModelDownloads(faceDetectorModel);
        return hashes.Values.All(download => FaceFusion.Core.FileSystem.IsFile(download.Path))
            && sources.Values.All(download => FaceFusion.Core.FileSystem.IsFile(download.Path));
    }

    // -----------------------------------------------------------------
    // Inference pool (thin wrappers around FaceFusion.Inference.InferenceManager)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>get_inference_pool</c>. Takes the pool-owning <see cref="InferenceManager"/>
    /// as a parameter instead of a module-level singleton (PORT_CONVENTIONS.md rule 5).
    /// </summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        FaceDetectorModel faceDetectorModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var (_, modelSourceSet) = CollectModelDownloads(faceDetectorModel);
        var modelNames = new[] { faceDetectorModel.ToWireName() };
        return inferenceManager.GetInferencePool("facefusion.face_detector", modelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        FaceDetectorModel faceDetectorModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { faceDetectorModel.ToWireName() };
        inferenceManager.ClearInferencePool("facefusion.face_detector", modelNames, executionDeviceIds, executionProviders);
    }

    // -----------------------------------------------------------------
    // detect_faces / prepare_margin / detect_faces_by_angle
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>detect_faces</c>. <paramref name="inferenceSessions"/> stands in for
    /// Python's <c>get_inference_pool()</c> lookup — pass the dictionary returned by
    /// <see cref="GetInferencePool"/> (or any dictionary keyed the same way, e.g. by a test's
    /// own directly-loaded <see cref="InferenceSession"/>s). Does not take ownership of
    /// <paramref name="visionFrame"/>.
    /// </summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectFaces(
        Mat visionFrame,
        FaceDetectorModel faceDetectorModel,
        string faceDetectorSize,
        double faceDetectorScore,
        IReadOnlyList<int> faceDetectorMargin,
        IReadOnlyDictionary<string, InferenceSession> inferenceSessions)
    {
        var (marginTop, marginRight, marginBottom, marginLeft) = PrepareMargin(visionFrame, faceDetectorMargin);

        using var marginVisionFrame = new Mat();
        Cv2.CopyMakeBorder(visionFrame, marginVisionFrame, marginTop, marginBottom, marginLeft, marginRight, BorderTypes.Constant, Scalar.All(0));

        var allBoundingBoxes = new List<float[]>();
        var allFaceScores = new List<double>();
        var allFaceLandmarks5 = new List<float[,]>();

        if (faceDetectorModel is FaceDetectorModel.Many or FaceDetectorModel.Retinaface)
        {
            var (boundingBoxes, faceScores, faceLandmarks5) = DetectWithRetinaface(marginVisionFrame, faceDetectorSize, faceDetectorScore, inferenceSessions["retinaface"]);
            allBoundingBoxes.AddRange(boundingBoxes);
            allFaceScores.AddRange(faceScores);
            allFaceLandmarks5.AddRange(faceLandmarks5);
        }

        if (faceDetectorModel is FaceDetectorModel.Many or FaceDetectorModel.Scrfd)
        {
            var (boundingBoxes, faceScores, faceLandmarks5) = DetectWithScrfd(marginVisionFrame, faceDetectorSize, faceDetectorScore, inferenceSessions["scrfd"]);
            allBoundingBoxes.AddRange(boundingBoxes);
            allFaceScores.AddRange(faceScores);
            allFaceLandmarks5.AddRange(faceLandmarks5);
        }

        if (faceDetectorModel is FaceDetectorModel.Many or FaceDetectorModel.YoloFace)
        {
            var (boundingBoxes, faceScores, faceLandmarks5) = DetectWithYoloFace(marginVisionFrame, faceDetectorSize, faceDetectorScore, inferenceSessions["yolo_face"]);
            allBoundingBoxes.AddRange(boundingBoxes);
            allFaceScores.AddRange(faceScores);
            allFaceLandmarks5.AddRange(faceLandmarks5);
        }

        // Python: `if state_manager.get_item('face_detector_model') == 'yunet':` — deliberately
        // NOT `in ['many', 'yunet']` like the three branches above. See the class remarks.
        if (faceDetectorModel == FaceDetectorModel.Yunet)
        {
            var (boundingBoxes, faceScores, faceLandmarks5) = DetectWithYunet(marginVisionFrame, faceDetectorSize, faceDetectorScore, inferenceSessions["yunet"]);
            allBoundingBoxes.AddRange(boundingBoxes);
            allFaceScores.AddRange(faceScores);
            allFaceLandmarks5.AddRange(faceLandmarks5);
        }

        var resultBoundingBoxes = new List<float[]>(allBoundingBoxes.Count);
        foreach (var boundingBox in allBoundingBoxes)
        {
            var normalized = FaceHelper.NormalizeBoundingBox(boundingBox);
            resultBoundingBoxes.Add(new[]
            {
                normalized[0] - marginLeft,
                normalized[1] - marginTop,
                normalized[2] - marginLeft,
                normalized[3] - marginTop,
            });
        }

        var resultFaceLandmarks5 = new List<float[,]>(allFaceLandmarks5.Count);
        foreach (var faceLandmark5 in allFaceLandmarks5)
        {
            var shifted = new float[5, 2];
            for (var i = 0; i < 5; i++)
            {
                shifted[i, 0] = faceLandmark5[i, 0] - marginLeft;
                shifted[i, 1] = faceLandmark5[i, 1] - marginTop;
            }

            resultFaceLandmarks5.Add(shifted);
        }

        return (resultBoundingBoxes, allFaceScores, resultFaceLandmarks5);
    }

    /// <summary>Python: <c>prepare_margin</c>.</summary>
    public static (int Top, int Right, int Bottom, int Left) PrepareMargin(Mat visionFrame, IReadOnlyList<int> faceDetectorMargin)
    {
        var height = visionFrame.Rows;
        var width = visionFrame.Cols;
        var interpXp = new[] { 0f, 100f };
        var interpFp = new[] { 0f, 0.5f };

        // Python: int(shape * numpy.interp(margin, [0, 100], [0, 0.5])) — truncates toward
        // zero, same as a plain (int) cast of a non-negative float here.
        var marginTop = (int)(height * NumPy.Interp(faceDetectorMargin[0], interpXp, interpFp));
        var marginRight = (int)(width * NumPy.Interp(faceDetectorMargin[1], interpXp, interpFp));
        var marginBottom = (int)(height * NumPy.Interp(faceDetectorMargin[2], interpXp, interpFp));
        var marginLeft = (int)(width * NumPy.Interp(faceDetectorMargin[3], interpXp, interpFp));

        return (marginTop, marginRight, marginBottom, marginLeft);
    }

    /// <summary>
    /// Python: <c>detect_faces_by_angle</c>. Does not take ownership of
    /// <paramref name="visionFrame"/>.
    /// </summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectFacesByAngle(
        Mat visionFrame,
        int faceAngle,
        FaceDetectorModel faceDetectorModel,
        string faceDetectorSize,
        double faceDetectorScore,
        IReadOnlyList<int> faceDetectorMargin,
        IReadOnlyDictionary<string, InferenceSession> inferenceSessions)
    {
        var (rotationMatrix, rotationSize) = FaceHelper.CreateRotationMatrixAndSize(faceAngle, new Size(visionFrame.Cols, visionFrame.Rows));

        try
        {
            using var rotationVisionFrame = new Mat();
            Cv2.WarpAffine(visionFrame, rotationVisionFrame, rotationMatrix, rotationSize);

            using var rotationInverseMatrix = new Mat();
            Cv2.InvertAffineTransform(rotationMatrix, rotationInverseMatrix);

            var (boundingBoxes, faceScores, faceLandmarks5) = DetectFaces(rotationVisionFrame, faceDetectorModel, faceDetectorSize, faceDetectorScore, faceDetectorMargin, inferenceSessions);

            var transformedBoundingBoxes = new List<float[]>(boundingBoxes.Count);
            foreach (var boundingBox in boundingBoxes)
            {
                transformedBoundingBoxes.Add(FaceHelper.TransformBoundingBox(boundingBox, rotationInverseMatrix));
            }

            var transformedFaceLandmarks5 = new List<float[,]>(faceLandmarks5.Count);
            foreach (var faceLandmark5 in faceLandmarks5)
            {
                transformedFaceLandmarks5.Add(FaceHelper.TransformPoints(faceLandmark5, rotationInverseMatrix));
            }

            return (transformedBoundingBoxes, faceScores, transformedFaceLandmarks5);
        }
        finally
        {
            rotationMatrix.Dispose();
        }
    }

    // -----------------------------------------------------------------
    // Per-family detection
    // -----------------------------------------------------------------

    /// <summary>Python: <c>detect_with_retinaface</c> (+ inlined <c>forward_with_retinaface</c>).</summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectWithRetinaface(
        Mat visionFrame, string faceDetectorSize, double faceDetectorScore, InferenceSession inferenceSession)
        => DetectWithStrideModel(visionFrame, faceDetectorSize, faceDetectorScore, inferenceSession, anchorTotal: 2, normalizeLow: -1f, normalizeHigh: 1f);

    /// <summary>Python: <c>detect_with_scrfd</c> (+ inlined <c>forward_with_scrfd</c>) — a
    /// byte-for-byte duplicate of <c>detect_with_retinaface</c> in Python, reproduced here as
    /// a second call into the same shared helper rather than a second copy of the body.</summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectWithScrfd(
        Mat visionFrame, string faceDetectorSize, double faceDetectorScore, InferenceSession inferenceSession)
        => DetectWithStrideModel(visionFrame, faceDetectorSize, faceDetectorScore, inferenceSession, anchorTotal: 2, normalizeLow: -1f, normalizeHigh: 1f);

    private static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectWithStrideModel(
        Mat visionFrame, string faceDetectorSize, double faceDetectorScore, InferenceSession inferenceSession, int anchorTotal, float normalizeLow, float normalizeHigh)
    {
        var boundingBoxes = new List<float[]>();
        var faceScores = new List<double>();
        var faceLandmarks5 = new List<float[,]>();

        var faceDetectorResolution = Vision.Vision.UnpackResolution(faceDetectorSize);
        var faceDetectorWidth = faceDetectorResolution.Width;
        var faceDetectorHeight = faceDetectorResolution.Height;

        using var tempVisionFrame = Vision.Vision.RestrictFrame(visionFrame, faceDetectorResolution);
        var ratioHeight = (double)visionFrame.Rows / tempVisionFrame.Rows;
        var ratioWidth = (double)visionFrame.Cols / tempVisionFrame.Cols;

        var detectVisionFrame = PrepareDetectFrame(tempVisionFrame, faceDetectorSize);
        detectVisionFrame = NormalizeDetectFrame(detectVisionFrame, normalizeLow, normalizeHigh);

        var outputs = RunSession(inferenceSession, detectVisionFrame, new long[] { 1, 3, faceDetectorHeight, faceDetectorWidth });

        for (var index = 0; index < FeatureStrides.Length; index++)
        {
            var featureStride = FeatureStrides[index];
            var (scoresData, scoresRows, _) = GetOutput(outputs[index]);

            var keepIndices = new List<int>();
            for (var i = 0; i < scoresRows; i++)
            {
                if (scoresData[i] >= faceDetectorScore)
                {
                    keepIndices.Add(i);
                }
            }

            if (keepIndices.Count == 0)
            {
                continue;
            }

            var strideHeight = faceDetectorHeight / featureStride;
            var strideWidth = faceDetectorWidth / featureStride;
            var anchors = FaceHelper.CreateStaticAnchors(featureStride, anchorTotal, strideHeight, strideWidth);

            var (boundingBoxData, boundingBoxRows, _) = GetOutput(outputs[index + FeatureMapChannel]);
            var boundingBoxesRaw = new float[boundingBoxRows, 4];
            for (var r = 0; r < boundingBoxRows; r++)
            {
                for (var c = 0; c < 4; c++)
                {
                    boundingBoxesRaw[r, c] = boundingBoxData[(r * 4) + c] * featureStride;
                }
            }

            var (landmarkData, landmarkRows, _) = GetOutput(outputs[index + (FeatureMapChannel * 2)]);
            var faceLandmarks5Raw = new float[landmarkRows, 10];
            for (var r = 0; r < landmarkRows; r++)
            {
                for (var c = 0; c < 10; c++)
                {
                    faceLandmarks5Raw[r, c] = landmarkData[(r * 10) + c] * featureStride;
                }
            }

            var decodedBoundingBoxes = FaceHelper.DistanceToBoundingBox(anchors, boundingBoxesRaw);
            var decodedFaceLandmarks5 = FaceHelper.DistanceToFaceLandmark5(anchors, faceLandmarks5Raw);

            foreach (var i in keepIndices)
            {
                boundingBoxes.Add(new[]
                {
                    decodedBoundingBoxes[i, 0] * (float)ratioWidth,
                    decodedBoundingBoxes[i, 1] * (float)ratioHeight,
                    decodedBoundingBoxes[i, 2] * (float)ratioWidth,
                    decodedBoundingBoxes[i, 3] * (float)ratioHeight,
                });

                faceScores.Add(scoresData[i]);

                var landmark = new float[5, 2];
                for (var k = 0; k < 5; k++)
                {
                    landmark[k, 0] = decodedFaceLandmarks5[i, k, 0] * (float)ratioWidth;
                    landmark[k, 1] = decodedFaceLandmarks5[i, k, 1] * (float)ratioHeight;
                }

                faceLandmarks5.Add(landmark);
            }
        }

        return (boundingBoxes, faceScores, faceLandmarks5);
    }

    /// <summary>Python: <c>detect_with_yolo_face</c> (+ inlined <c>forward_with_yolo_face</c>).</summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectWithYoloFace(
        Mat visionFrame, string faceDetectorSize, double faceDetectorScore, InferenceSession inferenceSession)
    {
        var boundingBoxes = new List<float[]>();
        var faceScores = new List<double>();
        var faceLandmarks5 = new List<float[,]>();

        var faceDetectorResolution = Vision.Vision.UnpackResolution(faceDetectorSize);
        var faceDetectorWidth = faceDetectorResolution.Width;
        var faceDetectorHeight = faceDetectorResolution.Height;

        using var tempVisionFrame = Vision.Vision.RestrictFrame(visionFrame, faceDetectorResolution);
        var ratioHeight = (double)visionFrame.Rows / tempVisionFrame.Rows;
        var ratioWidth = (double)visionFrame.Cols / tempVisionFrame.Cols;

        var detectVisionFrame = PrepareDetectFrame(tempVisionFrame, faceDetectorSize);
        detectVisionFrame = NormalizeDetectFrame(detectVisionFrame, 0f, 1f);

        var outputs = RunSession(inferenceSession, detectVisionFrame, new long[] { 1, 3, faceDetectorHeight, faceDetectorWidth });

        // Python: `detection = numpy.squeeze(detection).T` on a (1, 20, 8400) output ->
        // (20, 8400) -> (8400, 20); detection[box, channel] == data[channel * boxTotal + box]
        // in the original (channel-major) row-major layout, so this is read directly from the
        // untransposed buffer rather than materialising the transpose.
        var (data, shape) = outputs[0];
        var channelTotal = (int)shape[1];
        var boxTotal = (int)shape[2];

        for (var box = 0; box < boxTotal; box++)
        {
            var score = data[(4 * boxTotal) + box];

            // Python: `keep_indices = numpy.where(face_scores_raw > face_detector_score)[0]`
            // — strict greater-than, unlike the other three families' `>=`.
            if (!(score > faceDetectorScore))
            {
                continue;
            }

            var cx = data[(0 * boxTotal) + box];
            var cy = data[(1 * boxTotal) + box];
            var w = data[(2 * boxTotal) + box];
            var h = data[(3 * boxTotal) + box];

            boundingBoxes.Add(new[]
            {
                (cx - (w / 2f)) * (float)ratioWidth,
                (cy - (h / 2f)) * (float)ratioHeight,
                (cx + (w / 2f)) * (float)ratioWidth,
                (cy + (h / 2f)) * (float)ratioHeight,
            });

            faceScores.Add(score);

            var landmark = new float[5, 2];
            for (var k = 0; k < 5; k++)
            {
                var channel = 5 + (3 * k);
                landmark[k, 0] = data[(channel * boxTotal) + box] * (float)ratioWidth;
                landmark[k, 1] = data[((channel + 1) * boxTotal) + box] * (float)ratioHeight;
            }

            faceLandmarks5.Add(landmark);

            _ = channelTotal; // channelTotal == 20 == 4 + 1 + 15, asserted implicitly by the fixed offsets above.
        }

        return (boundingBoxes, faceScores, faceLandmarks5);
    }

    /// <summary>Python: <c>detect_with_yunet</c> (+ inlined <c>forward_with_yunet</c>).</summary>
    public static (IReadOnlyList<float[]> BoundingBoxes, IReadOnlyList<double> FaceScores, IReadOnlyList<float[,]> FaceLandmarks5) DetectWithYunet(
        Mat visionFrame, string faceDetectorSize, double faceDetectorScore, InferenceSession inferenceSession)
    {
        const int anchorTotal = 1;

        var boundingBoxes = new List<float[]>();
        var faceScores = new List<double>();
        var faceLandmarks5 = new List<float[,]>();

        var faceDetectorResolution = Vision.Vision.UnpackResolution(faceDetectorSize);
        var faceDetectorWidth = faceDetectorResolution.Width;
        var faceDetectorHeight = faceDetectorResolution.Height;

        using var tempVisionFrame = Vision.Vision.RestrictFrame(visionFrame, faceDetectorResolution);
        var ratioHeight = (double)visionFrame.Rows / tempVisionFrame.Rows;
        var ratioWidth = (double)visionFrame.Cols / tempVisionFrame.Cols;

        var detectVisionFrame = PrepareDetectFrame(tempVisionFrame, faceDetectorSize);
        // Python: normalize_range == [0, 255] falls through normalize_detect_frame's two
        // explicit branches unchanged — the identity case, called here for documentation
        // parity even though NormalizeDetectFrame's identity branch is a plain copy.
        detectVisionFrame = NormalizeDetectFrame(detectVisionFrame, 0f, 255f);

        var outputs = RunSession(inferenceSession, detectVisionFrame, new long[] { 1, 3, faceDetectorHeight, faceDetectorWidth });

        for (var index = 0; index < FeatureStrides.Length; index++)
        {
            var featureStride = FeatureStrides[index];
            var (objScoreData, rowTotal, _) = GetOutput(outputs[index]);
            var (clsScoreData, _, _) = GetOutput(outputs[index + FeatureMapChannel]);

            var faceScoresRaw = new float[rowTotal];
            var keepIndices = new List<int>();
            for (var i = 0; i < rowTotal; i++)
            {
                faceScoresRaw[i] = objScoreData[i] * clsScoreData[i];
                if (faceScoresRaw[i] >= faceDetectorScore)
                {
                    keepIndices.Add(i);
                }
            }

            if (keepIndices.Count == 0)
            {
                continue;
            }

            var strideHeight = faceDetectorHeight / featureStride;
            var strideWidth = faceDetectorWidth / featureStride;
            var anchors = FaceHelper.CreateStaticAnchors(featureStride, anchorTotal, strideHeight, strideWidth);

            var (boundingBoxData, boundingBoxRows, _) = GetOutput(outputs[index + (FeatureMapChannel * 2)]);
            var boundingBoxesRaw = new float[boundingBoxRows, 4];
            for (var r = 0; r < boundingBoxRows; r++)
            {
                var cx = (boundingBoxData[(r * 4) + 0] * featureStride) + anchors[r, 0];
                var cy = (boundingBoxData[(r * 4) + 1] * featureStride) + anchors[r, 1];
                var w = MathF.Exp(boundingBoxData[(r * 4) + 2]) * featureStride;
                var h = MathF.Exp(boundingBoxData[(r * 4) + 3]) * featureStride;

                boundingBoxesRaw[r, 0] = cx - (w / 2f);
                boundingBoxesRaw[r, 1] = cy - (h / 2f);
                boundingBoxesRaw[r, 2] = cx + (w / 2f);
                boundingBoxesRaw[r, 3] = cy + (h / 2f);
            }

            var (landmarkData, landmarkRows, _) = GetOutput(outputs[index + (FeatureMapChannel * 3)]);
            var faceLandmarks5Raw = new float[landmarkRows, 5, 2];
            for (var r = 0; r < landmarkRows; r++)
            {
                for (var k = 0; k < 5; k++)
                {
                    faceLandmarks5Raw[r, k, 0] = (landmarkData[(r * 10) + (2 * k)] * featureStride) + anchors[r, 0];
                    faceLandmarks5Raw[r, k, 1] = (landmarkData[(r * 10) + (2 * k) + 1] * featureStride) + anchors[r, 1];
                }
            }

            foreach (var i in keepIndices)
            {
                boundingBoxes.Add(new[]
                {
                    boundingBoxesRaw[i, 0] * (float)ratioWidth,
                    boundingBoxesRaw[i, 1] * (float)ratioHeight,
                    boundingBoxesRaw[i, 2] * (float)ratioWidth,
                    boundingBoxesRaw[i, 3] * (float)ratioHeight,
                });

                faceScores.Add(faceScoresRaw[i]);

                var landmark = new float[5, 2];
                for (var k = 0; k < 5; k++)
                {
                    landmark[k, 0] = faceLandmarks5Raw[i, k, 0] * (float)ratioWidth;
                    landmark[k, 1] = faceLandmarks5Raw[i, k, 1] * (float)ratioHeight;
                }

                faceLandmarks5.Add(landmark);
            }
        }

        return (boundingBoxes, faceScores, faceLandmarks5);
    }

    // -----------------------------------------------------------------
    // prepare_detect_frame / normalize_detect_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_detect_frame</c>. Returns the model-input tensor in NCHW layout
    /// (batch = 1, implicit — the returned array has length <c>3 * height * width</c>,
    /// matching Python's <c>(1, 3, height, width)</c> array flattened), <b>before</b>
    /// normalization (see <see cref="NormalizeDetectFrame"/>). Per docs/DOTNET_PORT_PLAN.md
    /// §4/§5, model input tensors are plain arrays destined for
    /// <c>OrtValue.CreateTensorValueFromMemory</c>, not a <see cref="Mat"/> — an NCHW buffer
    /// has no OpenCV representation. <paramref name="tempVisionFrame"/> must be
    /// <c>CV_8UC3</c> (every real caller passes the BGR frame straight out of
    /// <see cref="Vision.Vision.RestrictFrame"/>).
    /// </summary>
    public static float[] PrepareDetectFrame(Mat tempVisionFrame, string faceDetectorSize)
    {
        if (tempVisionFrame.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("tempVisionFrame must be CV_8UC3.", nameof(tempVisionFrame));
        }

        var faceDetectorResolution = Vision.Vision.UnpackResolution(faceDetectorSize);
        var faceDetectorWidth = faceDetectorResolution.Width;
        var faceDetectorHeight = faceDetectorResolution.Height;

        // Python: `detect_vision_frame = numpy.zeros((height, width, 3))` then
        // `detect_vision_frame[:h, :w, :] = temp_vision_frame` — a zero-padded HWC buffer with
        // the (possibly smaller) source frame copied into the top-left corner.
        var hwc = new float[faceDetectorHeight * faceDetectorWidth * 3];

        var sourceHeight = tempVisionFrame.Rows;
        var sourceWidth = tempVisionFrame.Cols;
        tempVisionFrame.GetArray(out Vec3b[] pixels);

        for (var h = 0; h < sourceHeight; h++)
        {
            for (var w = 0; w < sourceWidth; w++)
            {
                var pixel = pixels[(h * sourceWidth) + w];
                var destinationOffset = ((h * faceDetectorWidth) + w) * 3;
                hwc[destinationOffset + 0] = pixel.Item0;
                hwc[destinationOffset + 1] = pixel.Item1;
                hwc[destinationOffset + 2] = pixel.Item2;
            }
        }

        // Python: `.transpose(2, 0, 1)` then `numpy.expand_dims(..., axis = 0).astype(float32)`.
        return NumPy.TransposeHwcToChw(hwc, faceDetectorHeight, faceDetectorWidth, 3);
    }

    /// <summary>
    /// Python: <c>normalize_detect_frame</c>. <paramref name="rangeLow"/>/<paramref name="rangeHigh"/>
    /// stand in for Python's <c>normalize_range : Sequence[int]</c> list-equality check
    /// (<c>[-1, 1]</c> or <c>[0, 1]</c>); any other pair (including yunet's <c>[0, 255]</c>)
    /// takes the identity branch, matching Python's <c>return detect_vision_frame</c>
    /// fallthrough exactly. Arithmetic stays float32 throughout (matching numpy's NEP 50
    /// scalar-promotion rule: a float32 array combined with a plain Python float/int scalar —
    /// <c>127.5</c>, <c>128.0</c>, <c>255.0</c> here — does not upcast).
    /// </summary>
    public static float[] NormalizeDetectFrame(float[] detectVisionFrame, float rangeLow, float rangeHigh)
    {
        if (rangeLow == -1f && rangeHigh == 1f)
        {
            var result = new float[detectVisionFrame.Length];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = (detectVisionFrame[i] - 127.5f) / 128f;
            }

            return result;
        }

        if (rangeLow == 0f && rangeHigh == 1f)
        {
            var result = new float[detectVisionFrame.Length];
            for (var i = 0; i < result.Length; i++)
            {
                result[i] = detectVisionFrame[i] / 255f;
            }

            return result;
        }

        return (float[])detectVisionFrame.Clone();
    }

    // -----------------------------------------------------------------
    // ONNX Runtime plumbing (OrtValue zero-copy, per docs/DOTNET_PORT_PLAN.md §5.3)
    // -----------------------------------------------------------------

    /// <summary>
    /// Runs <paramref name="inferenceSession"/> over <paramref name="inputData"/> (NCHW,
    /// matching <see cref="PrepareDetectFrame"/>'s output) via the zero-copy
    /// <see cref="OrtValue"/> calling convention (never <c>NamedOnnxValue</c>/<c>DenseTensor</c>),
    /// and materializes every output as a managed array + shape pair. Python:
    /// <c>face_detector.run(None, { 'input': detect_vision_frame })</c> (the <c>forward_with_*</c>
    /// functions).
    /// </summary>
    private static IReadOnlyList<(float[] Data, long[] Shape)> RunSession(InferenceSession inferenceSession, float[] inputData, long[] inputShape)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(inputData, inputShape);
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = inferenceSession.Run(runOptions, inputs, inferenceSession.OutputNames);

        var outputs = new List<(float[], long[])>(results.Count);
        foreach (var result in results)
        {
            var shape = result.GetTensorTypeAndShape().Shape;
            var data = result.GetTensorDataAsSpan<float>().ToArray();
            outputs.Add((data, shape));
        }

        return outputs;
    }

    /// <summary>
    /// Reads an ORT output as a flat row-major (rows, cols) buffer, dropping a leading
    /// batch dimension of 1 where present (retinaface/scrfd's outputs have no batch
    /// dimension at all — e.g. shape <c>(12800, 1)</c> — while yunet's do — e.g.
    /// <c>(1, 6400, 1)</c> — both represent the same per-anchor layout).
    /// </summary>
    private static (float[] Data, int Rows, int Cols) GetOutput((float[] Data, long[] Shape) output)
    {
        var shape = output.Shape;

        if (shape.Length == 2)
        {
            return (output.Data, (int)shape[0], (int)shape[1]);
        }

        if (shape.Length == 3 && shape[0] == 1)
        {
            return (output.Data, (int)shape[1], (int)shape[2]);
        }

        throw new InvalidOperationException($"Unexpected detector output shape rank {shape.Length}.");
    }
}
