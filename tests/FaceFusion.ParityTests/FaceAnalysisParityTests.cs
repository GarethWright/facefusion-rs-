using System.Text.Json;
using FaceFusion.Face;
using FaceFusion.Parity;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>FaceFusion.Face.FaceLandmarker</c>/<c>FaceRecognizer</c>/
/// <c>FaceClassifier</c> against the real Python <c>facefusion.face_landmarker</c>/
/// <c>face_recognizer</c>/<c>face_classifier</c>, run against the real
/// <c>arcface_w600k_r50</c>/<c>fairface</c>/<c>2dfan4</c>/<c>peppa_wutz</c>/<c>fan_68_5</c>
/// ONNX models and the largest detected face on the real <c>source.jpg</c> example image.
/// Ground truth was captured with <c>tools/parity/dump_face_analysis.py</c>; see that script's
/// docstring and docs/PARITY_HARNESS.md.
///
/// <para>
/// <b>Two tiers of tests, gated differently.</b> The preprocessing-tensor tests below (channel
/// reversal, normalisation, contrast optimisation, translate+rotate warp geometry) need only
/// the committed <c>.npy</c> fixtures and a source image — no <c>.onnx</c> model — so they run
/// unconditionally once the example media is present. The end-to-end tests that run a real
/// <see cref="InferenceSession"/> additionally need the corresponding model file under
/// <c>.assets/models/</c>, which is <c>.gitignore</c>'d and never present on CI; those are
/// gated with <see cref="ModelFactAttribute"/> and skip with a clear message instead of
/// failing. Splitting the tiers this way means a model-input-tensor mismatch and a genuine ORT
/// divergence are never conflated (see PARITY_HARNESS.md's tolerance guidance): if the
/// preprocessing test passes and the end-to-end test still disagrees beyond ~0, the bug is not
/// in this port's tensor construction.
/// </para>
///
/// <para>
/// <b>Reference face.</b> All fixtures describe the largest (by bounding-box area) of the ten
/// faces YOLO-face detects on <c>source.jpg</c>, so every test in this file exercises the same
/// face end to end.
/// </para>
/// </summary>
public sealed class FaceAnalysisParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "face_analysis");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    // -----------------------------------------------------------------
    // Shared fixture / environment helpers
    // -----------------------------------------------------------------

    internal static string? FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);
        while (directory is not null)
        {
            if (File.Exists(Path.Combine(directory.FullName, "FaceFusion.sln")))
            {
                return directory.FullName;
            }

            directory = directory.Parent;
        }

        return null;
    }

    internal static string? FindModelPath(string modelFileName)
    {
        var repoRoot = FindRepoRoot();
        return repoRoot is null ? null : Path.Combine(repoRoot, ".assets", "models", modelFileName);
    }

    internal static bool ModelAvailable(string modelFileName)
    {
        var modelPath = FindModelPath(modelFileName);
        return modelPath is not null && File.Exists(modelPath) && new FileInfo(modelPath).Length > 0;
    }

    internal static bool SourceImageAvailable =>
        File.Exists(SourceImage) && new FileInfo(SourceImage).Length > 0;

    private static NpyArray LoadNpy(params string[] pathParts) =>
        NpyReader.Load(Path.Combine(new[] { FixturesDirectory }.Concat(pathParts).ToArray()));

    private static double ReadJsonDouble(string relativePath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, relativePath)));
        return document.RootElement.GetDouble();
    }

    private static string ReadJsonString(string relativePath)
    {
        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, relativePath)));
        return document.RootElement.GetString()!;
    }

    /// <summary>
    /// Builds an 8-bit BGR <see cref="Mat"/> (the <c>FaceHelper</c>/<c>Vision</c> VisionFrame
    /// convention) from a dumped <c>uint8 (H, W, 3)</c> fixture. Caller owns the result.
    /// </summary>
    private static Mat MatFromUInt8HwcFixture(NpyArray array)
    {
        Assert.Equal("uint8", array.DType);
        Assert.Equal(3, array.Shape.Count);
        Assert.Equal(3, array.Shape[2]);

        var height = array.Shape[0];
        var width = array.Shape[1];
        var raw = array.RawData;

        var mat = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[height * width];
        for (var i = 0; i < pixels.Length; i++)
        {
            pixels[i] = new Vec3b(raw[i * 3], raw[(i * 3) + 1], raw[(i * 3) + 2]);
        }

        mat.SetArray(pixels);
        return mat;
    }

    private static float[,] LoadLandmark5(NpyArray array)
    {
        Assert.Equal(new[] { 5, 2 }, array.Shape);
        var values = array.AsFloats();
        var result = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            result[i, 0] = values[i * 2];
            result[i, 1] = values[(i * 2) + 1];
        }

        return result;
    }

    private static float[] LoadBoundingBox(NpyArray array)
    {
        Assert.Equal(new[] { 4 }, array.Shape);
        return array.AsFloats();
    }

    // -----------------------------------------------------------------
    // FaceRecognizer.PrepareInput — model-input tensor, no ONNX Runtime required
    // -----------------------------------------------------------------

    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestFaceRecognizerPrepareInputMatchesPython()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("face_recognizer", "crop_vision_frame.npy"));

        var actual = FaceRecognizer.PrepareInput(cropVisionFrame);
        var expected = LoadNpy("face_recognizer", "input.npy").AsDoubles();

        // ONNX Runtime does no arithmetic here — this is pure managed preprocessing
        // reproducing numpy's promote-to-float64-then-narrow behaviour exactly (see
        // FaceRecognizer's class remarks), so PARITY_HARNESS.md's "expect ~0" bar applies.
        var actualDoubles = Array.ConvertAll(actual, value => (double)value);
        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // FaceClassifier.PrepareInput — model-input tensor, no ONNX Runtime required
    // -----------------------------------------------------------------

    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestFaceClassifierPrepareInputMatchesPython()
    {
        using var cropVisionFrame = MatFromUInt8HwcFixture(LoadNpy("face_classifier", "crop_vision_frame.npy"));

        var actual = FaceClassifier.PrepareInput(cropVisionFrame);
        var expected = LoadNpy("face_classifier", "input.npy").AsDoubles();

        var actualDoubles = Array.ConvertAll(actual, value => (double)value);
        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 1e-6, absoluteTolerance: 1e-6);
        Assert.True(result.Passed, result.Describe());
    }

    // -----------------------------------------------------------------
    // FaceLandmarker preprocessing — scale/translation, contrast optimisation, model input
    // -----------------------------------------------------------------

    [Fact]
    public void TestComputeScaleAndTranslationMatchesPython()
    {
        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var expectedScale = ReadJsonDouble(Path.Combine("face_landmarker", "2dfan4_scale.json"));
        var expectedTranslation = LoadNpy("face_landmarker", "2dfan4_translation.npy").AsDoubles();

        var (actualScale, actualTranslation) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, FaceLandmarker.TwoDFan4ModelSize);

        Assert.True(TensorComparison.ElementCloses(actualScale, expectedScale, 1e-9, 1e-9), $"scale: actual={actualScale}, expected={expectedScale}");
        var result = TensorComparison.Compare(actualTranslation, expectedTranslation, relativeTolerance: 1e-9, absoluteTolerance: 1e-9);
        Assert.True(result.Passed, result.Describe());
    }

    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestConditionalOptimizeContrastMatchesPython()
    {
        using var preContrastCrop = MatFromUInt8HwcFixture(LoadNpy("face_landmarker", "pre_contrast_crop.npy"));
        using var actual = FaceLandmarker.ConditionalOptimizeContrast(preContrastCrop);

        var expected = LoadNpy("face_landmarker", "optimized_contrast_crop.npy");
        Assert.Equal(new[] { actual.Rows, actual.Cols, actual.Channels() }, expected.Shape);

        actual.GetArray(out Vec3b[] actualPixels);
        var expectedRaw = expected.RawData;

        // OpenCV arithmetic (CvtColor/CLAHE/Merge) on both sides -> expect exact equality,
        // not just closeness, per PARITY_HARNESS.md.
        var mismatches = 0;
        for (var i = 0; i < actualPixels.Length; i++)
        {
            if (actualPixels[i].Item0 != expectedRaw[i * 3] || actualPixels[i].Item1 != expectedRaw[(i * 3) + 1] || actualPixels[i].Item2 != expectedRaw[(i * 3) + 2])
            {
                mismatches++;
            }
        }

        Assert.True(mismatches == 0, $"{mismatches} of {actualPixels.Length} pixels differ after conditional_optimize_contrast");
    }

    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestTwoDFan4PreprocessingPipelineMatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];

        var modelSize = FaceLandmarker.TwoDFan4ModelSize;
        var (scale, translation) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, modelSize);

        var (rotationMatrix, rotationSize) = FaceHelper.CreateRotationMatrixAndSize(faceAngle, modelSize);
        using var rotationMatrixDisposable = rotationMatrix;

        var (translatedCrop, affineMatrix) = FaceHelper.WarpFaceByTranslation(visionFrame!, translation, scale, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var translatedCropDisposable = translatedCrop;

        using var rotatedCrop = new Mat();
        Cv2.WarpAffine(translatedCrop, rotatedCrop, rotationMatrix, rotationSize);

        using var contrastCrop = FaceLandmarker.ConditionalOptimizeContrast(rotatedCrop);
        var inputTensor = FaceLandmarker.PrepareLandmarkerInput(contrastCrop);

        var expected = LoadNpy("face_landmarker", "model_input.npy").AsDoubles();
        var actualDoubles = Array.ConvertAll(inputTensor, value => (double)value);

        // NOT an allclose comparison — see TestWarpFaceByTranslationIsCloseToPythonWarpAffine's
        // remarks for why. This tensor stacks two cv2.warpAffine calls (translate, then the
        // angle=0 "rotation" — still a real resample, not a no-op) on top of that per-pixel
        // divergence, then always round-trips through 8-bit Lab (conditional_optimize_
        // contrast), whose quantisation can amplify a +-2/255 input difference at some pixels.
        // Measured PSNR here is ~56 dB; 50 dB leaves comfortable headroom while still catching
        // a real preprocessing regression (which would be off by much more than a couple of
        // 8-bit levels).
        var psnr = ImageMetrics.Psnr(actualDoubles, expected, maxValue: 1.0);
        Assert.True(psnr > 50.0, $"model input tensor PSNR {psnr:F2} dB fell at/below the 50 dB threshold.");
    }

    /// <summary>
    /// Isolates the root cause of <see cref="TestTwoDFan4PreprocessingPipelineMatchesPython"/>'s
    /// non-zero divergence to a single cause: <c>Cv2.WarpAffine</c> (called from
    /// <c>FaceHelper.WarpFaceByTranslation</c>, out of this module's scope to change) disagrees
    /// with Python's <c>cv2.warpAffine</c> by up to 2 of 255 levels on ~9% of pixels for this
    /// crop, given the *exact same* affine matrix (verified: the scale/translation/matrix
    /// values themselves match Python to 15+ significant digits — see
    /// <see cref="TestComputeScaleAndTranslationMatchesPython"/> — so this is not a geometry
    /// bug in this port's matrix construction). This matches the class of divergence
    /// <c>VisionParityTests.ReadStaticVideoFrameIsCloseToFfmpegDecode</c> already documents for
    /// a different pair of decoders: two independent native builds of the same bilinear
    /// resampling algorithm, agreeing almost everywhere but not bit-for-bit. Measured PSNR here
    /// is ~62 dB.
    /// </summary>
    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestWarpFaceByTranslationIsCloseToPythonWarpAffine()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];
        var modelSize = FaceLandmarker.TwoDFan4ModelSize;
        var (scale, translation) = FaceLandmarker.ComputeScaleAndTranslation(boundingBox, modelSize);

        var (rotationMatrix, rotationSize) = FaceHelper.CreateRotationMatrixAndSize(faceAngle, modelSize);
        using var rotationMatrixDisposable = rotationMatrix;

        var (translatedCrop, affineMatrix) = FaceHelper.WarpFaceByTranslation(visionFrame!, translation, scale, modelSize);
        using var affineMatrixDisposable = affineMatrix;
        using var translatedCropDisposable = translatedCrop;

        using var rotatedCrop = new Mat();
        Cv2.WarpAffine(translatedCrop, rotatedCrop, rotationMatrix, rotationSize);

        rotatedCrop.GetArray(out Vec3b[] actualPixels);
        var actual = new double[actualPixels.Length * 3];
        for (var i = 0; i < actualPixels.Length; i++)
        {
            actual[i * 3] = actualPixels[i].Item0;
            actual[(i * 3) + 1] = actualPixels[i].Item1;
            actual[(i * 3) + 2] = actualPixels[i].Item2;
        }

        var expected = LoadNpy("face_landmarker", "pre_contrast_crop.npy").AsDoubles();
        var psnr = ImageMetrics.Psnr(actual, expected, maxValue: 255.0);
        Assert.True(psnr > 55.0, $"warp PSNR {psnr:F2} dB fell at/below the 55 dB threshold.");
    }

    /// <summary>
    /// Same isolation as <see cref="TestWarpFaceByTranslationIsCloseToPythonWarpAffine"/>, for
    /// <c>FaceHelper.WarpFaceByFaceLandmark5</c> (the warp <c>FaceRecognizer</c>/
    /// <c>FaceClassifier</c> use). Measured PSNR here is ~62 dB.
    /// </summary>
    [SkippableFact(nameof(SourceImageAvailable))]
    public void TestWarpFaceByFaceLandmark5IsCloseToPythonWarpAffine()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var faceLandmark5 = LoadLandmark5(LoadNpy("reference", "face_landmark_5.npy"));
        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(visionFrame!, faceLandmark5, FaceRecognizer.ModelTemplate, FaceRecognizer.ModelSize);
        using var cropDisposable = cropVisionFrame;
        using var matrixDisposable = affineMatrix;

        cropVisionFrame.GetArray(out Vec3b[] actualPixels);
        var actual = new double[actualPixels.Length * 3];
        for (var i = 0; i < actualPixels.Length; i++)
        {
            actual[i * 3] = actualPixels[i].Item0;
            actual[(i * 3) + 1] = actualPixels[i].Item1;
            actual[(i * 3) + 2] = actualPixels[i].Item2;
        }

        var expected = LoadNpy("face_recognizer", "crop_vision_frame.npy").AsDoubles();
        var psnr = ImageMetrics.Psnr(actual, expected, maxValue: 255.0);
        Assert.True(psnr > 55.0, $"warp PSNR {psnr:F2} dB fell at/below the 55 dB threshold.");
    }

    // -----------------------------------------------------------------
    // End-to-end (real ONNX Runtime inference) — gated on the model file being present
    // -----------------------------------------------------------------

    /// <summary>
    /// <b>Not a raw allclose comparison, deliberately.</b> The crop this feeds into
    /// <c>arcface_w600k_r50</c> carries the same <c>Cv2.WarpAffine</c>-vs-<c>cv2.warpAffine</c>
    /// divergence documented by <see cref="TestWarpFaceByFaceLandmark5IsCloseToPythonWarpAffine"/>
    /// (a handful of pixels off by up to 2 of 255 levels); a deep CNN is not linear, so that
    /// tiny input perturbation does not stay tiny through 512 dimensions of embedding (measured
    /// max per-element difference ~0.02, on values with magnitude up to ~3). What actually
    /// matters for this embedding — per the assignment brief, "embeddings are compared by
    /// distance downstream" — is direction, not the raw components: measured cosine similarity
    /// between the two embeddings is 0.99996. This test asserts that (with real headroom below
    /// the measured value), which is the property face-matching downstream logic actually
    /// relies on, rather than an elementwise tolerance that would need loosening to the point
    /// of being meaningless just to absorb an upstream warp-interpolation rounding difference
    /// this module does not own (<c>FaceHelper.WarpFaceByFaceLandmark5</c> is out of scope —
    /// see the assignment's file list).
    /// </summary>
    [ModelFact("arcface_w600k_r50.onnx")]
    public void TestCalculateFaceEmbeddingMatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var faceLandmark5 = LoadLandmark5(LoadNpy("reference", "face_landmark_5.npy"));

        using var session = new InferenceSession(FindModelPath("arcface_w600k_r50.onnx"));
        var (embedding, embeddingNorm) = FaceRecognizer.CalculateFaceEmbedding(session, visionFrame!, faceLandmark5);

        var expectedEmbedding = LoadNpy("face_recognizer", "embedding.npy").AsDoubles();
        var expectedEmbeddingNorm = LoadNpy("face_recognizer", "embedding_norm.npy").AsDoubles();

        var cosineSimilarity = CosineSimilarity(Array.ConvertAll(embedding, value => (double)value), expectedEmbedding);
        Assert.True(cosineSimilarity > 0.999, $"embedding cosine similarity {cosineSimilarity:F6} fell at/below 0.999.");

        var cosineSimilarityNorm = CosineSimilarity(Array.ConvertAll(embeddingNorm, value => (double)value), expectedEmbeddingNorm);
        Assert.True(cosineSimilarityNorm > 0.999, $"embedding_norm cosine similarity {cosineSimilarityNorm:F6} fell at/below 0.999.");

        // embedding_norm must itself be unit-length regardless of the above -- this is exact
        // managed float math (FaceFusion.Tensors.NumPy.LinalgNorm + a division loop) with no
        // upstream warp dependency, so it gets the tight PARITY_HARNESS.md tolerance.
        var norm = Math.Sqrt(embeddingNorm.Sum(value => (double)value * value));
        Assert.True(Math.Abs(norm - 1.0) < 1e-4, $"embedding_norm L2 norm {norm} is not ~1.0.");
    }

    private static double CosineSimilarity(double[] actual, double[] expected)
    {
        double dot = 0, actualNormSquared = 0, expectedNormSquared = 0;
        for (var i = 0; i < actual.Length; i++)
        {
            dot += actual[i] * expected[i];
            actualNormSquared += actual[i] * actual[i];
            expectedNormSquared += expected[i] * expected[i];
        }

        return dot / (Math.Sqrt(actualNormSquared) * Math.Sqrt(expectedNormSquared));
    }

    [ModelFact("fairface.onnx")]
    public void TestClassifyFaceMatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var faceLandmark5 = LoadLandmark5(LoadNpy("reference", "face_landmark_5.npy"));

        using var session = new InferenceSession(FindModelPath("fairface.onnx"));
        var (gender, age, race) = FaceClassifier.ClassifyFace(session, visionFrame!, faceLandmark5);

        var expectedGender = ReadJsonString(Path.Combine("face_classifier", "gender.json"));
        var expectedAge = JsonDocument.Parse(File.ReadAllText(Path.Combine(FixturesDirectory, "face_classifier", "age.json"))).RootElement;
        var expectedRace = ReadJsonString(Path.Combine("face_classifier", "race.json"));

        Assert.Equal(expectedGender, gender == Gender.Female ? "female" : "male");
        Assert.Equal(expectedAge[0].GetInt32(), age.Start.Value);
        Assert.Equal(expectedAge[1].GetInt32(), age.End.Value);
        Assert.Equal(expectedRace, race.ToString().ToLowerInvariant());
    }

    [ModelFact("2dfan4.onnx")]
    public void TestDetectWith2dFan4MatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];

        using var session = new InferenceSession(FindModelPath("2dfan4.onnx"));
        var (faceLandmark68, score) = FaceLandmarker.DetectWith2dFan4(session, visionFrame!, boundingBox, faceAngle);

        var expectedLandmark68 = LoadNpy("face_landmarker", "2dfan4_landmark_68.npy").AsDoubles();
        var expectedScore = ReadJsonDouble(Path.Combine("face_landmarker", "2dfan4_score.json"));

        var actualLandmark68 = FlattenLandmark68(faceLandmark68);

        // 2dfan4 landmarks are pixel coordinates in a 256x256-scale crop then transformed back
        // through two affine inversions -- ONNX Runtime + OpenCV arithmetic throughout, so a
        // sub-thousandth-pixel tolerance is appropriate (not "loosened to pass").
        var result = TensorComparison.Compare(actualLandmark68, expectedLandmark68, relativeTolerance: 1e-3, absoluteTolerance: 1e-3);
        Assert.True(result.Passed, result.Describe());
        Assert.True(TensorComparison.ElementCloses(score, expectedScore, 1e-4, 1e-4), $"score: actual={score}, expected={expectedScore}");
    }

    [ModelFact("peppa_wutz.onnx")]
    public void TestDetectWithPeppaWutzMatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];

        using var session = new InferenceSession(FindModelPath("peppa_wutz.onnx"));
        var (faceLandmark68, score) = FaceLandmarker.DetectWithPeppaWutz(session, visionFrame!, boundingBox, faceAngle);

        var expectedLandmark68 = LoadNpy("face_landmarker", "peppa_wutz_landmark_68.npy").AsDoubles();
        var expectedScore = ReadJsonDouble(Path.Combine("face_landmarker", "peppa_wutz_score.json"));

        var actualLandmark68 = FlattenLandmark68(faceLandmark68);

        var result = TensorComparison.Compare(actualLandmark68, expectedLandmark68, relativeTolerance: 1e-3, absoluteTolerance: 1e-3);
        Assert.True(result.Passed, result.Describe());
        Assert.True(TensorComparison.ElementCloses(score, expectedScore, 1e-4, 1e-4), $"score: actual={score}, expected={expectedScore}");
    }

    [ModelFact("2dfan4.onnx", "peppa_wutz.onnx")]
    public void TestDetectFaceLandmarkPicksTheHigherScoringModel()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];

        using var twoDFan4Session = new InferenceSession(FindModelPath("2dfan4.onnx"));
        using var peppaWutzSession = new InferenceSession(FindModelPath("peppa_wutz.onnx"));

        var (faceLandmark68, score) = FaceLandmarker.DetectFaceLandmark(
            FaceLandmarkerModel.Many, twoDFan4Session, peppaWutzSession, visionFrame!, boundingBox, faceAngle);

        var expectedLandmark68 = LoadNpy("face_landmarker", "detect_face_landmark_68.npy").AsDoubles();
        var expectedScore = ReadJsonDouble(Path.Combine("face_landmarker", "detect_face_landmark_score.json"));

        Assert.NotNull(faceLandmark68);
        var actualLandmark68 = FlattenLandmark68(faceLandmark68!);

        var result = TensorComparison.Compare(actualLandmark68, expectedLandmark68, relativeTolerance: 1e-3, absoluteTolerance: 1e-3);
        Assert.True(result.Passed, result.Describe());
        Assert.True(TensorComparison.ElementCloses(score, expectedScore, 1e-4, 1e-4), $"score: actual={score}, expected={expectedScore}");
    }

    [ModelFact("2dfan4.onnx", "fan_68_5.onnx")]
    public void TestEstimateFaceLandmark685MatchesPython()
    {
        using var visionFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(visionFrame);

        var boundingBox = LoadBoundingBox(LoadNpy("face_landmarker", "bounding_box.npy"));
        var faceAngle = (int)LoadNpy("face_landmarker", "face_angle.npy").AsDoubles()[0];

        using var twoDFan4Session = new InferenceSession(FindModelPath("2dfan4.onnx"));
        var (faceLandmark68, _) = FaceLandmarker.DetectWith2dFan4(twoDFan4Session, visionFrame!, boundingBox, faceAngle);
        var faceLandmark5Of68 = FaceHelper.ConvertToFaceLandmark5(faceLandmark68);

        using var fan685Session = new InferenceSession(FindModelPath("fan_68_5.onnx"));
        var actual68 = FaceLandmarker.EstimateFaceLandmark685(fan685Session, faceLandmark5Of68);

        var expected = LoadNpy("face_landmarker", "fan_68_5_output.npy").AsDoubles();
        var actualDoubles = FlattenLandmark68(actual68);

        var result = TensorComparison.Compare(actualDoubles, expected, relativeTolerance: 1e-3, absoluteTolerance: 1e-3);
        Assert.True(result.Passed, result.Describe());
    }

    private static double[] FlattenLandmark68(float[,] landmark68)
    {
        var flat = new double[68 * 2];
        for (var i = 0; i < 68; i++)
        {
            flat[i * 2] = landmark68[i, 0];
            flat[(i * 2) + 1] = landmark68[i, 1];
        }

        return flat;
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c>
/// file(s) are not present — same reasoning as
/// <c>FaceFusion.UnitTests.MediaFactAttribute</c>, but for models rather than example media.
/// xunit 2.4.2 evaluates the constructor once per test method at discovery time, so this can
/// only depend on static, already-known-at-discovery state (the file system), matching how
/// <c>MediaFactAttribute</c> depends on <c>TestHelper.ExamplesAvailable</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class ModelFactAttribute : FactAttribute
{
    public ModelFactAttribute(params string[] modelFileNames)
    {
        foreach (var modelFileName in modelFileNames)
        {
            if (!FaceAnalysisParityTests.ModelAvailable(modelFileName))
            {
                Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                       "run `FACEFUSION_PARITY_DIR=... python3 tools/parity/dump_face_analysis.py` once with " +
                       "network access to populate .assets/models via pre_check(), then retry";
                return;
            }
        }
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named static boolean property (found by
/// reflection on <see cref="FaceAnalysisParityTests"/>) is false — used here for
/// <see cref="FaceAnalysisParityTests.SourceImageAvailable"/> so the preprocessing-tensor
/// tests skip cleanly rather than throwing a confusing null-reference deep inside
/// <c>Vision.ReadStaticImage</c> when <c>source.jpg</c> has not been fetched.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class SkippableFactAttribute : FactAttribute
{
    public SkippableFactAttribute(string propertyName)
    {
        var property = typeof(FaceAnalysisParityTests).GetProperty(propertyName, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static);
        var available = property is not null && (bool)property.GetValue(null)!;

        if (!available)
        {
            Skip = "requires the example media in /tmp/facefusion-test-examples (source.jpg) — run tools/parity/fetch_examples.sh, then retry";
        }
    }
}
