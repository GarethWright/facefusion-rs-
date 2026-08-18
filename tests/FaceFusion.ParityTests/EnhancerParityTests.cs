using System.Text.Json;
using FaceFusion.Face;
using FaceFusion.Inference;
using FaceFusion.Parity;
using FaceFusion.Processors;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for <c>facefusion.processors.modules.face_enhancer</c> and
/// <c>facefusion.processors.modules.frame_enhancer</c> (docs/PARITY_HARNESS.md). Ground truth
/// was dumped from the real Python modules (<c>tools/parity/dump_face_enhancer.py</c> /
/// <c>tools/parity/dump_frame_enhancer.py</c>), running real ONNX inference against the real
/// example media — not synthetic data.
///
/// <para>
/// <b>Fixture layout</b> — <c>fixtures/enhancers/</c>:
/// <list type="bullet">
/// <item><description><c>face_enhancer/reference/{bounding_box,face_landmark_5,model_size}</c> —
/// the largest detected face on source.jpg (via yolo_face), dumped once so this test does not
/// need its own detector session.</description></item>
/// <item><description><c>face_enhancer/gpen_bfr_256/{crop_vision_frame,input,forward_output,
/// normalized_crop_vision_frame,enhance_face_output}</c> — every stage of <c>enhance_face()</c>
/// for the smallest face_enhancer model (no <c>'weight'</c> input).</description></item>
/// <item><description><c>frame_enhancer/reference/{crop_frame,crop_shape}</c> — a small 96x80
/// crop of source.jpg (kept small deliberately — see the dumper's docstring for why the full
/// 1024x1024 frame is not tiled here).</description></item>
/// <item><description><c>frame_enhancer/real_web_photo_x4/{model_size,model_scale,tile_count,
/// pad_width,pad_height,tile_0,tile_1,enhance_frame_output}</c> — every stage of
/// <c>enhance_frame()</c> for the smallest-tile frame_enhancer model, first two tiles only.
/// </description></item>
/// </list>
/// </para>
///
/// <para>
/// <b>Tolerances</b> (per PARITY_HARNESS.md's "choosing a tolerance"):
/// <list type="bullet">
/// <item><description><b>Model-input tensors</b> (<c>PrepareCropFrame</c>/<c>PrepareTileFrame</c>,
/// fed the Python-dumped crop/tile directly rather than a locally-rewarped one — isolating
/// preprocessing arithmetic from the separate warp-affine/tiling divergence below) use
/// <c>rtol = atol = 0</c>. This matches every other ported stage's model-input check. For
/// <c>face_enhancer</c> this required a real fix during porting: Python's
/// <c>prepare_crop_frame</c> divides by the Python float <c>255.0</c>, which numpy promotes
/// to <c>float64</c> and keeps at <c>float64</c> through the following <c>- 0.5</c> /
/// <c>/ 0.5</c>, narrowing to <c>float32</c> only at the very end — an initial float32-throughout
/// implementation (naively mirroring <c>frame_enhancer.prepare_tile_frame</c>, which genuinely
/// *is* float32-throughout because its own <c>.astype(float32)</c> happens *before* the
/// divide) measurably diverged from the fixture; see
/// <see cref="FaceEnhancer.PrepareCropFrame"/>'s remarks for the fix and
/// <see cref="FaceFusion.Face.FaceRecognizer.PrepareInput"/> for the established precedent of
/// the same double-vs-float32 distinction.</description></item>
/// <item><description><b>ONNX model outputs</b> (<c>Forward</c>, fed the Python-dumped input
/// tensor directly) use a tight but non-zero tolerance — both sides call the same ORT kernels
/// on the same input, so PARITY_HARNESS.md's "expect ~0 divergence" applies; the measured
/// worst case (see the port report) is far below the threshold used here.</description></item>
/// <item><description><b>Post-processing math</b> (<c>NormalizeCropFrame</c>/<c>NormalizeTileFrame</c>,
/// fed the Python-dumped model output directly) is deterministic per-element arithmetic
/// (clip/scale/round/cast) and is asserted with a 1-count tolerance to absorb the documented
/// double-vs-float32 rounding-boundary cases (see <c>NormalizeCropFrameClipsOutOfRangeValues</c>
/// in FaceEnhancerTests for why an exact half-integer can round either way at a float32
/// boundary), not because the arithmetic itself is approximate.</description></item>
/// <item><description><b>Full-pipeline image outputs</b> (<c>EnhanceFace</c>/<c>EnhanceFrame</c>,
/// starting from the real source frame and running this port's own warp/tile/paste) use
/// PSNR, per PARITY_HARNESS.md's "final frames compare with PSNR/SSIM ... assert a threshold,
/// not equality" — <c>cv2.warpAffine</c>/<c>cv2.resize</c> cascade the documented
/// OpenCvSharp-vs-opencv-python bilinear-interpolation divergence (~2/255 on ~9% of pixels,
/// ~62 dB PSNR for a single warp) through a warp, an ONNX model, a paste-back warp and a
/// blend, so a much lower threshold than 62 dB is used here and is not a blanket excuse — see
/// the measured values in the port report.</description></item>
/// </list>
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class EnhancerParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "enhancers");

    private const string SourceImage = "/tmp/facefusion-test-examples/source.jpg";

    private static bool SourceImageAvailable => File.Exists(SourceImage);

    private const string MissingSourceImageMessage =
        "requires /tmp/facefusion-test-examples/source.jpg — run tools/parity/fetch_examples.sh, then retry";

    private static bool FaceEnhancerModelAvailable => FaceEnhancer.PreCheck(FaceEnhancerModel.GpenBfr256);

    private const string MissingFaceEnhancerModelMessage =
        "requires .assets/models/gpen_bfr_256.onnx — run `FACEFUSION_PARITY_DIR=/tmp/x " +
        "python3 tools/parity/dump_face_enhancer.py` with network access (or any other way " +
        "of running facefusion.processors.modules.face_enhancer.core.pre_check()) to fetch it, then retry";

    private static bool FrameEnhancerModelAvailable => FrameEnhancer.PreCheck(FrameEnhancerModel.RealWebPhotoX4);

    private const string MissingFrameEnhancerModelMessage =
        "requires .assets/models/real_web_photo_x4.onnx — run `FACEFUSION_PARITY_DIR=/tmp/x " +
        "python3 tools/parity/dump_frame_enhancer.py` with network access (or any other way " +
        "of running facefusion.processors.modules.frame_enhancer.core.pre_check()) to fetch it, then retry";

    /// <summary><c>[Fact]</c> that skips at discovery time when the real gpen_bfr_256 model
    /// (and source.jpg) are not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class FaceEnhancerModelFactAttribute : FactAttribute
    {
        public FaceEnhancerModelFactAttribute()
        {
            if (!FaceEnhancerModelAvailable)
            {
                Skip = MissingFaceEnhancerModelMessage;
            }
            else if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    /// <summary><c>[Fact]</c> that skips at discovery time when the real real_web_photo_x4
    /// model (and source.jpg) are not present.</summary>
    [AttributeUsage(AttributeTargets.Method)]
    private sealed class FrameEnhancerModelFactAttribute : FactAttribute
    {
        public FrameEnhancerModelFactAttribute()
        {
            if (!FrameEnhancerModelAvailable)
            {
                Skip = MissingFrameEnhancerModelMessage;
            }
            else if (!SourceImageAvailable)
            {
                Skip = MissingSourceImageMessage;
            }
        }
    }

    // -----------------------------------------------------------------
    // Shared inference sessions (one per model, loaded once for the whole test run).
    // -----------------------------------------------------------------

    private static readonly object SessionLock = new();
    private static InferenceSession? faceEnhancerSession;
    private static InferenceSession? frameEnhancerSession;

    private static InferenceSession GetFaceEnhancerSession()
    {
        lock (SessionLock)
        {
            if (faceEnhancerSession is not null)
            {
                return faceEnhancerSession;
            }

            var modelPath = FaceEnhancer.GetModelOptions(FaceEnhancerModel.GpenBfr256).Source.Path;
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });
            var inferenceManager = new InferenceManager();
            faceEnhancerSession = inferenceManager.CreateInferenceSession(modelPath, inferenceProviders);
            return faceEnhancerSession;
        }
    }

    private static InferenceSession GetFrameEnhancerSession()
    {
        lock (SessionLock)
        {
            if (frameEnhancerSession is not null)
            {
                return frameEnhancerSession;
            }

            var modelPath = FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4).Source.Path;
            var inferenceProviders = Execution.CreateInferenceProviders(0, new[] { ExecutionProvider.Cpu });
            var inferenceManager = new InferenceManager();
            frameEnhancerSession = inferenceManager.CreateInferenceSession(modelPath, inferenceProviders);
            return frameEnhancerSession;
        }
    }

    // ===================================================================
    // face_enhancer
    // ===================================================================

    private static string FaceEnhancerFixturesDirectory => Path.Combine(FixturesDirectory, "face_enhancer");

    /// <summary>Python: `face_landmark_5` — shape (5, 2), float32/float64 depending on the
    /// detector family (yolo_face has no anchor step, so it is float32; see
    /// FaceDetector's remarks). Read generically via AsDoubles then narrowed.</summary>
    private static float[,] LoadFaceLandmark5()
    {
        var array = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "reference", "face_landmark_5.npy"));
        var flat = array.AsDoubles();
        var result = new float[5, 2];
        for (var i = 0; i < 5; i++)
        {
            result[i, 0] = (float)flat[i * 2];
            result[i, 1] = (float)flat[(i * 2) + 1];
        }

        return result;
    }

    /// <summary>Loads a (H, W, 3) uint8 .npy fixture into a caller-owned CV_8UC3 <see cref="Mat"/>.</summary>
    private static Mat LoadUint8Mat(string relativeFixturePath)
    {
        var array = NpyReader.Load(Path.Combine(FixturesDirectory, relativeFixturePath));
        var shape = array.Shape;
        Assert.Equal(3, shape.Count);
        Assert.Equal(3, shape[2]);
        Assert.Equal("uint8", array.DType);

        var mat = new Mat(shape[0], shape[1], MatType.CV_8UC3);
        var bytes = array.RawData.ToArray();
        System.Runtime.InteropServices.Marshal.Copy(bytes, 0, mat.Data, bytes.Length);
        return mat;
    }

    private static double[] MatToBgrDoubles(Mat mat)
    {
        mat.GetArray(out Vec3b[] pixels);
        var result = new double[pixels.Length * 3];
        for (var i = 0; i < pixels.Length; i++)
        {
            result[(i * 3) + 0] = pixels[i].Item0;
            result[(i * 3) + 1] = pixels[i].Item1;
            result[(i * 3) + 2] = pixels[i].Item2;
        }

        return result;
    }

    [FaceEnhancerModelFact]
    public void ModelSizeMatchesPythonModelOptions()
    {
        var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(FaceEnhancerFixturesDirectory, "reference", "model_size.json")));
        var expectedWidth = document.RootElement[0].GetInt32();
        var expectedHeight = document.RootElement[1].GetInt32();

        var options = FaceEnhancer.GetModelOptions(FaceEnhancerModel.GpenBfr256);
        Assert.Equal(expectedWidth, options.Size.Width);
        Assert.Equal(expectedHeight, options.Size.Height);
        Assert.Equal(WarpTemplate.Arcface128, options.Template);
    }

    /// <summary>
    /// The highest-value dump: feeds Python's own dumped crop directly into
    /// <see cref="FaceEnhancer.PrepareCropFrame"/>, isolating the preprocessing arithmetic
    /// from the separate warp-affine divergence (see class remarks). rtol = atol = 0.
    /// </summary>
    [FaceEnhancerModelFact]
    public void PrepareCropFrameMatchesPythonExactly()
    {
        using var pythonCrop = LoadUint8Mat(Path.Combine("face_enhancer", "gpen_bfr_256", "crop_vision_frame.npy"));

        var (actual, height, width) = FaceEnhancer.PrepareCropFrame(pythonCrop);
        Assert.Equal(256, height);
        Assert.Equal(256, width);

        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);
        var expected = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "input.npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    /// <summary>ORT does the arithmetic on both sides — expect ~0 divergence.</summary>
    [FaceEnhancerModelFact]
    public void FaceEnhancerForwardMatchesPythonOrtOutput()
    {
        var expectedInput = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "input.npy")).AsFloats();
        var session = GetFaceEnhancerSession();

        var actual = FaceEnhancer.Forward(expectedInput, 256, 256, 0.5, session);
        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);

        var expected = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "forward_output.npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, result.Describe());
    }

    /// <summary>Deterministic post-processing math (clip/scale/round/cast), fed Python's own
    /// dumped model output directly.</summary>
    [FaceEnhancerModelFact]
    public void NormalizeCropFrameMatchesPython()
    {
        var forwardOutput = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "forward_output.npy")).AsFloats();

        using var actual = FaceEnhancer.NormalizeCropFrame(forwardOutput, 256, 256);
        var actualAsDouble = MatToBgrDoubles(actual);

        var expected = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "normalized_crop_vision_frame.npy")).AsDoubles();

        // A 1-count absolute tolerance absorbs float32-vs-Python-float64 rounding-boundary
        // disagreement at exact .5 values (round-half-to-even on two different binary
        // representations of the same decimal can round to adjacent integers) — see class
        // remarks. The arithmetic itself is not approximate.
        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 1.0);
        Assert.True(result.Passed, result.Describe());
    }

    /// <summary>
    /// End-to-end <c>enhance_face()</c>, starting from the real source frame and running this
    /// port's own warp/mask/paste/blend — PSNR, not exact equality (see class remarks).
    /// </summary>
    [FaceEnhancerModelFact]
    public void EnhanceFaceEndToEndIsCloseToPython()
    {
        using var sourceFrame = FaceFusion.Vision.Vision.ReadStaticImage(SourceImage);
        Assert.NotNull(sourceFrame);

        var faceLandmark5 = LoadFaceLandmark5();
        var targetFace = new FaceFusion.Types.Face(
            Origin: "detect",
            BoundingBox: Array.Empty<float>(),
            ScoreSet: new FaceScoreSet(1.0, 0.0),
            LandmarkSet: new FaceLandmarkSet(faceLandmark5, faceLandmark5, faceLandmark5, faceLandmark5),
            Angle: 0,
            Embedding: Array.Empty<float>(),
            EmbeddingNorm: Array.Empty<float>(),
            Age: 0..100,
            Gender: Gender.Male,
            Race: Race.White);

        var modelOptions = FaceEnhancer.GetModelOptions(FaceEnhancerModel.GpenBfr256);
        var occluderPool = new Dictionary<string, InferenceSession>();
        var session = GetFaceEnhancerSession();

        using var actual = FaceEnhancer.EnhanceFace(
            targetFace, sourceFrame!, modelOptions,
            faceMaskBlur: 0.3, faceMaskTypes: new[] { FaceMaskType.Box },
            faceOccluderModel: FaceOccluderModel.Xseg1, occluderInferencePool: occluderPool,
            faceEnhancerSession: session, faceEnhancerWeight: 0.5, faceEnhancerBlend: 80);

        var actualAsDouble = MatToBgrDoubles(actual);
        var expected = NpyReader.Load(Path.Combine(FaceEnhancerFixturesDirectory, "gpen_bfr_256", "enhance_face_output.npy")).AsDoubles();

        var psnr = ImageMetrics.Psnr(actualAsDouble, expected);
        Assert.True(psnr > 30.0, $"EnhanceFace PSNR {psnr:F2} dB fell at/below the 30 dB threshold — see class remarks for the cascaded-interpolation baseline this compares against.");
    }

    // ===================================================================
    // frame_enhancer
    // ===================================================================

    private static string FrameEnhancerFixturesDirectory => Path.Combine(FixturesDirectory, "frame_enhancer");

    private static int ReadJsonInt(string path) => JsonDocument.Parse(File.ReadAllText(path)).RootElement.GetInt32();

    [FrameEnhancerModelFact]
    public void ModelOptionsMatchPythonModelSet()
    {
        var expectedSize = JsonDocument.Parse(File.ReadAllText(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "model_size.json"))).RootElement;
        var expectedScale = ReadJsonInt(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "model_scale.json"));

        var options = FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4);
        Assert.Equal(expectedSize[0].GetInt32(), options.Size.TileSize);
        Assert.Equal(expectedSize[1].GetInt32(), options.Size.PadSize);
        Assert.Equal(expectedSize[2].GetInt32(), options.Size.OverlapSize);
        Assert.Equal(expectedScale, options.Scale);
    }

    [FrameEnhancerModelFact]
    public void CreateTileFramesMatchesPythonTileCountAndPadding()
    {
        using var cropFrame = LoadUint8Mat(Path.Combine("frame_enhancer", "reference", "crop_frame.npy"));
        var options = FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4);

        var (tiles, padWidth, padHeight) = FaceFusion.Vision.Vision.CreateTileFrames(cropFrame, options.Size);
        try
        {
            var expectedTileCount = ReadJsonInt(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "tile_count.json"));
            var expectedPadWidth = ReadJsonInt(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "pad_width.json"));
            var expectedPadHeight = ReadJsonInt(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "pad_height.json"));

            Assert.Equal(expectedTileCount, tiles.Count);
            Assert.Equal(expectedPadWidth, padWidth);
            Assert.Equal(expectedPadHeight, padHeight);

            for (var index = 0; index < 2 && index < tiles.Count; index++)
            {
                var actualAsDouble = MatToBgrDoubles(tiles[index]);
                var expected = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{index}", "raw.npy")).AsDoubles();
                var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 0);
                Assert.True(result.Passed, $"tile {index}: {result.Describe()}");
            }
        }
        finally
        {
            foreach (var tile in tiles)
            {
                tile.Dispose();
            }
        }
    }

    /// <summary>The highest-value dump, per tile: feeds Python's own dumped raw tile directly
    /// into <see cref="FrameEnhancer.PrepareTileFrame"/>. rtol = atol = 0.</summary>
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void PrepareTileFrameMatchesPythonExactly(int tileIndex)
    {
        if (!FrameEnhancerModelAvailable || !SourceImageAvailable)
        {
            return; // see FrameEnhancerModelFactAttribute — Theory has no direct equivalent.
        }

        using var pythonTile = LoadUint8Mat(Path.Combine("frame_enhancer", "real_web_photo_x4", $"tile_{tileIndex}", "raw.npy"));

        var (actual, height, width) = FrameEnhancer.PrepareTileFrame(pythonTile);
        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);

        var expected = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{tileIndex}", "input.npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 0);
        Assert.True(result.Passed, result.Describe());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void FrameEnhancerForwardMatchesPythonOrtOutput(int tileIndex)
    {
        if (!FrameEnhancerModelAvailable || !SourceImageAvailable)
        {
            return;
        }

        var expectedInput = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{tileIndex}", "input.npy")).AsFloats();
        var session = GetFrameEnhancerSession();

        var actual = FrameEnhancer.Forward(expectedInput, 64, 64, session, out var outputHeight, out var outputWidth);
        var actualAsDouble = Array.ConvertAll(actual, x => (double)x);

        var expected = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{tileIndex}", "forward_output.npy")).AsDoubles();

        Assert.Equal(256, outputHeight);
        Assert.Equal(256, outputWidth);

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, result.Describe());
    }

    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    public void NormalizeTileFrameMatchesPython(int tileIndex)
    {
        if (!FrameEnhancerModelAvailable || !SourceImageAvailable)
        {
            return;
        }

        var forwardOutput = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{tileIndex}", "forward_output.npy")).AsFloats();

        using var actual = FrameEnhancer.NormalizeTileFrame(forwardOutput, 256, 256);
        var actualAsDouble = MatToBgrDoubles(actual);

        var expected = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", $"tile_{tileIndex}", "normalized.npy")).AsDoubles();

        var result = TensorComparison.Compare(actualAsDouble, expected, relativeTolerance: 0, absoluteTolerance: 1.0);
        Assert.True(result.Passed, result.Describe());
    }

    /// <summary>
    /// End-to-end <c>enhance_frame()</c> — PSNR, not exact equality (see class remarks).
    /// </summary>
    [FrameEnhancerModelFact]
    public void EnhanceFrameEndToEndIsCloseToPython()
    {
        using var cropFrame = LoadUint8Mat(Path.Combine("frame_enhancer", "reference", "crop_frame.npy"));
        var modelOptions = FrameEnhancer.GetModelOptions(FrameEnhancerModel.RealWebPhotoX4);
        var session = GetFrameEnhancerSession();

        using var actual = FrameEnhancer.EnhanceFrame(cropFrame, modelOptions, session, frameEnhancerBlend: 80);
        var actualAsDouble = MatToBgrDoubles(actual);

        var expected = NpyReader.Load(Path.Combine(FrameEnhancerFixturesDirectory, "real_web_photo_x4", "enhance_frame_output.npy")).AsDoubles();

        var psnr = ImageMetrics.Psnr(actualAsDouble, expected);
        Assert.True(psnr > 30.0, $"EnhanceFrame PSNR {psnr:F2} dB fell at/below the 30 dB threshold — see class remarks.");
    }
}
