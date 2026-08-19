using FaceFusion.Face;
using FaceFusion.Parity;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.ParityTests;

/// <summary>
/// Cross-language parity for the two model-backed masks in <c>FaceFusion.Face.FaceMasker</c>
/// (Python: <c>facefusion.face_masker.create_occlusion_mask</c>/<c>create_region_mask</c>),
/// against the real <c>xseg_1</c>/<c>bisenet_resnet_18</c> ONNX models. The pure, model-free
/// <c>create_box_mask</c>/<c>create_area_mask</c> are covered with real Python ground truth in
/// <c>tests/FaceFusion.UnitTests/FaceMaskerTests.cs</c> instead (no ONNX session needed there).
///
/// <para>
/// <b>Ground truth.</b> Captured ad hoc (there is no <c>tools/parity/dump_face_masker.py</c>):
/// a 256x256 resize of <c>source.jpg</c> fed directly to the real
/// <c>facefusion.face_masker.create_occlusion_mask</c>/<c>create_region_mask</c> with
/// <c>face_occluder_model = 'xseg_1'</c>, <c>face_parser_model = 'bisenet_resnet_18'</c> (both
/// present locally, unlike the <c>xseg_2</c>/<c>xseg_3</c>/<c>bisenet_resnet_34</c> variants the
/// module also supports) and <c>face_mask_regions = ['skin', 'nose', 'mouth']</c> for the region
/// mask. The input does not need to be a real face-aligned crop — both functions are pure
/// per-pixel model calls plus a resize/blur, with no assumption baked in about what the pixels
/// depict — so a plain resized photo exercises the exact same code path a real aligned crop
/// would.
/// </para>
///
/// <para>
/// <b>Tolerance.</b> Both masks are ONNX Runtime output post-processed with OpenCV arithmetic
/// only (clip, resize, Gaussian blur) — per PARITY_HARNESS.md, expect ~0 divergence. Asserted
/// with <see cref="TensorComparison"/> at <c>rtol = 1e-4, atol = 1e-4</c>, loose enough to
/// absorb the documented OpenCvSharp/opencv-python resize-interpolation non-defect, tight
/// enough to catch a real preprocessing or postprocessing bug.
/// </para>
/// </summary>
[Collection("NativeInference")]
public sealed class FaceMaskerParityTests
{
    private static string FixturesDirectory =>
        Path.Combine(System.AppContext.BaseDirectory, "fixtures", "face_masker");

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

    private static NpyArray LoadNpy(string fileName) =>
        NpyReader.Load(Path.Combine(FixturesDirectory, fileName));

    private static Mat LoadCrop()
    {
        var array = LoadNpy("crop.npy");
        var height = array.Shape[0];
        var width = array.Shape[1];
        var raw = array.RawData;
        var pixels = new Vec3b[height * width];

        for (var i = 0; i < pixels.Length; i++)
        {
            var offset = i * 3;
            pixels[i] = new Vec3b { Item0 = raw[offset], Item1 = raw[offset + 1], Item2 = raw[offset + 2] };
        }

        var mat = new Mat(height, width, MatType.CV_8UC3);
        mat.SetArray(pixels);
        return mat;
    }

    private static double[] MatToDoubles(Mat mat)
    {
        mat.GetArray(out float[] flat);
        var result = new double[flat.Length];
        for (var i = 0; i < flat.Length; i++)
        {
            result[i] = flat[i];
        }

        return result;
    }

    [FaceMaskerModelFact("xseg_1.onnx")]
    public void CreateOcclusionMaskMatchesPython()
    {
        using var session = new InferenceSession(FindModelPath("xseg_1.onnx"));
        var pool = new Dictionary<string, InferenceSession> { ["xseg_1"] = session };

        using var crop = LoadCrop();
        using var mask = FaceMasker.CreateOcclusionMask(crop, FaceOccluderModel.Xseg1, pool);

        Assert.Equal(256, mask.Rows);
        Assert.Equal(256, mask.Cols);

        var expected = LoadNpy("occlusion_mask.npy").AsDoubles();
        var result = TensorComparison.Compare(MatToDoubles(mask), expected, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, result.Describe());
    }

    [FaceMaskerModelFact("bisenet_resnet_18.onnx")]
    public void CreateRegionMaskMatchesPython()
    {
        using var session = new InferenceSession(FindModelPath("bisenet_resnet_18.onnx"));
        var pool = new Dictionary<string, InferenceSession> { ["bisenet_resnet_18"] = session };

        using var crop = LoadCrop();
        var regions = new[] { FaceMaskRegion.Skin, FaceMaskRegion.Nose, FaceMaskRegion.Mouth };
        using var mask = FaceMasker.CreateRegionMask(crop, regions, FaceParserModel.BisenetResnet18, pool);

        Assert.Equal(256, mask.Rows);
        Assert.Equal(256, mask.Cols);

        var expected = LoadNpy("region_mask.npy").AsDoubles();
        var result = TensorComparison.Compare(MatToDoubles(mask), expected, relativeTolerance: 1e-4, absoluteTolerance: 1e-4);
        Assert.True(result.Passed, result.Describe());
    }
}

/// <summary>
/// <c>[Fact]</c> that skips at discovery time when the named <c>.assets/models/*.onnx</c> file
/// is not present — same pattern as <c>ContentAnalyserModelFactAttribute</c>.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class FaceMaskerModelFactAttribute : FactAttribute
{
    public FaceMaskerModelFactAttribute(string modelFileName)
    {
        if (!FaceMaskerParityTests.ModelAvailable(modelFileName))
        {
            Skip = $"requires .assets/models/{modelFileName} (gitignored, not present in CI) — " +
                   "populate via the real Python face_masker.pre_check() with network access, then retry";
        }
    }
}
