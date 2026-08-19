using FaceFusion.Face;
using FaceFusion.Types;
using FaceFusion.Vision;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Processors;

/// <summary>Python: <c>AgeModifierModel = Literal['fran', 'styleganex_age']</c>.</summary>
public enum AgeModifierModel
{
    [WireName("fran")]
    Fran,

    [WireName("styleganex_age")]
    StyleganexAge,
}

/// <summary>
/// Python: one entry of <c>age_modifier/core.py</c>'s <c>create_static_model_set</c>. Unlike
/// <c>fran</c> (a single template/size), <c>styleganex_age</c> needs a second template/size pair
/// for its background-inclusive crop (Python: <c>templates</c>/<c>sizes</c> dicts keyed
/// <c>'target'</c>/<c>'target_with_background'</c>) — both families share this one record shape,
/// with <see cref="TargetWithBackgroundTemplate"/>/<see cref="TargetWithBackgroundSize"/> unused
/// (<see langword="null"/>) for <c>fran</c>.
/// </summary>
public sealed record AgeModifierModelOptions(
    IReadOnlyDictionary<string, Download> Hashes,
    IReadOnlyDictionary<string, Download> Sources,
    WarpTemplate TargetTemplate,
    Size TargetSize,
    WarpTemplate? TargetWithBackgroundTemplate,
    Size? TargetWithBackgroundSize,
    float[] Mean,
    float[] StandardDeviation);

/// <summary>
/// Port of <c>facefusion/processors/modules/age_modifier/{core,types,choices}.py</c> — makes a
/// target face look older/younger via either the <c>fran</c> or <c>styleganex_age</c> model.
///
/// <para>
/// <b>No global state; sessions and settings taken as parameters (PORT_CONVENTIONS.md rule 5).</b>
/// Same reasoning as <c>FaceSwapper</c>/<c>ExpressionRestorer</c> — see
/// <see cref="AgeModifierInputs"/>.
/// </para>
///
/// <para>
/// <b>Parity coverage: <c>fran</c> only (real ONNX fixtures); <c>styleganex_age</c> is ported and
/// unit-tested but not ONNX-fixture-verified.</b> <c>fran</c> is the default model and the
/// smaller of the two ONNX families (see <c>tools/parity/dump_processors3.py</c>'s docstring for
/// the full reasoning, matching how <c>FaceSwapper</c> left <c>blendswap</c>/<c>uniface</c>
/// fixture-free). <see cref="ModifyAgeStyleganexAge"/>'s own arithmetic (the
/// <c>extend_affine_matrix</c> rescale, the <c>merge_matrix</c>-based occlusion-mask reprojection)
/// is unit-tested against hand-computed values instead.
/// </para>
///
/// <para>
/// <b>Mat / dtype conventions.</b> <c>VisionFrame</c>s are <see cref="Mat"/>, <c>CV_8UC3</c> BGR,
/// caller-owned, matching <c>FaceHelper</c>. <see cref="NormalizeVisionFrame"/> (the <c>fran</c>
/// path) deliberately does **not** narrow to <c>uint8</c> — Python's <c>normalize_vision_frame</c>
/// ends with <c>vision_frame[:, :, ::-1] * 255</c> and no <c>.astype</c> call at all, so the value
/// handed to <c>paste_back</c> stays float — and, per the same "bare Python list of floats
/// promotes float32*list to float64" rule <c>FaceSwapper.NormalizeCropFrame</c> already documents
/// and verified empirically there, the <c>* model_standard_deviation + model_mean</c> step
/// upcasts the array to <c>float64</c> even for <c>fran</c>'s trivial mean/std ([0,0,0]/[1,1,1]).
/// Reproduced as a returned <c>CV_64FC3</c> <see cref="Mat"/> and pasted back with a local
/// float-crop paste helper (<see cref="PasteBackFloatCrop"/>) — <c>FaceFusion.Face.FaceHelper.PasteBack</c>
/// is not reused for this one call site because it requires <c>CV_8UC3</c> (same gap
/// <c>FaceSwapper.PasteBackFloatCrop</c> already documents and works around for its own module;
/// duplicated here in miniature rather than exposing that method from <c>FaceSwapper.cs</c>, which
/// is out of this module's assignment to modify). <see cref="NormalizeExtendFrame"/> (the
/// <c>styleganex_age</c> path) narrows to <c>uint8</c> unconditionally, so that branch pastes with
/// the ordinary <c>FaceHelper.PasteBack</c>.
/// </para>
/// </summary>
public static class AgeModifier
{
    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>age_modifier_models = list(get_args(AgeModifierModel))</c>.</summary>
    public static readonly IReadOnlyList<AgeModifierModel> AgeModifierModels = Enum.GetValues<AgeModifierModel>();

    /// <summary>Python: <c>age_modifier_direction_range = create_int_range(-100, 100, 1)</c>.</summary>
    public static readonly IReadOnlyList<int> AgeModifierDirectionRange =
        FaceFusion.Core.CommonHelper.CreateIntRange(-100, 100, 1);

    // -----------------------------------------------------------------
    // create_static_model_set
    // -----------------------------------------------------------------

    private static readonly object ModelCatalogLock = new();
    private static IReadOnlyDictionary<AgeModifierModel, AgeModifierModelOptions>? _cachedModelCatalog;

    /// <summary>Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>).</summary>
    public static IReadOnlyDictionary<AgeModifierModel, AgeModifierModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        lock (ModelCatalogLock)
        {
            return _cachedModelCatalog ??= BuildModelCatalog();
        }
    }

    private static IReadOnlyDictionary<AgeModifierModel, AgeModifierModelOptions> BuildModelCatalog()
    {
        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];

        Download Component(string modelsBaseName, string fileName, string extension) => new(
            BuildDownloadUrl(githubProvider, modelsBaseName, fileName + extension),
            Path.Combine(modelsDirectory, fileName + extension));

        (IReadOnlyDictionary<string, Download> Hashes, IReadOnlyDictionary<string, Download> Sources) Component2(string modelsBaseName, string fileName)
        {
            var hashes = new Dictionary<string, Download> { ["age_modifier"] = Component(modelsBaseName, fileName, ".hash") };
            var sources = new Dictionary<string, Download> { ["age_modifier"] = Component(modelsBaseName, fileName, ".onnx") };
            return (hashes, sources);
        }

        var (franHashes, franSources) = Component2("models-3.6.0", "fran");
        var (styleganexHashes, styleganexSources) = Component2("models-3.1.0", "styleganex_age");

        return new Dictionary<AgeModifierModel, AgeModifierModelOptions>
        {
            [AgeModifierModel.Fran] = new AgeModifierModelOptions(
                franHashes, franSources,
                WarpTemplate.Ffhq512, new Size(1024, 1024),
                null, null,
                new[] { 0f, 0f, 0f }, new[] { 1f, 1f, 1f }),
            [AgeModifierModel.StyleganexAge] = new AgeModifierModelOptions(
                styleganexHashes, styleganexSources,
                WarpTemplate.Ffhq512, new Size(256, 256),
                WarpTemplate.Styleganex384, new Size(384, 384),
                new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }),
        };
    }

    /// <summary>Same reasoning/implementation as <c>FaceSwapper.ResolveModelsDirectory</c>.</summary>
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

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    // -----------------------------------------------------------------
    // pre_check (file-presence only)
    // -----------------------------------------------------------------

    /// <summary>Python: the <c>age_modifier</c>-specific half of <c>pre_check</c>.</summary>
    public static bool PreCheck(AgeModifierModel model)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[model];

        foreach (var download in modelOptions.Hashes.Values.Concat(modelOptions.Sources.Values))
        {
            if (!File.Exists(download.Path) || new FileInfo(download.Path).Length == 0)
            {
                return false;
            }
        }

        return true;
    }

    // -----------------------------------------------------------------
    // prepare_vision_frame / normalize_vision_frame / normalize_extend_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_vision_frame</c>. <paramref name="mean"/>/<paramref name="standardDeviation"/>
    /// are <see cref="AgeModifierModelOptions.Mean"/>/<see cref="AgeModifierModelOptions.StandardDeviation"/>.
    /// Computed in <see cref="double"/> then narrowed to <see cref="float"/> only at the end,
    /// matching numpy's promotion rules (same reasoning as <c>FaceSwapper.PrepareCropFrame</c>).
    /// </summary>
    public static float[] PrepareVisionFrame(Mat visionFrame, float[] mean, float[] standardDeviation)
    {
        var height = visionFrame.Rows;
        var width = visionFrame.Cols;
        var plane = height * width;
        var chw = new float[3 * plane];

        visionFrame.GetArray(out Vec3b[] pixels);

        for (var index = 0; index < plane; index++)
        {
            var pixel = pixels[index];
            var r = (pixel.Item2 / 255.0 - mean[0]) / standardDeviation[0];
            var g = (pixel.Item1 / 255.0 - mean[1]) / standardDeviation[1];
            var b = (pixel.Item0 / 255.0 - mean[2]) / standardDeviation[2];

            chw[index] = (float)r;
            chw[plane + index] = (float)g;
            chw[(2 * plane) + index] = (float)b;
        }

        return chw;
    }

    /// <summary>
    /// Python: <c>normalize_vision_frame</c> — the <c>fran</c> path. No <c>.astype(uint8)</c> in
    /// Python (see class remarks): stays float, returned here as <c>CV_64FC3</c> BGR (the
    /// <c>* model_standard_deviation + model_mean</c> list-arithmetic promotes to float64, per
    /// the same rule <c>FaceSwapper.NormalizeCropFrame</c> documents).
    /// </summary>
    public static Mat NormalizeVisionFrame(ReadOnlySpan<float> modelOutputChw, int height, int width, float[] mean, float[] standardDeviation)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        var result = new Mat(height, width, MatType.CV_64FC3);
        var data = new Vec3d[plane];

        for (var index = 0; index < plane; index++)
        {
            var r = (((double)modelOutputChw[index] * standardDeviation[0]) + mean[0]).ClampMinMax(0.0, 1.0);
            var g = (((double)modelOutputChw[plane + index] * standardDeviation[1]) + mean[1]).ClampMinMax(0.0, 1.0);
            var b = (((double)modelOutputChw[(2 * plane) + index] * standardDeviation[2]) + mean[2]).ClampMinMax(0.0, 1.0);

            // Python: `vision_frame.clip(0, 1)` then `[:, :, ::-1] * 255` (RGB -> BGR, no clip
            // applied after the *255 multiply, matching Python exactly — values can exceed
            // 255 momentarily is impossible here since the clip already bounds [0,1]).
            data[index] = new Vec3d(b * 255.0, g * 255.0, r * 255.0);
        }

        result.SetArray(data);
        return result;
    }

    /// <summary>
    /// Python: <c>normalize_extend_frame</c> — the <c>styleganex_age</c> path. Narrows to
    /// <c>uint8</c> unconditionally (Python: <c>.astype(numpy.uint8)</c>, truncation toward zero,
    /// not round — reproduced with a plain truncating cast since every value is non-negative
    /// after the clip), then <c>cv2.resize(..., interpolation = cv2.INTER_AREA)</c> up to
    /// <c>(targetSize * 4, targetSize * 4)</c>.
    /// </summary>
    public static Mat NormalizeExtendFrame(ReadOnlySpan<float> modelOutputChw, int height, int width, Size upscaledSize)
    {
        var plane = height * width;

        if (modelOutputChw.Length != 3 * plane)
        {
            throw new ArgumentException($"modelOutputChw has {modelOutputChw.Length} elements, expected {3 * plane} for a {width}x{height} CHW frame.", nameof(modelOutputChw));
        }

        using var raw = new Mat(height, width, MatType.CV_8UC3);
        var pixels = new Vec3b[plane];

        for (var index = 0; index < plane; index++)
        {
            // Python: `numpy.clip(x, -1, 1)`, then `(x + 1) / 2`, then `.transpose(1,2,0).clip(0,
            // 255)` (a no-op — the value is already in [0,1]), then `* 255.0`, then
            // `.astype(uint8)`, then `[:, :, ::-1]` (RGB -> BGR).
            var r = ToByteFromMinusOneToOne(modelOutputChw[index]);
            var g = ToByteFromMinusOneToOne(modelOutputChw[plane + index]);
            var b = ToByteFromMinusOneToOne(modelOutputChw[(2 * plane) + index]);

            pixels[index] = new Vec3b(b, g, r);
        }

        raw.SetArray(pixels);

        var resized = new Mat();
        Cv2.Resize(raw, resized, upscaledSize, 0, 0, InterpolationFlags.Area);
        return resized;
    }

    private static byte ToByteFromMinusOneToOne(float value)
    {
        var clipped = value < -1f ? -1f : (value > 1f ? 1f : value);
        var normalized = (clipped + 1f) / 2f;
        return (byte)(normalized * 255.0f);
    }

    // -----------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward</c>. Input names are resolved dynamically against
    /// <paramref name="ageModifierSession"/>'s own input metadata (Python: <c>for
    /// age_modifier_input in age_modifier.get_inputs()</c>), matching
    /// <c>FaceSwapper.ForwardSwapFace</c>'s precedent rather than assuming a fixed order.
    /// <paramref name="extendCropChw"/>/<paramref name="extendSize"/> are only used (and only
    /// non-null) for <see cref="AgeModifierModel.StyleganexAge"/> — <c>fran</c> has no
    /// <c>target_with_background</c> input, matching Python's conditional dict population.
    /// </summary>
    public static float[] Forward(
        InferenceSession ageModifierSession,
        ReadOnlySpan<float> cropChw, Size cropSize,
        ReadOnlySpan<float> extendCropChw, Size? extendSize,
        ReadOnlySpan<float> direction)
    {
        using var cropOrtValue = OrtValue.CreateTensorValueFromMemory(cropChw.ToArray(), new long[] { 1, 3, cropSize.Height, cropSize.Width });
        using var directionOrtValue = OrtValue.CreateTensorValueFromMemory(direction.ToArray(), new long[] { direction.Length });

        OrtValue? extendOrtValue = null;

        try
        {
            var inputs = new Dictionary<string, OrtValue>();
            foreach (var inputName in ageModifierSession.InputNames)
            {
                if (inputName == "target")
                {
                    inputs[inputName] = cropOrtValue;
                }
                else if (inputName == "target_with_background")
                {
                    if (extendSize is not { } size)
                    {
                        throw new ArgumentException("age_modifier session requires a 'target_with_background' input but extendSize was not provided.", nameof(extendSize));
                    }

                    extendOrtValue = OrtValue.CreateTensorValueFromMemory(extendCropChw.ToArray(), new long[] { 1, 3, size.Height, size.Width });
                    inputs[inputName] = extendOrtValue;
                }
                else if (inputName == "direction")
                {
                    inputs[inputName] = directionOrtValue;
                }
            }

            using var runOptions = new RunOptions();
            using var results = ageModifierSession.Run(runOptions, inputs, ageModifierSession.OutputNames);

            // Python: `age_modifier.run(None, age_modifier_inputs)[0][0]` — first output, batch
            // index 0.
            return results[0].GetTensorDataAsSpan<float>().ToArray();
        }
        finally
        {
            extendOrtValue?.Dispose();
        }
    }

    // -----------------------------------------------------------------
    // modify_age
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>modify_age</c>'s <c>fran</c> branch. Caller owns the returned <see cref="Mat"/>.
    /// Does not take ownership of <paramref name="tempVisionFrame"/>.
    /// </summary>
    public static Mat ModifyAgeFran(
        FaceFusion.Types.Face targetFace,
        Mat tempVisionFrame,
        InferenceSession ageModifierSession,
        int ageModifierDirectionSetting,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        double faceMaskBlur,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[AgeModifierModel.Fran];
        var faceLandmark5 = (float[,])((float[,])targetFace.LandmarkSet.FiveOn68).Clone();

        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5, modelOptions.TargetTemplate, modelOptions.TargetSize);
        using var cropDisposable = cropVisionFrame;
        using var affineDisposable = affineMatrix;

        var cropMasks = new List<Mat>();

        try
        {
            cropMasks.Add(FaceMasker.CreateBoxMask(cropVisionFrame, faceMaskBlur, new Padding(0, 0, 0, 0)));

            if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool));
            }

            var preparedCrop = PrepareVisionFrame(cropVisionFrame, modelOptions.Mean, modelOptions.StandardDeviation);

            // Python: `target_age = numpy.mean(target_face.age)` — target_face.age is a Python
            // range(start, stop); numpy.mean of a 2-element range is (start + stop) / 2.
            var targetAge = (targetFace.Age.Start.Value + targetFace.Age.End.Value) / 2.0;
            var direction = new[]
            {
                (float)Math.Clamp((targetAge) / 100.0, 0.0, 1.0),
                (float)Math.Clamp((targetAge + ageModifierDirectionSetting) / 100.0, 0.0, 1.0),
            };

            var forwardOutput = Forward(ageModifierSession, preparedCrop, modelOptions.TargetSize, ReadOnlySpan<float>.Empty, null, direction);

            using var normalizedCropVisionFrame = NormalizeVisionFrame(forwardOutput, modelOptions.TargetSize.Height, modelOptions.TargetSize.Width, modelOptions.Mean, modelOptions.StandardDeviation);

            using var cropMask = ReduceMinimumClip01(cropMasks);
            return PasteBackFloatCrop(tempVisionFrame, normalizedCropVisionFrame, cropMask, affineMatrix);
        }
        finally
        {
            foreach (var cropMask in cropMasks)
            {
                cropMask.Dispose();
            }
        }
    }

    /// <summary>
    /// Python: <c>modify_age</c>'s <c>styleganex_age</c> branch. Caller owns the returned
    /// <see cref="Mat"/>. Does not take ownership of <paramref name="tempVisionFrame"/>.
    ///
    /// <para>
    /// <b>Quirk reproduced deliberately.</b> Python's occlusion-mask branch computes
    /// <c>create_occlusion_mask(crop_vision_frame)</c> — the *256x256 target-template* crop, not
    /// <c>extend_vision_frame</c> (the 384x384 background-inclusive crop) — then reprojects it
    /// into the extend crop's coordinate space via <c>merge_matrix([extend_affine_matrix,
    /// invertAffineTransform(affine_matrix)])</c>. This reads like it should have used
    /// <c>extend_vision_frame</c> directly, but the Python source does not, so this port doesn't
    /// either — see PORT_CONVENTIONS.md rule 1.
    /// </para>
    /// </summary>
    public static Mat ModifyAgeStyleganexAge(
        FaceFusion.Types.Face targetFace,
        Mat tempVisionFrame,
        InferenceSession ageModifierSession,
        int ageModifierDirectionSetting,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        double faceMaskBlur,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1)
    {
        var modelOptions = CreateStaticModelSet(DownloadScope.Full)[AgeModifierModel.StyleganexAge];

        if (modelOptions.TargetWithBackgroundTemplate is not { } extendTemplate || modelOptions.TargetWithBackgroundSize is not { } extendSize)
        {
            throw new InvalidOperationException("styleganex_age model options are missing target_with_background template/size.");
        }

        var faceLandmark5 = (float[,])((float[,])targetFace.LandmarkSet.FiveOn68).Clone();

        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5, modelOptions.TargetTemplate, modelOptions.TargetSize);
        using var cropDisposable = cropVisionFrame;
        using var affineDisposable = affineMatrix;

        var extendFaceLandmark5 = FaceHelper.ScaleFaceLandmark5(faceLandmark5, 0.875);
        var (extendVisionFrame, extendAffineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, extendFaceLandmark5, extendTemplate, extendSize);
        using var extendDisposable = extendVisionFrame;
        var extendAffineMatrixOwned = extendAffineMatrix;

        using var extendVisionFrameRaw = extendVisionFrame.Clone();

        var cropMasks = new List<Mat>();

        try
        {
            cropMasks.Add(FaceMasker.CreateBoxMask(extendVisionFrame, faceMaskBlur, new Padding(0, 0, 0, 0)));

            if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
            {
                if (occluderInferencePool is null)
                {
                    throw new ArgumentNullException(nameof(occluderInferencePool), "FaceMaskType.Occlusion requires occluderInferencePool.");
                }

                // Quirk: computed against `cropVisionFrame`, not `extendVisionFrame` — see this
                // method's remarks.
                using var occlusionMaskRaw = FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool);

                using var invertedAffineMatrix = new Mat();
                Cv2.InvertAffineTransform(affineMatrix, invertedAffineMatrix);

                using var tempMatrix = FaceHelper.MergeMatrix(new[] { extendAffineMatrixOwned, invertedAffineMatrix });

                var occlusionMask = new Mat();
                Cv2.WarpAffine(occlusionMaskRaw, occlusionMask, tempMatrix, extendSize);
                cropMasks.Add(occlusionMask);
            }

            var preparedCrop = PrepareVisionFrame(cropVisionFrame, modelOptions.Mean, modelOptions.StandardDeviation);
            var preparedExtend = PrepareVisionFrame(extendVisionFrame, modelOptions.Mean, modelOptions.StandardDeviation);

            // Python: `numpy.array(numpy.interp(direction, [-100, 100], [2.5, -2.5])).astype(float32)`
            // — a linear map through the origin, computed directly rather than via
            // FaceFusion.Tensors.NumPy.Interp (float32-only) since Python's own `numpy.interp`
            // here runs on a bare int (state_manager's stored value) and float64 knots, i.e.
            // float64 throughout until the explicit final `.astype(float32)`.
            var direction = new[] { (float)(ageModifierDirectionSetting * -0.025) };

            var forwardOutput = Forward(ageModifierSession, preparedCrop, modelOptions.TargetSize, preparedExtend, extendSize, direction);

            var targetUpscaledSize = new Size(modelOptions.TargetSize.Width * 4, modelOptions.TargetSize.Height * 4);
            using var normalizedExtendFrame = NormalizeExtendFrame(forwardOutput, extendSize.Height, extendSize.Width, targetUpscaledSize);
            using var colorMatchedExtendFrame = VisionHelper.MatchFrameColor(extendVisionFrameRaw, normalizedExtendFrame);

            // Python: `extend_affine_matrix *= (model_sizes.get('target')[0] * 4) /
            // model_sizes.get('target_with_background')[0]` — scales the affine matrix's own
            // coefficients (not its output size) so the paste destination coordinates land in the
            // 4x-upscaled space.
            var rescale = (modelOptions.TargetSize.Width * 4.0) / extendSize.Width;
            using var rescaledExtendAffineMatrix = ScaleAffineMatrix(extendAffineMatrixOwned, rescale);

            using var cropMask = ReduceMinimumClip01(cropMasks);
            using var resizedCropMask = new Mat();
            Cv2.Resize(cropMask, resizedCropMask, targetUpscaledSize);

            return FaceHelper.PasteBack(tempVisionFrame, colorMatchedExtendFrame, resizedCropMask, rescaledExtendAffineMatrix);
        }
        finally
        {
            extendAffineMatrixOwned.Dispose();

            foreach (var cropMask in cropMasks)
            {
                cropMask.Dispose();
            }
        }
    }

    /// <summary>Multiplies every element of a 2x3 <c>CV_64F</c> affine matrix by a scalar. Caller owns the returned <see cref="Mat"/>.</summary>
    private static Mat ScaleAffineMatrix(Mat affineMatrix, double scale)
    {
        var result = new Mat(2, 3, MatType.CV_64FC1);
        for (var r = 0; r < 2; r++)
        {
            for (var c = 0; c < 3; c++)
            {
                result.Set(r, c, affineMatrix.At<double>(r, c) * scale);
            }
        }

        return result;
    }

    /// <summary>
    /// Python: <c>paste_back</c> called with a float <c>crop_vision_frame</c> (the <c>fran</c>
    /// path's <see cref="NormalizeVisionFrame"/> output — see class remarks for why this
    /// duplicates <c>FaceSwapper.PasteBackFloatCrop</c> rather than reusing it).
    /// </summary>
    private static Mat PasteBackFloatCrop(Mat tempVisionFrame, Mat cropVisionFrame, Mat cropVisionMask, Mat affineMatrix)
    {
        if (tempVisionFrame.Type() != MatType.CV_8UC3)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_8UC3 tempVisionFrame.", nameof(tempVisionFrame));
        }

        if (cropVisionFrame.Type() != MatType.CV_64FC3)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_64FC3 cropVisionFrame.", nameof(cropVisionFrame));
        }

        if (cropVisionMask.Type() != MatType.CV_32FC1)
        {
            throw new ArgumentException("PasteBackFloatCrop requires a CV_32FC1 mask.", nameof(cropVisionMask));
        }

        var (pasteBoundingBox, pasteMatrix) = FaceHelper.CalculatePasteArea(tempVisionFrame, cropVisionFrame, affineMatrix);
        using var _ = pasteMatrix;

        var x1 = pasteBoundingBox[0];
        var y1 = pasteBoundingBox[1];
        var x2 = pasteBoundingBox[2];
        var y2 = pasteBoundingBox[3];
        var pasteWidth = x2 - x1;
        var pasteHeight = y2 - y1;
        var pasteSize = new Size(pasteWidth, pasteHeight);

        using var inverseVisionMaskRaw = new Mat();
        Cv2.WarpAffine(cropVisionMask, inverseVisionMaskRaw, pasteMatrix, pasteSize);

        using var inverseVisionFrame = new Mat();
        Cv2.WarpAffine(cropVisionFrame, inverseVisionFrame, pasteMatrix, pasteSize, InterpolationFlags.Linear, BorderTypes.Replicate);

        var resultVisionFrame = tempVisionFrame.Clone();

        for (var row = 0; row < pasteHeight; row++)
        {
            for (var col = 0; col < pasteWidth; col++)
            {
                var maskValue = Math.Clamp(inverseVisionMaskRaw.At<float>(row, col), 0f, 1f);
                var oneMinusMask = 1.0 - maskValue;

                var destRow = y1 + row;
                var destCol = x1 + col;
                var original = resultVisionFrame.At<Vec3b>(destRow, destCol);
                var warped = inverseVisionFrame.At<Vec3d>(row, col);

                var blended = new Vec3b
                {
                    Item0 = BlendChannel(original.Item0, warped.Item0, oneMinusMask, maskValue),
                    Item1 = BlendChannel(original.Item1, warped.Item1, oneMinusMask, maskValue),
                    Item2 = BlendChannel(original.Item2, warped.Item2, oneMinusMask, maskValue),
                };

                resultVisionFrame.Set(destRow, destCol, blended);
            }
        }

        return resultVisionFrame;
    }

    private static byte BlendChannel(byte original, double warped, double oneMinusMask, double mask)
    {
        var value = ((double)original * oneMinusMask) + (warped * mask);

        if (value <= 0)
        {
            return 0;
        }

        if (value >= 255)
        {
            return 255;
        }

        return (byte)value;
    }

    /// <summary>Python: <c>numpy.minimum.reduce(crop_masks).clip(0, 1)</c>. Caller owns the returned <see cref="Mat"/> (<c>CV_32FC1</c>).</summary>
    private static Mat ReduceMinimumClip01(IReadOnlyList<Mat> masks)
    {
        if (masks.Count == 0)
        {
            throw new ArgumentException("at least one face mask must be present.", nameof(masks));
        }

        var result = masks[0].Clone();
        for (var i = 1; i < masks.Count; i++)
        {
            Cv2.Min(result, masks[i], result);
        }

        Cv2.Min(result, new Scalar(1.0), result);
        Cv2.Max(result, new Scalar(0.0), result);
        return result;
    }

    /// <summary>Python: <c>modify_age</c>. Dispatches on <paramref name="model"/>; the (unreachable in Python's own literal-typed model) "no match" fallthrough returns <paramref name="tempVisionFrame"/> unchanged, matching Python's trailing <c>return temp_vision_frame</c>.</summary>
    public static Mat ModifyAge(
        AgeModifierModel model,
        FaceFusion.Types.Face targetFace,
        Mat tempVisionFrame,
        InferenceSession ageModifierSession,
        int ageModifierDirectionSetting,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        double faceMaskBlur,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool = null,
        FaceOccluderModel faceOccluderModel = FaceOccluderModel.Xseg1)
    {
        return model switch
        {
            AgeModifierModel.Fran => ModifyAgeFran(targetFace, tempVisionFrame, ageModifierSession, ageModifierDirectionSetting, faceMaskTypes, faceMaskBlur, occluderInferencePool, faceOccluderModel),
            AgeModifierModel.StyleganexAge => ModifyAgeStyleganexAge(targetFace, tempVisionFrame, ageModifierSession, ageModifierDirectionSetting, faceMaskTypes, faceMaskBlur, occluderInferencePool, faceOccluderModel),
            _ => tempVisionFrame,
        };
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>facefusion.processors.modules.age_modifier.core</c>'s per-call inputs, extended
    /// per <see cref="IProcessorInputs"/>'s remarks — see each field's comment for the Python
    /// <c>state_manager</c> key/session it replaces.
    /// </summary>
    public sealed record AgeModifierInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        AgeModifierModel Model,
        int AgeModifierDirection,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        double FaceMaskBlur,
        InferenceSession AgeModifierSession,
        FaceSelectorMode FaceSelectorMode,
        double FaceTrackerScore,
        FaceSelectorOrder FaceSelectorOrder,
        FaceSelectorGender? FaceSelectorGender,
        FaceSelectorRace? FaceSelectorRace,
        int? FaceSelectorAgeStart,
        int? FaceSelectorAgeEnd,
        int ReferenceFacePosition,
        double ReferenceFaceDistance,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> GetStaticFaces,
        Func<IReadOnlyList<FaceFusion.Types.Face?>, IReadOnlyList<FaceFusion.Types.Face>> RefillFaces) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/age_modifier/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract — see <c>FaceSwapper.Processor</c>
    /// for the same pattern.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "age_modifier";

        /// <inheritdoc />
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>
        /// Python: the <c>age_modifier</c>-specific half of <c>pre_check</c>. Needs
        /// <paramref name="model"/> since there is no <c>state_manager</c> to read it from — same
        /// gap <c>FaceSwapper.Processor.PreCheck</c> documents.
        /// </summary>
        public bool PreCheck(AgeModifierModel model) => AgeModifier.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "age_modifier.PreCheck requires an AgeModifierModel (no state_manager to read it from — call the AgeModifierModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Same scope note as <c>FaceSwapper.Processor.PreProcess</c>
        /// — <c>facefusion/filesystem.py</c> checks are out of this module's assignment.
        /// </summary>
        public bool PreProcess(ProcessMode mode, ProcessorRunPaths paths)
        {
            _ = mode;
            _ = paths;
            return true;
        }

        /// <inheritdoc />
        public ProcessorOutputs ProcessFrame(IProcessorInputs inputs)
        {
            if (inputs is not AgeModifierInputs ageModifierInputs)
            {
                throw new ArgumentException($"expected {nameof(AgeModifierInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return AgeModifier.ProcessFrame(ageModifierInputs);
        }

        /// <inheritdoc />
        public void PostProcess()
        {
        }
    }

    // -----------------------------------------------------------------
    // process_frame orchestration
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Returns the (possibly unchanged, if no target face was
    /// found) frame and mask, same ownership convention as <c>FaceSwapper.ProcessFrame</c>.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(AgeModifierInputs inputs)
    {
        var targetVisionFrame = FaceFusion.Core.CommonHelper.GetMiddle(inputs.TargetVisionFrames);
        var targetFaces = FaceSelector.SelectFaces(
            inputs.ReferenceVisionFrame,
            inputs.SourceVisionFrames,
            inputs.TargetVisionFrames,
            inputs.FaceSelectorMode,
            inputs.FaceTrackerScore,
            inputs.FaceSelectorOrder,
            inputs.FaceSelectorGender,
            inputs.FaceSelectorRace,
            inputs.FaceSelectorAgeStart,
            inputs.FaceSelectorAgeEnd,
            inputs.ReferenceFacePosition,
            inputs.ReferenceFaceDistance,
            inputs.GetStaticFaces,
            inputs.RefillFaces);

        var tempVisionFrame = inputs.TempVisionFrame;

        if (targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            foreach (var rawTargetFace in targetFaces)
            {
                var targetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, tempVisionFrame);
                var nextTempVisionFrame = ModifyAge(
                    inputs.Model,
                    targetFace,
                    tempVisionFrame,
                    inputs.AgeModifierSession,
                    inputs.AgeModifierDirection,
                    inputs.FaceMaskTypes,
                    inputs.FaceMaskBlur);

                if (!ReferenceEquals(tempVisionFrame, inputs.TempVisionFrame))
                {
                    tempVisionFrame.Dispose();
                }

                tempVisionFrame = nextTempVisionFrame;
            }
        }

        return new ProcessorOutputs(tempVisionFrame, inputs.TempVisionMask);
    }
}

internal static class DoubleClampExtensions
{
    public static double ClampMinMax(this double value, double min, double max) => value < min ? min : (value > max ? max : value);
}
