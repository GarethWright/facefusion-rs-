using FaceFusion.Core;
using FaceFusion.Inference;
using FaceFusion.Tensors;
using FaceFusion.Types;
using FaceFusion.Vision;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Python: <c>facefusion/processors/modules/frame_colorizer/types.py</c>'s
/// <c>FrameColorizerModel = Literal['ddcolor', 'ddcolor_artistic', 'deoldify',
/// 'deoldify_artistic', 'deoldify_stable']</c>. Declared here (not <c>FaceFusion.Types</c>) per
/// the file-scope constraint — only <c>FrameColorizer.cs</c> is this agent's to touch for this
/// model, matching how <c>FaceEnhancer.cs</c> declares <c>FaceEnhancerModel</c> locally.
/// </summary>
public enum FrameColorizerModel
{
    [WireName("ddcolor")]
    Ddcolor,

    [WireName("ddcolor_artistic")]
    DdcolorArtistic,

    [WireName("deoldify")]
    Deoldify,

    [WireName("deoldify_artistic")]
    DeoldifyArtistic,

    [WireName("deoldify_stable")]
    DeoldifyStable,
}

/// <summary>
/// Python: the model set's <c>'type'</c> field — either <c>'ddcolor'</c> (predicts Lab a/b
/// channels only) or <c>'deoldify'</c> (predicts a full 3-channel image, then recombined through
/// a second, deliberately odd Lab round-trip — see <see cref="FrameColorizer.MergeColorFrame"/>).
/// </summary>
public enum FrameColorizerModelType
{
    Ddcolor,
    Deoldify,
}

/// <summary>
/// Port of <c>facefusion/processors/modules/frame_colorizer/{core,types,choices}.py</c> —
/// colorizes a grayscale-looking frame by running an ONNX colorization model
/// (DDColor/DeOldify family) and blending the result back over the original frame.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5); plain static methods, no <c>IProcessor</c>
/// wiring yet.</b> Same posture as <see cref="FaceEnhancer"/> at the time this file was written:
/// <c>IProcessor</c>/<c>ProcessorRegistry</c> exist in this project now, but no processor has
/// wired into them yet (no <c>FaceSwapper</c> reference implementation has landed), so this stays
/// a set of plain static methods mirroring <see cref="FaceFusion.Face.FaceDetector"/>'s shape.
/// <see cref="ProcessFrame"/>'s return type is already <see cref="ProcessorOutputs"/>, so wiring
/// into <c>IProcessor.ProcessFrame</c> later is a thin adapter, not a rewrite. Every Python
/// <c>state_manager.get_item(...)</c> call (<c>frame_colorizer_model</c>,
/// <c>frame_colorizer_size</c>, <c>frame_colorizer_blend</c>) becomes an explicit parameter.
/// </para>
///
/// <para>
/// <b>Reduced-scope <c>pre_check</c>/model set</b> (same shape as <see cref="FaceEnhancer"/>/
/// <see cref="FaceFusion.Face.FaceDetector"/>): checks only this module's own hash/source files
/// on disk; does not download or verify hashes (<c>download.py</c>/<c>hash_helper.py</c> are not
/// ported anywhere in this repo).
/// </para>
///
/// <para>
/// <b>The Lab colour-space arithmetic — the highest-risk part of this file, per the assignment
/// brief.</b> <c>prepare_temp_frame</c>/<c>merge_color_frame</c> round-trip through
/// <c>cv2.cvtColor</c>'s <c>RGB2LAB</c>/<c>LAB2RGB</c>/<c>BGR2LAB</c>/<c>LAB2BGR</c> conversions
/// on <c>float32</c> Mats — OpenCV's float Lab convention is <c>L in [0, 100]</c>, <c>a, b in
/// about [-127, 127]</c>, and this port calls <c>Cv2.CvtColor</c> with the matching
/// <see cref="ColorConversionCodes"/> members directly (same native OpenCV kernel Python's cv2
/// binding calls), rather than re-deriving the Lab math by hand — so there is no separate scale
/// factor to get wrong on the .NET side beyond picking the right enum member and feeding it
/// data at the range it expects, which the parity tests verify against the model-input tensor
/// directly (see <c>tests/FaceFusion.ParityTests/FrameColorizerParityTests.cs</c>).
/// </para>
///
/// <para>
/// <b>The <c>deoldify</c> branch's channel-index quirk (reproduced deliberately, per
/// PORT_CONVENTIONS.md rule 1).</b> Python's <c>merge_color_frame</c> for
/// <c>model_type == 'deoldify'</c> converts the model's 3-channel output with
/// <c>COLOR_BGR2RGB</c> (a plain channel-0/channel-2 swap — the array is not actually BGR at
/// that point, it is whatever channel order the model emitted), casts to <c>uint8</c>
/// (truncating, not rounding — Python's bare <c>.astype(uint8)</c>, no preceding <c>.round()</c>
/// unlike the <c>ddcolor</c> branch), then feeds the swapped/truncated array through
/// <c>COLOR_BGR2LAB</c> a second time (again not actually a BGR image), keeps only that
/// conversion's <c>a</c>/<c>b</c> channels, and merges them with the *original* frame's own blue
/// channel standing in for <c>L</c>, before a final <c>COLOR_LAB2BGR</c>. This is reproduced
/// index-for-index in <see cref="MergeColorFrame"/> rather than "fixed" to a more sensible
/// pipeline — see <see cref="MergeColorFrame"/>'s own remarks for the exact operation sequence.
/// This port builds each output channel as its own single-channel <see cref="Mat"/>
/// (<see cref="SplitChwToChannelMats"/>) and merges/reorders with <c>Cv2.Merge</c>/
/// <c>Cv2.CvtColor</c> rather than materialising an HWC array by hand, which keeps every
/// channel-index operation identical to Python's array-index operations while staying entirely
/// in <see cref="Mat"/> (native memory, no managed per-pixel array beyond the raw ONNX output).
/// </para>
///
/// <para>
/// <b>VisionFrame / Mask representation</b> — same convention as every other ported module:
/// <see cref="Mat"/>, native memory, every returned <see cref="Mat"/> caller-owned, parameters
/// never disposed by the callee unless documented otherwise.
/// </para>
/// </summary>
public static class FrameColorizer
{
    private const string ModuleName = "facefusion.processors.modules.frame_colorizer.core";
    private const string ModelBaseName = "models-3.0.0";

    /// <summary>One entry of Python's <c>create_static_model_set('full')</c> — the fields this
    /// port needs (<c>type</c>, <c>hashes.frame_colorizer</c>, <c>sources.frame_colorizer</c>);
    /// the <c>__metadata__</c> vendor/license/year entries are download-manifest bookkeeping
    /// with no behavioural effect, same reduced scope as <see cref="FaceEnhancer.ModelOptions"/>.</summary>
    public sealed record ModelOptions(FrameColorizerModelType Type, Download Hash, Download Source);

    private static readonly IReadOnlyList<FrameColorizerModel> AllModels = Enum.GetValues<FrameColorizerModel>();

    private static readonly IReadOnlyDictionary<FrameColorizerModel, string> ModelFileNames = new Dictionary<FrameColorizerModel, string>
    {
        [FrameColorizerModel.Ddcolor] = "ddcolor",
        [FrameColorizerModel.DdcolorArtistic] = "ddcolor_artistic",
        [FrameColorizerModel.Deoldify] = "deoldify",
        [FrameColorizerModel.DeoldifyArtistic] = "deoldify_artistic",
        [FrameColorizerModel.DeoldifyStable] = "deoldify_stable",
    };

    private static readonly IReadOnlyDictionary<FrameColorizerModel, FrameColorizerModelType> ModelTypes = new Dictionary<FrameColorizerModel, FrameColorizerModelType>
    {
        [FrameColorizerModel.Ddcolor] = FrameColorizerModelType.Ddcolor,
        [FrameColorizerModel.DdcolorArtistic] = FrameColorizerModelType.Ddcolor,
        [FrameColorizerModel.Deoldify] = FrameColorizerModelType.Deoldify,
        [FrameColorizerModel.DeoldifyArtistic] = FrameColorizerModelType.Deoldify,
        [FrameColorizerModel.DeoldifyStable] = FrameColorizerModelType.Deoldify,
    };

    // -----------------------------------------------------------------
    // choices.py
    // -----------------------------------------------------------------

    /// <summary>Python: <c>frame_colorizer_models</c>.</summary>
    public static IReadOnlyList<FrameColorizerModel> FrameColorizerModels => AllModels;

    /// <summary>Python: <c>frame_colorizer_sizes</c>.</summary>
    public static readonly IReadOnlyList<string> FrameColorizerSizes = new[] { "192x192", "256x256", "384x384", "512x512" };

    /// <summary>Python: <c>frame_colorizer_blend_range</c> (<c>create_int_range(0, 100, 1)</c>).</summary>
    public static readonly IReadOnlyList<int> FrameColorizerBlendRange = CommonHelper.CreateIntRange(0, 100, 1);

    // -----------------------------------------------------------------
    // Model set / downloads / pre_check
    // -----------------------------------------------------------------

    /// <summary>Python: <c>create_static_model_set</c> (<c>@lru_cache()</c>).</summary>
    public static IReadOnlyDictionary<FrameColorizerModel, ModelOptions> CreateStaticModelSet(DownloadScope downloadScope)
    {
        _ = downloadScope;

        var modelsDirectory = ResolveModelsDirectory();
        var githubProvider = Choices.DownloadProviderSet[DownloadProvider.Github];
        var result = new Dictionary<FrameColorizerModel, ModelOptions>();

        foreach (var model in AllModels)
        {
            var fileName = ModelFileNames[model];
            var hash = new Download(
                BuildDownloadUrl(githubProvider, ModelBaseName, fileName + ".hash"),
                Path.Combine(modelsDirectory, fileName + ".hash"));
            var source = new Download(
                BuildDownloadUrl(githubProvider, ModelBaseName, fileName + ".onnx"),
                Path.Combine(modelsDirectory, fileName + ".onnx"));

            result[model] = new ModelOptions(ModelTypes[model], hash, source);
        }

        return result;
    }

    private static string BuildDownloadUrl(DownloadProviderValue provider, string baseName, string fileName)
        => provider.Urls[0] + provider.Path.Replace("{base_name}", baseName).Replace("{file_name}", fileName);

    /// <summary>Same repo-root-walking approach as <see cref="FaceFusion.Face.FaceDetector"/>'s
    /// private helper of the same name.</summary>
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

    /// <summary>Python: <c>get_model_options</c>.</summary>
    public static ModelOptions GetModelOptions(FrameColorizerModel frameColorizerModel)
        => CreateStaticModelSet(DownloadScope.Full)[frameColorizerModel];

    /// <summary>Python: <c>pre_check</c>. See the class remarks — checks only this module's own
    /// hash/source files, not the <c>content_analyser</c> common-module pre-check (out of
    /// scope).</summary>
    public static bool PreCheck(FrameColorizerModel frameColorizerModel)
    {
        var options = GetModelOptions(frameColorizerModel);
        return FileSystem.IsFile(options.Hash.Path) && FileSystem.IsFile(options.Source.Path);
    }

    // -----------------------------------------------------------------
    // Inference pool
    // -----------------------------------------------------------------

    /// <summary>Python: <c>get_inference_pool</c>.</summary>
    public static IReadOnlyDictionary<string, InferenceSession> GetInferencePool(
        InferenceManager inferenceManager,
        FrameColorizerModel frameColorizerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var options = GetModelOptions(frameColorizerModel);
        var modelSourceSet = new Dictionary<string, Download> { ["frame_colorizer"] = options.Source };
        var modelNames = new[] { frameColorizerModel.ToWireName() };
        return inferenceManager.GetInferencePool(ModuleName, modelNames, modelSourceSet, executionDeviceIds, executionProviders);
    }

    /// <summary>Python: <c>clear_inference_pool</c>.</summary>
    public static void ClearInferencePool(
        InferenceManager inferenceManager,
        FrameColorizerModel frameColorizerModel,
        IReadOnlyList<int> executionDeviceIds,
        IReadOnlyList<ExecutionProvider> executionProviders)
    {
        var modelNames = new[] { frameColorizerModel.ToWireName() };
        inferenceManager.ClearInferencePool(ModuleName, modelNames, executionDeviceIds, executionProviders);
    }

    // -----------------------------------------------------------------
    // prepare_temp_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>prepare_temp_frame</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>. Returns the flat <c>(1, 3, modelSize.Height,
    /// modelSize.Width)</c> CHW float32 model input.
    /// </summary>
    public static float[] PrepareTempFrame(Mat tempVisionFrame, FrameColorizerModelType modelType, Resolution modelSize)
    {
        using var gray = new Mat();
        Cv2.CvtColor(tempVisionFrame, gray, ColorConversionCodes.BGR2GRAY);

        using var grayRgb = new Mat();
        Cv2.CvtColor(gray, grayRgb, ColorConversionCodes.GRAY2RGB);

        Mat prepared;
        var disposePrepared = false;

        if (modelType == FrameColorizerModelType.Ddcolor)
        {
            using var floatRgb = new Mat();
            grayRgb.ConvertTo(floatRgb, MatType.CV_32FC3, 1.0 / 255.0);

            using var lab = new Mat();
            Cv2.CvtColor(floatRgb, lab, ColorConversionCodes.RGB2Lab);

            var labChannels = Cv2.Split(lab);
            using var lChannel = labChannels[0];
            labChannels[1].Dispose();
            labChannels[2].Dispose();

            using var zero = Mat.Zeros(lChannel.Rows, lChannel.Cols, MatType.CV_32FC1).ToMat();
            using var combinedLab = new Mat();
            Cv2.Merge(new[] { lChannel, zero, zero }, combinedLab);

            prepared = new Mat();
            Cv2.CvtColor(combinedLab, prepared, ColorConversionCodes.Lab2RGB);
            disposePrepared = true;
        }
        else
        {
            prepared = grayRgb;
        }

        try
        {
            using var resized = new Mat();
            Cv2.Resize(prepared, resized, new Size(modelSize.Width, modelSize.Height));

            using var resizedFloat = new Mat();
            resized.ConvertTo(resizedFloat, MatType.CV_32FC3);

            resizedFloat.GetArray(out Vec3f[] hwcPixels);
            var hwcData = new float[hwcPixels.Length * 3];
            for (var i = 0; i < hwcPixels.Length; i++)
            {
                var offset = i * 3;
                hwcData[offset] = hwcPixels[i].Item0;
                hwcData[offset + 1] = hwcPixels[i].Item1;
                hwcData[offset + 2] = hwcPixels[i].Item2;
            }

            return NumPy.TransposeHwcToChw(hwcData, modelSize.Height, modelSize.Width, 3);
        }
        finally
        {
            if (disposePrepared)
            {
                prepared.Dispose();
            }
        }
    }

    // -----------------------------------------------------------------
    // forward
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>forward</c>. Does not take ownership of
    /// <paramref name="frameColorizerSession"/>. Returns the flat CHW output plus its
    /// dynamically-read channel count/height/width (2 channels for <c>ddcolor</c> — Lab a/b
    /// only — 3 for <c>deoldify</c> — a full image).
    /// </summary>
    public static (float[] Data, int Channels, int Height, int Width) Forward(float[] chwInput, int height, int width, InferenceSession frameColorizerSession)
    {
        using var inputOrtValue = OrtValue.CreateTensorValueFromMemory(chwInput, new long[] { 1, 3, height, width });
        var inputs = new Dictionary<string, OrtValue> { ["input"] = inputOrtValue };

        using var runOptions = new RunOptions();
        using var results = frameColorizerSession.Run(runOptions, inputs, frameColorizerSession.OutputNames);

        var outputInfo = results[0].GetTensorTypeAndShape();
        var outputShape = outputInfo.Shape;
        var outputChannels = checked((int)outputShape[1]);
        var outputHeight = checked((int)outputShape[2]);
        var outputWidth = checked((int)outputShape[3]);

        var data = results[0].GetTensorDataAsSpan<float>().ToArray();
        return (data, outputChannels, outputHeight, outputWidth);
    }

    // -----------------------------------------------------------------
    // merge_color_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>merge_color_frame</c>. Does not take ownership of
    /// <paramref name="tempVisionFrame"/>. Caller owns the returned <see cref="Mat"/>
    /// (<c>CV_8UC3</c>, BGR, sized to <paramref name="tempVisionFrame"/>). See the class remarks
    /// for the deliberately-reproduced <c>deoldify</c> channel-index quirk.
    /// </summary>
    public static Mat MergeColorFrame(Mat tempVisionFrame, float[] outputChwData, int outputChannels, int outputHeight, int outputWidth, FrameColorizerModelType modelType)
    {
        var tempSize = new Size(tempVisionFrame.Cols, tempVisionFrame.Rows);
        var channelMats = SplitChwToChannelMats(outputChwData, outputChannels, outputHeight, outputWidth);

        try
        {
            var resizedChannels = new Mat[outputChannels];
            for (var c = 0; c < outputChannels; c++)
            {
                resizedChannels[c] = new Mat();
                Cv2.Resize(channelMats[c], resizedChannels[c], tempSize);
            }

            try
            {
                return modelType == FrameColorizerModelType.Ddcolor
                    ? MergeDdcolor(tempVisionFrame, resizedChannels)
                    : MergeDeoldify(tempVisionFrame, resizedChannels);
            }
            finally
            {
                foreach (var mat in resizedChannels)
                {
                    mat.Dispose();
                }
            }
        }
        finally
        {
            foreach (var mat in channelMats)
            {
                mat.Dispose();
            }
        }
    }

    /// <summary>
    /// The <c>model_type == 'ddcolor'</c> branch of <c>merge_color_frame</c>: takes the L
    /// channel from the *original* frame (normalised to <c>[0, 1]</c> BGR first, per Python's
    /// <c>(temp_vision_frame / 255.0).astype(float32)</c>), combines it with the model's
    /// resized a/b channels into a Lab image, converts back to BGR, then
    /// <c>(x * 255.0).round().astype(uint8)</c>.
    /// </summary>
    private static Mat MergeDdcolor(Mat tempVisionFrame, Mat[] resizedChannels)
    {
        using var tempNorm = new Mat();
        tempVisionFrame.ConvertTo(tempNorm, MatType.CV_32FC3, 1.0 / 255.0);

        using var tempLab = new Mat();
        Cv2.CvtColor(tempNorm, tempLab, ColorConversionCodes.BGR2Lab);

        var tempLabChannels = Cv2.Split(tempLab);
        using var lChannel = tempLabChannels[0];
        tempLabChannels[1].Dispose();
        tempLabChannels[2].Dispose();

        using var combinedLab = new Mat();
        Cv2.Merge(new[] { lChannel, resizedChannels[0], resizedChannels[1] }, combinedLab);

        using var resultFloatBgr = new Mat();
        Cv2.CvtColor(combinedLab, resultFloatBgr, ColorConversionCodes.Lab2BGR);

        using var scaled = new Mat();
        Cv2.Multiply(resultFloatBgr, new Scalar(255.0, 255.0, 255.0), scaled);

        return AstypeUInt8(scaled, round: true);
    }

    /// <summary>
    /// The <c>model_type == 'deoldify'</c> branch of <c>merge_color_frame</c>. Reproduces
    /// Python's operation sequence index-for-index (see the class remarks):
    /// <list type="number">
    /// <item><description><c>cv2.cvtColor(color_vision_frame, COLOR_BGR2RGB)</c> — swaps
    /// channel 0 and channel 2 of the model's (resized) 3-channel output; no value change.</description></item>
    /// <item><description><c>.astype(uint8)</c> — a bare truncating cast, <b>not</b> preceded
    /// by <c>.round()</c> (unlike the <c>ddcolor</c> branch above).</description></item>
    /// <item><description><c>cv2.cvtColor(..., COLOR_BGR2LAB)</c> on that swapped/truncated
    /// array — again just an index operation on whatever the array happens to hold.</description></item>
    /// <item><description>Keep only the resulting a/b channels (<c>cv2.split</c>'s second/third
    /// outputs); discard its L.</description></item>
    /// <item><description><c>cv2.merge</c> the *original* frame's own blue channel in as if it
    /// were an L channel, with the a/b channels from step 4.</description></item>
    /// <item><description><c>cv2.cvtColor(..., COLOR_LAB2BGR)</c> — the final result, already
    /// <c>uint8</c>, no further scaling.</description></item>
    /// </list>
    /// </summary>
    private static Mat MergeDeoldify(Mat tempVisionFrame, Mat[] resizedChannels)
    {
        using var colorMat = new Mat();
        Cv2.Merge(resizedChannels, colorMat);

        using var swapped = new Mat();
        Cv2.CvtColor(colorMat, swapped, ColorConversionCodes.BGR2RGB);

        using var swappedUInt8 = AstypeUInt8(swapped, round: false);

        using var lab = new Mat();
        Cv2.CvtColor(swappedUInt8, lab, ColorConversionCodes.BGR2Lab);

        var labChannels = Cv2.Split(lab);
        labChannels[0].Dispose();
        using var aChannel = labChannels[1];
        using var bChannel = labChannels[2];

        var tempChannels = Cv2.Split(tempVisionFrame);
        using var tempBlueChannel = tempChannels[0];
        tempChannels[1].Dispose();
        tempChannels[2].Dispose();

        using var merged = new Mat();
        Cv2.Merge(new[] { tempBlueChannel, aChannel, bChannel }, merged);

        var result = new Mat();
        Cv2.CvtColor(merged, result, ColorConversionCodes.Lab2BGR);
        return result;
    }

    /// <summary>
    /// Python: <c>.astype(numpy.uint8)</c> (<paramref name="round"/> <see langword="false"/>) or
    /// <c>.round().astype(numpy.uint8)</c> (<paramref name="round"/> <see langword="true"/>) on
    /// a float32 <see cref="Mat"/>. Deliberately <b>not</b> a saturating cast: numpy's
    /// <c>astype</c> truncates toward zero and wraps modulo 256 for out-of-range values, unlike
    /// <see cref="Mat.ConvertTo"/>'s <c>uchar</c> path, which rounds-and-saturates. Values are
    /// expected in <c>[0, 255]</c> in practice (post Lab-&gt;BGR conversion of a valid image),
    /// where truncation/wraparound and saturation agree — this exists for the rare
    /// out-of-range case, so the behaviour still matches Python rather than silently clamping.
    /// Python's own <c>numpy.round</c> is round-half-to-even, matching
    /// <see cref="MidpointRounding.ToEven"/>. Caller owns the returned <see cref="Mat"/> (<c>CV_8UC3</c>).
    /// </summary>
    private static Mat AstypeUInt8(Mat float32Mat, bool round)
    {
        float32Mat.GetArray(out Vec3f[] pixels);
        var bytePixels = new Vec3b[pixels.Length];

        for (var i = 0; i < pixels.Length; i++)
        {
            bytePixels[i] = new Vec3b
            {
                Item0 = ToUInt8Wrapping(pixels[i].Item0, round),
                Item1 = ToUInt8Wrapping(pixels[i].Item1, round),
                Item2 = ToUInt8Wrapping(pixels[i].Item2, round),
            };
        }

        var result = new Mat(float32Mat.Rows, float32Mat.Cols, MatType.CV_8UC3);
        result.SetArray(bytePixels);
        return result;
    }

    private static byte ToUInt8Wrapping(float value, bool round)
    {
        var prepared = round ? MathF.Round(value, MidpointRounding.ToEven) : value;
        // Truncate toward zero (numpy astype's C-cast semantics), then wrap modulo 256 via an
        // unchecked narrowing cast, matching numpy's own unclamped float->uint8 conversion.
        unchecked
        {
            return (byte)(int)prepared;
        }
    }

    /// <summary>
    /// Builds one single-channel <c>CV_32FC1</c> <see cref="Mat"/> per channel directly from a
    /// flat CHW array — each channel is already a contiguous, row-major <c>height * width</c>
    /// plane in that layout, so this is a direct slice/copy, not a transpose. Caller owns every
    /// returned <see cref="Mat"/>.
    /// </summary>
    private static Mat[] SplitChwToChannelMats(float[] chwData, int channels, int height, int width)
    {
        var plane = height * width;
        var mats = new Mat[channels];

        for (var c = 0; c < channels; c++)
        {
            var channelData = new float[plane];
            Array.Copy(chwData, c * plane, channelData, 0, plane);

            var mat = new Mat(height, width, MatType.CV_32FC1);
            mat.SetArray(channelData);
            mats[c] = mat;
        }

        return mats;
    }

    // -----------------------------------------------------------------
    // blend_color_frame / colorize_frame / process_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>blend_color_frame</c>. Does not take ownership of either argument. Caller
    /// owns the returned <see cref="Mat"/>.
    ///
    /// <para>
    /// <b>Simplified from Python's double negation.</b> Python computes
    /// <c>frame_colorizer_blend = 1 - (frame_colorizer_blend / 100)</c> then calls
    /// <c>blend_frame(temp, color, 1 - frame_colorizer_blend)</c> — algebraically
    /// <c>1 - (1 - blend/100) == blend/100</c>, so this passes <c>frameColorizerBlend / 100.0</c>
    /// straight through to <see cref="Vision.BlendFrame"/> rather than reproducing the
    /// double-negation dance, which has no observable effect (PORT_CONVENTIONS.md rule 1 is
    /// about behavioural oddities; this one has none — it is pure algebraic simplification of
    /// dead arithmetic, verified in the ported unit tests against the same inputs/outputs
    /// Python's two-step version would produce).
    /// </para>
    /// </summary>
    public static Mat BlendColorFrame(Mat tempVisionFrame, Mat colorVisionFrame, int frameColorizerBlend)
        => Vision.Vision.BlendFrame(tempVisionFrame, colorVisionFrame, frameColorizerBlend / 100.0);

    /// <summary>
    /// Python: <c>colorize_frame</c>. Does not take ownership of <paramref name="tempVisionFrame"/>
    /// or <paramref name="frameColorizerSession"/>. Caller owns the returned <see cref="Mat"/>.
    /// </summary>
    public static Mat ColorizeFrame(
        Mat tempVisionFrame,
        FrameColorizerModelType modelType,
        Resolution modelSize,
        InferenceSession frameColorizerSession,
        int frameColorizerBlend)
    {
        var chwInput = PrepareTempFrame(tempVisionFrame, modelType, modelSize);
        var (outputData, outputChannels, outputHeight, outputWidth) = Forward(chwInput, modelSize.Height, modelSize.Width, frameColorizerSession);

        using var colorVisionFrame = MergeColorFrame(tempVisionFrame, outputData, outputChannels, outputHeight, outputWidth, modelType);
        return BlendColorFrame(tempVisionFrame, colorVisionFrame, frameColorizerBlend);
    }

    /// <summary>
    /// Python: <c>process_frame</c>. Does not take ownership of <paramref name="tempVisionFrame"/>
    /// or <paramref name="tempVisionMask"/>. The returned <see cref="ProcessorOutputs.Mask"/> is
    /// <paramref name="tempVisionMask"/> itself (Python returns it unmodified), not a copy — see
    /// <see cref="FaceEnhancer.ProcessFrame"/>'s remarks for the same shape of exception to the
    /// "caller owns both fields" convention.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(
        Mat tempVisionFrame,
        Mat tempVisionMask,
        FrameColorizerModelType modelType,
        Resolution modelSize,
        InferenceSession frameColorizerSession,
        int frameColorizerBlend)
    {
        var colorizedFrame = ColorizeFrame(tempVisionFrame, modelType, modelSize, frameColorizerSession, frameColorizerBlend);
        return new ProcessorOutputs(colorizedFrame, tempVisionMask);
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>facefusion.processors.modules.frame_colorizer.core</c> module's per-call
    /// inputs, extended per <see cref="IProcessorInputs"/>'s remarks — mirrors
    /// <c>FaceSwapper.FaceSwapperInputs</c>'s pattern. <c>frame_colorizer</c> has no source-face
    /// concept, so this is a flat wrapper over <see cref="ProcessFrame"/>'s own parameters.
    /// </summary>
    public sealed record FrameColorizerInputs(
        Mat TempVisionFrame,
        Mat TempVisionMask,
        FrameColorizerModelType ModelType,
        Resolution ModelSize,
        InferenceSession FrameColorizerSession,
        int FrameColorizerBlend) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/frame_colorizer/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract. Thin orchestration over
    /// <see cref="ProcessFrame"/> — mirrors <c>FaceSwapper.Processor</c>'s shape.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "frame_colorizer";

        /// <summary>Python: <c>get_common_modules()</c> (<c>[content_analyser]</c>).</summary>
        public IReadOnlyList<string> GetCommonModules() => new[] { "content_analyser" };

        /// <summary>
        /// Python: the <c>frame_colorizer</c>-specific half of <c>pre_check</c>. The
        /// common-module half is the caller's responsibility per <see cref="GetCommonModules"/>'s
        /// remarks; this overload needs the chosen <paramref name="model"/> since the
        /// parameterless <see cref="IProcessor.PreCheck"/> member has no <c>state_manager</c> to
        /// read it from.
        /// </summary>
        public bool PreCheck(FrameColorizerModel model) => FrameColorizer.PreCheck(model);

        /// <inheritdoc />
        bool IProcessor.PreCheck() => throw new InvalidOperationException(
            "frame_colorizer.PreCheck requires a FrameColorizerModel (no state_manager to read it from — call the FrameColorizerModel overload instead).");

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. Filesystem validation (<c>is_image</c>/<c>is_video</c>/
        /// <c>in_directory</c>/<c>same_file_extension</c>) is out of scope (same gap
        /// <c>FaceSwapper.Processor.PreProcess</c> documents); <c>frame_colorizer</c> has no
        /// source-path requirement of its own, so with that validation unavailable there is
        /// nothing left to check.
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
            if (inputs is not FrameColorizerInputs frameColorizerInputs)
            {
                throw new ArgumentException($"expected {nameof(FrameColorizerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return FrameColorizer.ProcessFrame(
                frameColorizerInputs.TempVisionFrame,
                frameColorizerInputs.TempVisionMask,
                frameColorizerInputs.ModelType,
                frameColorizerInputs.ModelSize,
                frameColorizerInputs.FrameColorizerSession,
                frameColorizerInputs.FrameColorizerBlend);
        }

        /// <summary>
        /// Python: <c>post_process()</c>. Cache clearing is out of scope without a real pool
        /// owner to clear (rule 5), same as <c>FaceSwapper.Processor.PostProcess</c>.
        /// </summary>
        public void PostProcess()
        {
        }
    }
}
