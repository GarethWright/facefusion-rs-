using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/modules/face_debugger/{core,types,choices}.py</c> — draws
/// the pipeline's internal state (bounding box, masks, landmarks) onto a frame for visual
/// debugging. No ONNX model of its own; the only inference it triggers is indirect, through the
/// occlusion/region face masks it can optionally draw.
///
/// <para>
/// <b>No global state (PORT_CONVENTIONS.md rule 5).</b> Every Python
/// <c>state_manager.get_item(...)</c> becomes an explicit parameter. <c>register_args</c>/
/// <c>apply_args</c>/<c>pre_check</c>/<c>pre_process</c>/<c>post_process</c>/
/// <c>get_inference_pool</c> are not ported: they depend on <c>job_store</c>/<c>download</c>/
/// <c>video_manager</c>, none of which exist in this project yet and none of which is part of
/// this module's assignment (processor algorithms, not CLI wiring). <see cref="ProcessFrame"/>
/// covers the actual <c>process_frame</c> contract this module exists to provide.
/// </para>
///
/// <para>
/// <b><c>FaceDebuggerItem</c> as a <see cref="FaceDebuggerItem"/> flags enum.</b> Python's
/// <c>FaceDebuggerItem = Literal['bounding-box', 'face-landmark-5', 'face-landmark-5/68',
/// 'face-landmark-68', 'face-landmark-68/5', 'face-mask']</c> plus a
/// <c>List[FaceDebuggerItem]</c> state value becomes a <see cref="FlagsAttribute"/> enum here so
/// <see cref="DebugFace"/> can test membership with a single bitwise check per item, matching
/// Python's <c>if 'x' in face_debugger_items</c> checks one-for-one.
/// </para>
///
/// <para>
/// <b>Cross-module collaborators taken as delegates, same pattern as <see cref="FaceSelector.SelectFaces"/>.</b>
/// <c>process_frame</c> calls <c>face_selector.select_faces</c> (this port's
/// <see cref="FaceSelector.SelectFaces"/>, called directly — same assignment) and
/// <c>face_creator.scale_face</c> (this port's <see cref="FaceCreator.ScaleFace"/>, also
/// available — <c>face_creator.py</c> has since landed in this project). <c>getStaticFaces</c>/
/// <c>refillFaces</c> are still threaded through as delegates because <see cref="FaceSelector.SelectFaces"/>
/// itself demands them in that shape (its own remarks explain why); <see cref="ProcessFrame"/>
/// simply forwards them.
/// </para>
///
/// <para>
/// <b><c>Face.BoundingBox</c> / <c>Face.LandmarkSet</c> runtime types.</b> Same convention as
/// <see cref="FaceSelector"/>/<see cref="FaceCreator"/>: <see cref="Types.Face.BoundingBox"/> is
/// assumed <c>float[4]</c> and every <see cref="Types.FaceLandmarkSet"/> entry is assumed
/// <c>float[,]</c> shaped <c>(N, 2)</c>; casts are documented at each site rather than silently
/// trusting <c>object</c>.
/// </para>
/// </summary>
public static class FaceDebugger
{
    // -----------------------------------------------------------------
    // debug_face dispatch
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>debug_face</c>. Mutates and returns <paramref name="tempVisionFrame"/> in
    /// place for the box/mask/landmark stages that reassign
    /// <c>temp_vision_frame = numpy.ascontiguousarray(temp_vision_frame)</c> (a no-op copy in
    /// Python since the array is already contiguous coming out of this pipeline) — this port
    /// draws directly onto <paramref name="tempVisionFrame"/> and returns the same
    /// <see cref="Mat"/> instance rather than cloning at each stage, since OpenCvSharp's
    /// <c>cv2.rectangle</c>/<c>circle</c>/<c>line</c>/<c>drawContours</c> equivalents all draw
    /// in place already. Caller retains ownership of <paramref name="tempVisionFrame"/>; this
    /// does not take ownership and does not dispose it.
    /// </summary>
    public static Mat DebugFace(
        Types.Face targetFace,
        Mat tempVisionFrame,
        FaceDebuggerItem faceDebuggerItems,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        Padding faceMaskPadding,
        IReadOnlyList<FaceMaskArea> faceMaskAreas,
        IReadOnlyList<FaceMaskRegion> faceMaskRegions,
        FaceOccluderModel faceOccluderModel,
        FaceParserModel faceParserModel,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool,
        IReadOnlyDictionary<string, InferenceSession>? parserInferencePool)
    {
        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.BoundingBox))
        {
            DrawBoundingBox(targetFace, tempVisionFrame);
        }

        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.FaceMask))
        {
            DrawFaceMask(
                targetFace, tempVisionFrame, faceMaskTypes, faceMaskPadding, faceMaskAreas, faceMaskRegions,
                faceOccluderModel, faceParserModel, occluderInferencePool, parserInferencePool);
        }

        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.FaceLandmark5))
        {
            DrawFaceLandmark5(targetFace, tempVisionFrame);
        }

        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.FaceLandmark5On68))
        {
            DrawFaceLandmark5On68(targetFace, tempVisionFrame);
        }

        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.FaceLandmark68))
        {
            DrawFaceLandmark68(targetFace, tempVisionFrame);
        }

        if (faceDebuggerItems.HasFlag(FaceDebuggerItem.FaceLandmark68On5))
        {
            DrawFaceLandmark68On5(targetFace, tempVisionFrame);
        }

        return tempVisionFrame;
    }

    // -----------------------------------------------------------------
    // Bounding box
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>draw_bounding_box</c>. Draws in place onto <paramref name="tempVisionFrame"/>;
    /// does not take ownership, does not dispose it. Assumes
    /// <see cref="Types.Face.BoundingBox"/> is <c>float[4]</c> (see class remarks).
    /// </summary>
    public static void DrawBoundingBox(Types.Face targetFace, Mat tempVisionFrame)
    {
        var boundingBox = (float[])targetFace.BoundingBox;

        // Python: `bounding_box.astype(numpy.int32)` truncates toward zero, same as a plain C#
        // `(int)` cast on a float — not `Math.Round`.
        var x1 = (int)boundingBox[0];
        var y1 = (int)boundingBox[1];
        var x2 = (int)boundingBox[2];
        var y2 = (int)boundingBox[3];
        var boxColor = new Scalar(0, 0, 255);
        var borderScale = CalculateScale(tempVisionFrame);
        var borderColor = new Scalar(100, 100, 255);

        Cv2.Rectangle(tempVisionFrame, new Point(x1, y1), new Point(x2, y2), boxColor, borderScale);

        if (targetFace.Angle == 0)
        {
            Cv2.Line(tempVisionFrame, new Point(x1, y1), new Point(x2, y1), borderColor, borderScale + 1);
        }

        if (targetFace.Angle == 180)
        {
            Cv2.Line(tempVisionFrame, new Point(x1, y2), new Point(x2, y2), borderColor, borderScale + 1);
        }

        if (targetFace.Angle == 90)
        {
            Cv2.Line(tempVisionFrame, new Point(x2, y1), new Point(x2, y2), borderColor, borderScale + 1);
        }

        if (targetFace.Angle == 270)
        {
            Cv2.Line(tempVisionFrame, new Point(x1, y1), new Point(x1, y2), borderColor, borderScale + 1);
        }
    }

    // -----------------------------------------------------------------
    // Face mask
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>draw_face_mask</c>. Draws in place onto <paramref name="tempVisionFrame"/>;
    /// does not take ownership, does not dispose it. <paramref name="occluderInferencePool"/>/
    /// <paramref name="parserInferencePool"/> are only actually consulted when
    /// <paramref name="faceMaskTypes"/> contains <see cref="FaceMaskType.Occlusion"/>/
    /// <see cref="FaceMaskType.Region"/> respectively — pass <see langword="null"/> when neither
    /// is in use.
    /// </summary>
    public static void DrawFaceMask(
        Types.Face targetFace,
        Mat tempVisionFrame,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        Padding faceMaskPadding,
        IReadOnlyList<FaceMaskArea> faceMaskAreas,
        IReadOnlyList<FaceMaskRegion> faceMaskRegions,
        FaceOccluderModel faceOccluderModel,
        FaceParserModel faceParserModel,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool,
        IReadOnlyDictionary<string, InferenceSession>? parserInferencePool)
    {
        var faceLandmark5 = (float[,])targetFace.LandmarkSet.Five;
        var faceLandmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;
        var faceLandmark5On68 = (float[,])targetFace.LandmarkSet.FiveOn68;

        var (cropVisionFrame, affineMatrix) = FaceHelper.WarpFaceByFaceLandmark5(tempVisionFrame, faceLandmark5On68, WarpTemplate.Arcface128, new Size(512, 512));
        using (cropVisionFrame)
        using (affineMatrix)
        {
            var inverseMatrix = new Mat();
            using (inverseMatrix)
            {
                Cv2.InvertAffineTransform(affineMatrix, inverseMatrix);
                var tempSize = new Size(tempVisionFrame.Cols, tempVisionFrame.Rows);
                var maskScale = CalculateScale(tempVisionFrame);
                var maskColor = new Scalar(0, 255, 0);

                if (ArraysEqual(faceLandmark5, faceLandmark5On68))
                {
                    maskColor = new Scalar(255, 255, 0);
                }

                if (targetFace.Origin == "refill")
                {
                    maskColor = new Scalar(0, 165, 255);
                }

                var cropMasks = new List<Mat>();

                try
                {
                    if (faceMaskTypes.Contains(FaceMaskType.Box))
                    {
                        cropMasks.Add(FaceMasker.CreateBoxMask(cropVisionFrame, 0, faceMaskPadding));
                    }

                    if (faceMaskTypes.Contains(FaceMaskType.Occlusion))
                    {
                        cropMasks.Add(FaceMasker.CreateOcclusionMask(cropVisionFrame, faceOccluderModel, occluderInferencePool!));
                    }

                    if (faceMaskTypes.Contains(FaceMaskType.Area))
                    {
                        // Python: `face_landmark_68 = cv2.transform(face_landmark_68.reshape(1,
                        // -1, 2), affine_matrix).reshape(-1, 2)` — transforms the *un-scaled*
                        // temp-frame landmark_68 into crop space using the same affine matrix
                        // the warp above used.
                        var transformedLandmark68 = FaceHelper.TransformPoints(faceLandmark68, affineMatrix);
                        cropMasks.Add(FaceMasker.CreateAreaMask(cropVisionFrame, transformedLandmark68, faceMaskAreas));
                    }

                    if (faceMaskTypes.Contains(FaceMaskType.Region))
                    {
                        cropMasks.Add(FaceMasker.CreateRegionMask(cropVisionFrame, faceMaskRegions, faceParserModel, parserInferencePool!));
                    }

                    if (cropMasks.Count == 0)
                    {
                        return;
                    }

                    using var reducedMask = cropMasks[0].Clone();
                    for (var index = 1; index < cropMasks.Count; index++)
                    {
                        Cv2.Min(reducedMask, cropMasks[index], reducedMask);
                    }

                    // Python: `crop_mask.clip(0, 1)` then `(crop_mask * 255).astype(uint8)`.
                    using var clippedMask = new Mat();
                    Cv2.Max(reducedMask, 0.0, clippedMask);
                    Cv2.Min(clippedMask, 1.0, clippedMask);

                    using var byteMask = new Mat();
                    clippedMask.ConvertTo(byteMask, MatType.CV_8UC1, 255.0);

                    using var inverseVisionFrame = new Mat();
                    Cv2.WarpAffine(byteMask, inverseVisionFrame, inverseMatrix, tempSize);

                    using var thresholded = new Mat();
                    Cv2.Threshold(inverseVisionFrame, thresholded, 100, 255, ThresholdTypes.Binary);

                    Cv2.FindContours(thresholded, out var contours, out _, RetrievalModes.List, ContourApproximationModes.ApproxNone);
                    Cv2.DrawContours(tempVisionFrame, contours, -1, maskColor, maskScale);
                }
                finally
                {
                    foreach (var mask in cropMasks)
                    {
                        mask.Dispose();
                    }
                }
            }
        }
    }

    // -----------------------------------------------------------------
    // Landmarks
    // -----------------------------------------------------------------

    /// <summary>Python: <c>draw_face_landmark_5</c>. Draws in place; does not dispose <paramref name="tempVisionFrame"/>.</summary>
    public static void DrawFaceLandmark5(Types.Face targetFace, Mat tempVisionFrame)
    {
        var faceLandmark5 = (float[,])targetFace.LandmarkSet.Five;
        var pointScale = CalculateScale(tempVisionFrame);
        var pointColor = new Scalar(0, 0, 255);

        if (targetFace.Origin == "refill")
        {
            pointColor = new Scalar(0, 165, 255);
        }

        DrawPoints(tempVisionFrame, faceLandmark5, pointScale, pointColor);
    }

    /// <summary>Python: <c>draw_face_landmark_5_68</c>. Draws in place; does not dispose <paramref name="tempVisionFrame"/>.</summary>
    public static void DrawFaceLandmark5On68(Types.Face targetFace, Mat tempVisionFrame)
    {
        var faceLandmark5 = (float[,])targetFace.LandmarkSet.Five;
        var faceLandmark5On68 = (float[,])targetFace.LandmarkSet.FiveOn68;
        var pointScale = CalculateScale(tempVisionFrame);
        var pointColor = new Scalar(0, 255, 0);

        if (ArraysEqual(faceLandmark5, faceLandmark5On68))
        {
            pointColor = new Scalar(255, 255, 0);
        }

        if (targetFace.Origin == "refill")
        {
            pointColor = new Scalar(0, 165, 255);
        }

        DrawPoints(tempVisionFrame, faceLandmark5On68, pointScale, pointColor);
    }

    /// <summary>Python: <c>draw_face_landmark_68</c>. Draws in place; does not dispose <paramref name="tempVisionFrame"/>.</summary>
    public static void DrawFaceLandmark68(Types.Face targetFace, Mat tempVisionFrame)
    {
        var faceLandmark68 = (float[,])targetFace.LandmarkSet.SixtyEight;
        var faceLandmark68On5 = (float[,])targetFace.LandmarkSet.SixtyEightOn5;
        var pointScale = CalculateScale(tempVisionFrame);
        var pointColor = new Scalar(0, 255, 0);

        if (ArraysEqual(faceLandmark68, faceLandmark68On5))
        {
            pointColor = new Scalar(255, 255, 0);
        }

        if (targetFace.Origin == "refill")
        {
            pointColor = new Scalar(0, 165, 255);
        }

        DrawPoints(tempVisionFrame, faceLandmark68, pointScale, pointColor);
    }

    /// <summary>Python: <c>draw_face_landmark_68_5</c>. Draws in place; does not dispose <paramref name="tempVisionFrame"/>.</summary>
    public static void DrawFaceLandmark68On5(Types.Face targetFace, Mat tempVisionFrame)
    {
        var faceLandmark68On5 = (float[,])targetFace.LandmarkSet.SixtyEightOn5;
        var pointScale = CalculateScale(tempVisionFrame);
        var pointColor = new Scalar(255, 255, 0);

        if (targetFace.Origin == "refill")
        {
            pointColor = new Scalar(0, 165, 255);
        }

        DrawPoints(tempVisionFrame, faceLandmark68On5, pointScale, pointColor);
    }

    /// <summary>
    /// Shared tail of the four landmark-drawing functions above: Python's
    /// <c>if numpy.any(face_landmark): ... for point in face_landmark.astype(int32): cv2.circle(...)</c>.
    /// </summary>
    private static void DrawPoints(Mat tempVisionFrame, float[,] points, int pointScale, Scalar pointColor)
    {
        if (!AnyNonzero(points))
        {
            return;
        }

        var rows = points.GetLength(0);
        for (var i = 0; i < rows; i++)
        {
            // Python: `.astype(numpy.int32)` truncates toward zero.
            var x = (int)points[i, 0];
            var y = (int)points[i, 1];
            Cv2.Circle(tempVisionFrame, new Point(x, y), pointScale, pointColor, thickness: -1);
        }
    }

    // -----------------------------------------------------------------
    // calculate_scale
    // -----------------------------------------------------------------

    /// <summary>Python: <c>calculate_scale</c>.</summary>
    public static int CalculateScale(Mat tempVisionFrame)
    {
        var frameHeight = tempVisionFrame.Rows;
        // Python: `round(frame_height / 270)` — banker's rounding (round-half-to-even), same as
        // MidpointRounding.ToEven, which is C#'s Math.Round default.
        var frameScale = (int)Math.Round(frameHeight / 270.0, MidpointRounding.ToEven);
        return Math.Max(1, Math.Min(10, frameScale));
    }

    // -----------------------------------------------------------------
    // process_frame
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: <c>process_frame</c>. Returns <c>(temp_vision_frame, temp_vision_mask)</c> as a
    /// <see cref="ProcessorOutputs"/>; <paramref name="tempVisionMask"/> is passed
    /// through unchanged, same as Python (face_debugger never touches the mask). Draws directly
    /// onto and returns <paramref name="tempVisionFrame"/> — see <see cref="DebugFace"/>'s
    /// remarks on why no extra clone is made. See the class remarks for why
    /// <paramref name="getStaticFaces"/>/<paramref name="refillFaces"/> are delegates.
    /// </summary>
    public static ProcessorOutputs ProcessFrame(
        Mat referenceVisionFrame,
        IReadOnlyList<Mat> sourceVisionFrames,
        IReadOnlyList<Mat> targetVisionFrames,
        Mat tempVisionFrame,
        Mat tempVisionMask,
        FaceDebuggerItem faceDebuggerItems,
        FaceSelectorMode faceSelectorMode,
        double faceTrackerScore,
        FaceSelectorOrder faceSelectorOrder,
        FaceSelectorGender? faceSelectorGender,
        FaceSelectorRace? faceSelectorRace,
        int? faceSelectorAgeStart,
        int? faceSelectorAgeEnd,
        int referenceFacePosition,
        double referenceFaceDistance,
        IReadOnlyList<FaceMaskType> faceMaskTypes,
        Padding faceMaskPadding,
        IReadOnlyList<FaceMaskArea> faceMaskAreas,
        IReadOnlyList<FaceMaskRegion> faceMaskRegions,
        FaceOccluderModel faceOccluderModel,
        FaceParserModel faceParserModel,
        IReadOnlyDictionary<string, InferenceSession>? occluderInferencePool,
        IReadOnlyDictionary<string, InferenceSession>? parserInferencePool,
        Func<IReadOnlyList<Mat>, IReadOnlyList<Types.Face>> getStaticFaces,
        Func<IReadOnlyList<Types.Face?>, IReadOnlyList<Types.Face>> refillFaces)
    {
        var targetVisionFrame = CommonHelper.GetMiddle(targetVisionFrames);

        var targetFaces = FaceSelector.SelectFaces(
            referenceVisionFrame, sourceVisionFrames, targetVisionFrames,
            faceSelectorMode, faceTrackerScore, faceSelectorOrder, faceSelectorGender, faceSelectorRace,
            faceSelectorAgeStart, faceSelectorAgeEnd, referenceFacePosition, referenceFaceDistance,
            getStaticFaces, refillFaces);

        if (targetFaces.Count > 0 && targetVisionFrame is not null)
        {
            foreach (var rawTargetFace in targetFaces)
            {
                var targetFace = FaceCreator.ScaleFace(rawTargetFace, targetVisionFrame, tempVisionFrame);
                DebugFace(
                    targetFace, tempVisionFrame, faceDebuggerItems, faceMaskTypes, faceMaskPadding,
                    faceMaskAreas, faceMaskRegions, faceOccluderModel, faceParserModel,
                    occluderInferencePool, parserInferencePool);
            }
        }

        return new ProcessorOutputs(tempVisionFrame, tempVisionMask);
    }

    // -----------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------

    private static bool ArraysEqual(float[,] a, float[,] b)
    {
        // Python: `numpy.array_equal(a, b)` — shape then elementwise equality, exact (no
        // tolerance).
        if (a.GetLength(0) != b.GetLength(0) || a.GetLength(1) != b.GetLength(1))
        {
            return false;
        }

        for (var i = 0; i < a.GetLength(0); i++)
        {
            for (var j = 0; j < a.GetLength(1); j++)
            {
                if (a[i, j] != b[i, j])
                {
                    return false;
                }
            }
        }

        return true;
    }

    private static bool AnyNonzero(float[,] points)
    {
        for (var i = 0; i < points.GetLength(0); i++)
        {
            for (var j = 0; j < points.GetLength(1); j++)
            {
                if (points[i, j] != 0f)
                {
                    return true;
                }
            }
        }

        return false;
    }

    // -----------------------------------------------------------------
    // Processor adapter (IProcessor)
    // -----------------------------------------------------------------

    /// <summary>
    /// Python: the <c>facefusion.processors.modules.face_debugger.core</c> module's per-call
    /// inputs, extended per <see cref="IProcessorInputs"/>'s remarks — see
    /// <c>FaceSwapper.FaceSwapperInputs</c> for the pattern this mirrors. <see cref="OccluderInferencePool"/>/
    /// <see cref="ParserInferencePool"/> may be <see langword="null"/> when <see cref="FaceMaskTypes"/>
    /// does not include <see cref="FaceMaskType.Occlusion"/>/<see cref="FaceMaskType.Region"/>
    /// (see <see cref="DrawFaceMask"/>'s remarks).
    /// </summary>
    public sealed record FaceDebuggerInputs(
        Mat ReferenceVisionFrame,
        IReadOnlyList<Mat> SourceVisionFrames,
        IReadOnlyList<Mat> TargetVisionFrames,
        Mat TempVisionFrame,
        Mat TempVisionMask,
        FaceDebuggerItem FaceDebuggerItems,
        FaceSelectorMode FaceSelectorMode,
        double FaceTrackerScore,
        FaceSelectorOrder FaceSelectorOrder,
        FaceSelectorGender? FaceSelectorGender,
        FaceSelectorRace? FaceSelectorRace,
        int? FaceSelectorAgeStart,
        int? FaceSelectorAgeEnd,
        int ReferenceFacePosition,
        double ReferenceFaceDistance,
        IReadOnlyList<FaceMaskType> FaceMaskTypes,
        Padding FaceMaskPadding,
        IReadOnlyList<FaceMaskArea> FaceMaskAreas,
        IReadOnlyList<FaceMaskRegion> FaceMaskRegions,
        FaceOccluderModel FaceOccluderModel,
        FaceParserModel FaceParserModel,
        IReadOnlyDictionary<string, InferenceSession>? OccluderInferencePool,
        IReadOnlyDictionary<string, InferenceSession>? ParserInferencePool,
        Func<IReadOnlyList<Mat>, IReadOnlyList<FaceFusion.Types.Face>> GetStaticFaces,
        Func<IReadOnlyList<FaceFusion.Types.Face?>, IReadOnlyList<FaceFusion.Types.Face>> RefillFaces) : IProcessorInputs;

    /// <summary>
    /// Python: <c>facefusion/processors/modules/face_debugger/core.py</c>'s module-level
    /// functions, adapted to the <see cref="IProcessor"/> contract. Thin orchestration over
    /// <see cref="ProcessFrame"/> — mirrors <c>FaceSwapper.Processor</c>'s shape.
    /// </summary>
    public sealed class Processor : IProcessor
    {
        /// <inheritdoc />
        public string Name => "face_debugger";

        /// <summary>
        /// Python: <c>get_common_modules()</c> — see <see cref="IProcessor.GetCommonModules"/>'s
        /// remarks for why these are names rather than callable references.
        /// </summary>
        public IReadOnlyList<string> GetCommonModules() =>
            new[] { "content_analyser", "face_classifier", "face_detector", "face_landmarker", "face_masker", "face_recognizer" };

        /// <summary>
        /// Python: <c>pre_check()</c>. <c>face_debugger</c> has no ONNX model of its own (see the
        /// class remarks) — Python's version is just the common-module loop, which is the
        /// caller's responsibility per <see cref="GetCommonModules"/>'s remarks. Nothing left
        /// for this module's own half to check, so this always returns <see langword="true"/>.
        /// </summary>
        public bool PreCheck() => true;

        /// <summary>
        /// Python: <c>pre_process(mode)</c>. <c>is_image</c>/<c>is_video</c>/<c>in_directory</c>/
        /// <c>same_file_extension</c> (facefusion/filesystem.py) are not ported (out of this
        /// assignment's scope, same gap <c>FaceSwapper.Processor.PreProcess</c> documents) —
        /// <c>face_debugger</c> has no source-path requirement of its own (unlike
        /// <c>face_swapper</c>), so with that filesystem validation unavailable there is nothing
        /// left to check; this returns <see langword="true"/> unconditionally.
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
            if (inputs is not FaceDebuggerInputs faceDebuggerInputs)
            {
                throw new ArgumentException($"expected {nameof(FaceDebuggerInputs)}, got {inputs.GetType().Name}.", nameof(inputs));
            }

            return FaceDebugger.ProcessFrame(
                faceDebuggerInputs.ReferenceVisionFrame,
                faceDebuggerInputs.SourceVisionFrames,
                faceDebuggerInputs.TargetVisionFrames,
                faceDebuggerInputs.TempVisionFrame,
                faceDebuggerInputs.TempVisionMask,
                faceDebuggerInputs.FaceDebuggerItems,
                faceDebuggerInputs.FaceSelectorMode,
                faceDebuggerInputs.FaceTrackerScore,
                faceDebuggerInputs.FaceSelectorOrder,
                faceDebuggerInputs.FaceSelectorGender,
                faceDebuggerInputs.FaceSelectorRace,
                faceDebuggerInputs.FaceSelectorAgeStart,
                faceDebuggerInputs.FaceSelectorAgeEnd,
                faceDebuggerInputs.ReferenceFacePosition,
                faceDebuggerInputs.ReferenceFaceDistance,
                faceDebuggerInputs.FaceMaskTypes,
                faceDebuggerInputs.FaceMaskPadding,
                faceDebuggerInputs.FaceMaskAreas,
                faceDebuggerInputs.FaceMaskRegions,
                faceDebuggerInputs.FaceOccluderModel,
                faceDebuggerInputs.FaceParserModel,
                faceDebuggerInputs.OccluderInferencePool,
                faceDebuggerInputs.ParserInferencePool,
                faceDebuggerInputs.GetStaticFaces,
                faceDebuggerInputs.RefillFaces);
        }

        /// <summary>
        /// Python: <c>post_process()</c>. Cache clearing is out of scope without a real pool
        /// owner to clear (rule 5), same as <c>FaceSwapper.Processor.PostProcess</c> — a caller
        /// that owns those caches clears them itself.
        /// </summary>
        public void PostProcess()
        {
        }
    }
}

/// <summary>
/// Ported from Python facefusion/processors/modules/face_debugger/types.py:
/// <c>FaceDebuggerItem = Literal['bounding-box', 'face-landmark-5', 'face-landmark-5/68',
/// 'face-landmark-68', 'face-landmark-68/5', 'face-mask']</c>, plus the state's
/// <c>List[FaceDebuggerItem]</c> — modelled as <see cref="FlagsAttribute"/> so multiple items
/// can be combined in one value the way <c>debug_face</c>'s membership checks imply. Python's
/// own <c>face_debugger_items</c> list order (see <c>choices.py</c>/the CLI default
/// <c>'face-landmark-5/68 face-mask'</c>) is not meaningful to the drawing order — every
/// <c>if 'x' in items:</c> check in <c>debug_face</c> is independent and always runs in the
/// same bounding-box/mask/5/5-68/68/68-5 order regardless of list order, which this flags enum
/// preserves via <see cref="FaceDebugger.DebugFace"/>'s fixed check sequence.
/// </summary>
[Flags]
public enum FaceDebuggerItem
{
    /// <summary>No items selected. Python has no literal for this — the Python type is a
    /// list of item strings, so "empty" is the empty list rather than a named value.</summary>
    [NonWireValue]
    None = 0,

    [WireName("bounding-box")]
    BoundingBox = 1 << 0,

    [WireName("face-landmark-5")]
    FaceLandmark5 = 1 << 1,

    [WireName("face-landmark-5/68")]
    FaceLandmark5On68 = 1 << 2,

    [WireName("face-landmark-68")]
    FaceLandmark68 = 1 << 3,

    [WireName("face-landmark-68/5")]
    FaceLandmark68On5 = 1 << 4,

    [WireName("face-mask")]
    FaceMask = 1 << 5,
}
