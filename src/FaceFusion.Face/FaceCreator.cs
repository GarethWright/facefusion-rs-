using System.Linq;
using FaceFusion.Types;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using VisionHelper = FaceFusion.Vision.Vision;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_creator.py</c> — builds <see cref="Types.Face"/> records from the
/// detector/landmarker/recognizer/classifier stages' raw outputs, and the small set of
/// higher-level helpers (get-one/get-many/get-static, refill, average, scale) built on top of
/// it.
///
/// <para>
/// <b>Static class, no download-backed session pools (documented divergence, consistent with
/// <see cref="FaceDetector"/>/<see cref="FaceLandmarker"/>/<see cref="FaceRecognizer"/>/
/// <see cref="FaceClassifier"/>).</b> Python's <c>get_many_faces</c> calls
/// <c>face_detector.detect_faces</c>/<c>detect_faces_by_angle</c>, which internally resolve
/// their own <c>InferencePool</c> via <c>state_manager</c>-keyed globals; <c>create_faces</c>
/// likewise calls <c>face_landmarker.detect_face_landmark</c>,
/// <c>face_recognizer.calculate_face_embedding</c> and <c>face_classifier.classify_face</c>,
/// each resolving their own pool the same way. Per PORT_CONVENTIONS.md rule 5 ("no global
/// state — take it as a parameter") and matching how every other stage in this project was
/// ported, every state_manager-sourced value (thresholds, angles, model selection) and every
/// inference session those stages need is an explicit parameter here instead. This class stays
/// a <c>public static class</c> (Python: a module of free functions) — it holds no state of
/// its own.
/// </para>
/// </summary>
public static class FaceCreator
{
    /// <summary>
    /// Python: <c>create_faces</c>. Builds one <see cref="Types.Face"/> per detection that
    /// survives NMS, running the landmarker/recognizer/classifier stages on each. Does not take
    /// ownership of <paramref name="visionFrame"/>.
    /// </summary>
    /// <param name="faceLandmarkerScoreThreshold">
    /// Python: <c>state_manager.get_item('face_landmarker_score')</c>. When &gt; 0, the 68-point
    /// landmarker actually runs (see Python's <c>if state_manager.get_item('face_landmarker_score')
    /// &gt; 0:</c>); <paramref name="faceLandmarkerModel"/>/<paramref name="twoDFan4Session"/>/
    /// <paramref name="peppaWutzSession"/> are only consulted in that case.
    /// </param>
    public static IReadOnlyList<Types.Face> CreateFaces(
        Mat visionFrame,
        IReadOnlyList<float[]> boundingBoxes,
        IReadOnlyList<double> faceScores,
        IReadOnlyList<float[,]> faceLandmarks5,
        FaceDetectorModel faceDetectorModel,
        IReadOnlyList<int> faceDetectorAngles,
        double faceDetectorScoreThreshold,
        double faceLandmarkerScoreThreshold,
        FaceLandmarkerModel faceLandmarkerModel,
        InferenceSession fan685Session,
        InferenceSession? twoDFan4Session,
        InferenceSession? peppaWutzSession,
        InferenceSession faceRecognizerSession,
        InferenceSession faceClassifierSession)
    {
        var faces = new List<Types.Face>();
        var nmsThreshold = FaceHelper.GetNmsThreshold(faceDetectorModel, faceDetectorAngles);
        // FaceHelper.ApplyNms works in float32 (matching the detector's own precision); the
        // detector's scores travel as `double` through this class's signatures (matching
        // FaceFusion.Types' Score = double alias), so they are narrowed only here, at the one
        // call site that actually needs float32.
        var faceScoresFloat = faceScores.Select(score => (float)score).ToList();
        var keepIndices = FaceHelper.ApplyNms(boundingBoxes, faceScoresFloat, (float)faceDetectorScoreThreshold, nmsThreshold);

        foreach (var index in keepIndices)
        {
            var boundingBox = boundingBoxes[index];
            var faceScore = faceScores[index];
            var faceLandmark5 = faceLandmarks5[index];

            // Python:
            //   face_landmark_5_68 = face_landmark_5
            //   face_landmark_68_5 = estimate_face_landmark_68_5(face_landmark_5_68)
            //   face_landmark_68 = face_landmark_68_5
            //   face_landmark_score_68 = 0.0
            //   face_angle = estimate_face_angle(face_landmark_68_5)
            var faceLandmark568 = faceLandmark5;
            var faceLandmark685 = FaceLandmarker.EstimateFaceLandmark685(fan685Session, faceLandmark568);
            var faceLandmark68 = faceLandmark685;
            var faceLandmarkScore68 = 0.0;
            var faceAngle = FaceHelper.EstimateFaceAngle(faceLandmark685);

            if (faceLandmarkerScoreThreshold > 0)
            {
                var (detectedLandmark68, detectedScore) = FaceLandmarker.DetectFaceLandmark(
                    faceLandmarkerModel, twoDFan4Session, peppaWutzSession, visionFrame, boundingBox, faceAngle);

                // Python: `face_landmark_68, face_landmark_score_68 = detect_face_landmark(...)`
                // unconditionally overwrites face_landmark_68 with whatever came back, including
                // None in the documented FaceLandmarker.DetectFaceLandmark edge case (only
                // reachable when a single non-'many' landmarker model is requested and the
                // *other*, never-run model's default 0.0 score wins the internal comparison —
                // not exercised by any caller in this port's test suite, which always requests
                // 'many'). FaceLandmarkSet's fields are non-nullable `object` (FaceFusion.Types,
                // outside this assignment's scope to change), so that one combination cannot be
                // reproduced bit-for-bit here: a null result is deliberately not stored, keeping
                // the previous (5-point-derived) landmark68 value instead of Python's None.
                if (detectedLandmark68 is not null)
                {
                    faceLandmark68 = detectedLandmark68;
                }

                faceLandmarkScore68 = detectedScore;
            }

            if (faceLandmarkScore68 > faceLandmarkerScoreThreshold)
            {
                faceLandmark568 = FaceHelper.ConvertToFaceLandmark5(faceLandmark68);
            }

            var faceLandmarkSet = new FaceLandmarkSet(
                Five: faceLandmark5,
                FiveOn68: faceLandmark568,
                SixtyEight: faceLandmark68,
                SixtyEightOn5: faceLandmark685);

            var faceScoreSet = new FaceScoreSet(faceScore, faceLandmarkScore68);

            var landmarkForEmbedding = (float[,])faceLandmarkSet.FiveOn68;
            var (faceEmbedding, faceEmbeddingNorm) = FaceRecognizer.CalculateFaceEmbedding(faceRecognizerSession, visionFrame, landmarkForEmbedding);
            var (gender, age, race) = FaceClassifier.ClassifyFace(faceClassifierSession, visionFrame, landmarkForEmbedding);

            faces.Add(new Types.Face(
                Origin: "detect",
                BoundingBox: boundingBox,
                ScoreSet: faceScoreSet,
                LandmarkSet: faceLandmarkSet,
                Angle: faceAngle,
                Embedding: faceEmbedding,
                EmbeddingNorm: faceEmbeddingNorm,
                Age: age,
                Gender: gender,
                Race: race));
        }

        return faces;
    }

    /// <summary>Python: <c>get_one_face</c>.</summary>
    public static Types.Face? GetOneFace(IReadOnlyList<Types.Face> faces, int position = 0)
    {
        if (faces.Count == 0)
        {
            return null;
        }

        position = Math.Min(position, faces.Count - 1);
        return faces[position];
    }

    /// <summary>
    /// Python: <c>get_many_faces</c>. Does not take ownership of any <see cref="Mat"/> in
    /// <paramref name="visionFrames"/>.
    /// </summary>
    public static IReadOnlyList<Types.Face> GetManyFaces(
        IReadOnlyList<Mat> visionFrames,
        FaceDetectorModel faceDetectorModel,
        string faceDetectorSize,
        double faceDetectorScoreThreshold,
        IReadOnlyList<int> faceDetectorMargin,
        IReadOnlyList<int> faceDetectorAngles,
        double faceLandmarkerScoreThreshold,
        FaceLandmarkerModel faceLandmarkerModel,
        IReadOnlyDictionary<string, InferenceSession> faceDetectorSessions,
        InferenceSession fan685Session,
        InferenceSession? twoDFan4Session,
        InferenceSession? peppaWutzSession,
        InferenceSession faceRecognizerSession,
        InferenceSession faceClassifierSession)
    {
        var manyFaces = new List<Types.Face>();

        foreach (var visionFrame in visionFrames)
        {
            if (!VisionHelper.IsVisionFrame(visionFrame))
            {
                continue;
            }

            var allBoundingBoxes = new List<float[]>();
            var allFaceScores = new List<double>();
            var allFaceLandmarks5 = new List<float[,]>();

            foreach (var faceDetectorAngle in faceDetectorAngles)
            {
                IReadOnlyList<float[]> boundingBoxes;
                IReadOnlyList<double> faceScores;
                IReadOnlyList<float[,]> faceLandmarks5;

                if (faceDetectorAngle == 0)
                {
                    (boundingBoxes, faceScores, faceLandmarks5) = FaceDetector.DetectFaces(
                        visionFrame, faceDetectorModel, faceDetectorSize, faceDetectorScoreThreshold, faceDetectorMargin, faceDetectorSessions);
                }
                else
                {
                    (boundingBoxes, faceScores, faceLandmarks5) = FaceDetector.DetectFacesByAngle(
                        visionFrame, faceDetectorAngle, faceDetectorModel, faceDetectorSize, faceDetectorScoreThreshold, faceDetectorMargin, faceDetectorSessions);
                }

                allBoundingBoxes.AddRange(boundingBoxes);
                allFaceScores.AddRange(faceScores);
                allFaceLandmarks5.AddRange(faceLandmarks5);
            }

            if (allBoundingBoxes.Count > 0 && allFaceScores.Count > 0 && allFaceLandmarks5.Count > 0 && faceDetectorScoreThreshold > 0)
            {
                var faces = CreateFaces(
                    visionFrame,
                    allBoundingBoxes,
                    allFaceScores,
                    allFaceLandmarks5,
                    faceDetectorModel,
                    faceDetectorAngles,
                    faceDetectorScoreThreshold,
                    faceLandmarkerScoreThreshold,
                    faceLandmarkerModel,
                    fan685Session,
                    twoDFan4Session,
                    peppaWutzSession,
                    faceRecognizerSession,
                    faceClassifierSession);

                if (faces.Count > 0)
                {
                    manyFaces.AddRange(faces);
                }
            }
        }

        return manyFaces;
    }

    /// <summary>
    /// Python: <c>get_static_faces</c>. Does not take ownership of any <see cref="Mat"/> in
    /// <paramref name="visionFrames"/>. <paramref name="faceStore"/> stands in for Python's
    /// module-global <c>face_store</c> functions — see <see cref="FaceStore"/>'s own remarks on
    /// why it is an instance rather than a module global; pass a shared instance to get
    /// Python's process-wide caching behaviour.
    /// </summary>
    public static IReadOnlyList<Types.Face> GetStaticFaces(
        IReadOnlyList<Mat> visionFrames,
        FaceStore faceStore,
        FaceDetectorModel faceDetectorModel,
        string faceDetectorSize,
        double faceDetectorScoreThreshold,
        IReadOnlyList<int> faceDetectorMargin,
        IReadOnlyList<int> faceDetectorAngles,
        double faceLandmarkerScoreThreshold,
        FaceLandmarkerModel faceLandmarkerModel,
        IReadOnlyDictionary<string, InferenceSession> faceDetectorSessions,
        InferenceSession fan685Session,
        InferenceSession? twoDFan4Session,
        InferenceSession? peppaWutzSession,
        InferenceSession faceRecognizerSession,
        InferenceSession faceClassifierSession)
    {
        var manyFaces = new List<Types.Face>();

        foreach (var visionFrame in visionFrames)
        {
            var faces = faceStore.GetFaces(visionFrame);

            if (faces is null)
            {
                lock (faceStore.ResolveLock(visionFrame))
                {
                    faces = faceStore.GetFaces(visionFrame);

                    if (faces is null)
                    {
                        var detected = GetManyFaces(
                            new[] { visionFrame },
                            faceDetectorModel,
                            faceDetectorSize,
                            faceDetectorScoreThreshold,
                            faceDetectorMargin,
                            faceDetectorAngles,
                            faceLandmarkerScoreThreshold,
                            faceLandmarkerModel,
                            faceDetectorSessions,
                            fan685Session,
                            twoDFan4Session,
                            peppaWutzSession,
                            faceRecognizerSession,
                            faceClassifierSession);

                        if (detected.Count > 0)
                        {
                            faceStore.SetFaces(visionFrame, detected);
                        }

                        faces = detected;
                    }
                }
            }

            manyFaces.AddRange(faces);
        }

        return manyFaces;
    }

    /// <summary>Python: <c>refill_faces</c>.</summary>
    public static IReadOnlyList<Types.Face> RefillFaces(IReadOnlyList<Types.Face?> faces)
    {
        var fillFaces = new List<Types.Face>();
        var anchorIndexPrevious = -1;

        for (var index = 0; index < faces.Count; index++)
        {
            var face = faces[index];

            if (face is not null)
            {
                for (var gapIndex = anchorIndexPrevious + 1; gapIndex < index; gapIndex++)
                {
                    var averageFactor = (double)(gapIndex - anchorIndexPrevious) / (index - anchorIndexPrevious);
                    var averageFace = AverageFaceGeometry(new[] { faces[anchorIndexPrevious]!, face }, averageFactor);
                    fillFaces.Add(averageFace);
                }

                fillFaces.Add(face);
                anchorIndexPrevious = index;
            }
        }

        return fillFaces;
    }

    /// <summary>
    /// Python: <c>average_face_geometry</c>. <paramref name="faces"/> must have length 2 (as
    /// every Python call site passes, via <c>get_first</c>/<c>get_middle</c> over a 2-element
    /// list — the "middle" of a 2-element list is Python's <c>len(faces) // 2</c> = index 1,
    /// i.e. the second/last element).
    /// </summary>
    public static Types.Face AverageFaceGeometry(IReadOnlyList<Types.Face> faces, double averageFactor)
    {
        var faceFirst = faces[0];
        var faceMiddle = faces[faces.Count / 2];
        var faceAnchor = averageFactor < 0.5 ? faceFirst : faceMiddle;

        var landmarkSetFirst = faceFirst.LandmarkSet;
        var landmarkSetMiddle = faceMiddle.LandmarkSet;

        var landmarkSet = new FaceLandmarkSet(
            Five: FaceHelper.AveragePoints((float[,])landmarkSetFirst.Five, (float[,])landmarkSetMiddle.Five, averageFactor),
            FiveOn68: FaceHelper.AveragePoints((float[,])landmarkSetFirst.FiveOn68, (float[,])landmarkSetMiddle.FiveOn68, averageFactor),
            SixtyEight: FaceHelper.AveragePoints((float[,])landmarkSetFirst.SixtyEight, (float[,])landmarkSetMiddle.SixtyEight, averageFactor),
            SixtyEightOn5: FaceHelper.AveragePoints((float[,])landmarkSetFirst.SixtyEightOn5, (float[,])landmarkSetMiddle.SixtyEightOn5, averageFactor));

        var boundingBox = AveragePointsFlat(faceFirst.BoundingBox, faceMiddle.BoundingBox, averageFactor);

        return new Types.Face(
            Origin: "refill",
            BoundingBox: boundingBox,
            ScoreSet: faceAnchor.ScoreSet,
            LandmarkSet: landmarkSet,
            Angle: FaceHelper.EstimateFaceAngle((float[,])landmarkSet.SixtyEightOn5),
            Embedding: faceAnchor.Embedding,
            EmbeddingNorm: faceAnchor.EmbeddingNorm,
            Age: faceAnchor.Age,
            Gender: faceAnchor.Gender,
            Race: faceAnchor.Race);
    }

    /// <summary>Python: <c>average_face_identity</c>.</summary>
    public static Types.Face? AverageFaceIdentity(IReadOnlyList<Types.Face> faces)
    {
        if (faces.Count == 0)
        {
            return null;
        }

        var firstFace = faces[0];
        var embeddingLength = ((float[])firstFace.Embedding).Length;
        var embeddingNormLength = ((float[])firstFace.EmbeddingNorm).Length;
        var embeddingSum = new double[embeddingLength];
        var embeddingNormSum = new double[embeddingNormLength];

        foreach (var face in faces)
        {
            var embedding = (float[])face.Embedding;
            var embeddingNorm = (float[])face.EmbeddingNorm;

            for (var i = 0; i < embeddingLength; i++)
            {
                embeddingSum[i] += embedding[i];
            }

            for (var i = 0; i < embeddingNormLength; i++)
            {
                embeddingNormSum[i] += embeddingNorm[i];
            }
        }

        // Python: `numpy.mean(face_embeddings, axis = 0)`. FaceFusion.Tensors.NumPy operates
        // on float32 only (see its class remarks); this averages in double precision (numpy's
        // own accumulation for a stack of float32 arrays promotes to float64 internally) and
        // narrows to float32 only at the end, matching numpy.mean's actual arithmetic more
        // closely than an all-float32 accumulation would for larger face counts.
        var meanEmbedding = new float[embeddingLength];
        for (var i = 0; i < embeddingLength; i++)
        {
            meanEmbedding[i] = (float)(embeddingSum[i] / faces.Count);
        }

        var meanEmbeddingNorm = new float[embeddingNormLength];
        for (var i = 0; i < embeddingNormLength; i++)
        {
            meanEmbeddingNorm[i] = (float)(embeddingNormSum[i] / faces.Count);
        }

        return new Types.Face(
            Origin: firstFace.Origin,
            BoundingBox: firstFace.BoundingBox,
            ScoreSet: firstFace.ScoreSet,
            LandmarkSet: firstFace.LandmarkSet,
            Angle: firstFace.Angle,
            Embedding: meanEmbedding,
            EmbeddingNorm: meanEmbeddingNorm,
            Age: firstFace.Age,
            Gender: firstFace.Gender,
            Race: firstFace.Race);
    }

    /// <summary>Python: <c>scale_face</c>.</summary>
    public static Types.Face ScaleFace(Types.Face targetFace, Mat targetVisionFrame, Mat tempVisionFrame)
    {
        var scaleX = (double)tempVisionFrame.Cols / targetVisionFrame.Cols;
        var scaleY = (double)tempVisionFrame.Rows / targetVisionFrame.Rows;

        var boundingBox = (float[])targetFace.BoundingBox;
        var scaledBoundingBox = new[]
        {
            (float)(boundingBox[0] * scaleX),
            (float)(boundingBox[1] * scaleY),
            (float)(boundingBox[2] * scaleX),
            (float)(boundingBox[3] * scaleY),
        };

        var landmarkSet = new FaceLandmarkSet(
            Five: ScalePoints((float[,])targetFace.LandmarkSet.Five, scaleX, scaleY),
            FiveOn68: ScalePoints((float[,])targetFace.LandmarkSet.FiveOn68, scaleX, scaleY),
            SixtyEight: ScalePoints((float[,])targetFace.LandmarkSet.SixtyEight, scaleX, scaleY),
            SixtyEightOn5: ScalePoints((float[,])targetFace.LandmarkSet.SixtyEightOn5, scaleX, scaleY));

        // Python: `target_face._replace(bounding_box = ..., landmark_set = ...)` — a namedtuple
        // _replace, i.e. every other field is copied from targetFace unchanged.
        return targetFace with { BoundingBox = scaledBoundingBox, LandmarkSet = landmarkSet };
    }

    // -----------------------------------------------------------------
    // Internal geometry helpers
    // -----------------------------------------------------------------

    private static float[,] ScalePoints(float[,] points, double scaleX, double scaleY)
    {
        var rows = points.GetLength(0);
        var result = new float[rows, 2];

        for (var i = 0; i < rows; i++)
        {
            result[i, 0] = (float)(points[i, 0] * scaleX);
            result[i, 1] = (float)(points[i, 1] * scaleY);
        }

        return result;
    }

    private static float[] AveragePointsFlat(object first, object middle, double averageFactor)
    {
        var firstArray = (float[])first;
        var middleArray = (float[])middle;
        var result = new float[firstArray.Length];

        for (var i = 0; i < firstArray.Length; i++)
        {
            result[i] = (float)((firstArray[i] * (1 - averageFactor)) + (middleArray[i] * averageFactor));
        }

        return result;
    }

}
