// Ported from Python facefusion/types.py.
//
// Per PORT_CONVENTIONS.md, simple `TypeAlias` declarations do not get C# wrapper types —
// call sites just use the underlying BCL type directly. This file is documentation only (no
// executable code) recording where each Python alias landed, so a later port of a module that
// references one of these names has a single place to check.
//
// Scalars
//   Scale, Score               : float            -> double
//   Angle                      : int               -> int   (see also the enum-adjacent Face.Angle field)
//   BitRate, SampleRate        : int               -> int
//   Fps, Duration              : float             -> double
//   Command, TableHeader       : str               -> string
//   ColorTransfer              : str               -> string
//   Age                        : range             -> System.Range (see Face.cs)
//
// Byte buffers
//   Buffer, AudioBuffer        : bytes             -> byte[]
//
// numpy-array-backed aliases (Detection, Prediction, BoundingBox, FaceLandmark5,
// FaceLandmark68, Embedding, VisionFrame, Mask, Points, Distance, Matrix, Anchors,
// Translation, Audio, AudioChunk, AudioFrame, Spectrogram, Mel, MelFilterBank, Voice,
// VoiceChunk, ModelInitializer): all `NDArray[Any]` (or a dtype-narrowed NDArray) in Python.
// FaceFusion.Types has no tensor dependency (PORT_CONVENTIONS.md forbids adding one here), so
// wherever one of these appears as a field within this project (FaceLandmarkSet, Face), the
// field is typed `object`. FaceFusion.Tensors owns the real array type and later layers should
// use it directly rather than reintroducing a named alias for `object`.
//
// Dict/collection aliases that are never used as a field inside a TypedDict/record in this
// file are represented directly as concrete `IReadOnlyDictionary<TKey, TValue>` /
// `IReadOnlyList<T>` at their point of use (mostly in Choices.cs) rather than as a named type:
//   FaceStore : Dict[str, FaceSet]                       -> IReadOnlyDictionary<string, FaceSet>
//   FaceTrack : Dict[int, Face]                          -> IReadOnlyDictionary<int, Face>
//   Locales : Dict[Language, Dict[str, Any]]             -> IReadOnlyDictionary<Language, IReadOnlyDictionary<string, object?>>
//   LocalePoolSet : Dict[str, Locales]                   -> IReadOnlyDictionary<string, IReadOnlyDictionary<Language, IReadOnlyDictionary<string, object?>>>
//   CameraCaptureSet : Dict[str, cv2.VideoCapture]        -> IReadOnlyDictionary<string, object> (see CameraPoolSet.cs)
//   VisionFrameSet : Dict[int, VisionFrame]              -> IReadOnlyDictionary<int, object>
//   FrameStoreSet : Dict[str, VisionFrameSet]            -> IReadOnlyDictionary<string, IReadOnlyDictionary<int, object>>
//   Args : Dict[str, Any]                                -> IReadOnlyDictionary<string, object?>
//   Content : Dict[str, Any]                             -> IReadOnlyDictionary<string, object?>
//   CommandSet : Dict[str, List[Command]]                -> IReadOnlyDictionary<string, IReadOnlyList<string>>
//   WarpTemplateSet : Dict[WarpTemplate, NDArray[Any]]   -> IReadOnlyDictionary<WarpTemplate, object>
//   FaceDetectorSet : Dict[FaceDetectorModel, List[str]] -> IReadOnlyDictionary<FaceDetectorModel, IReadOnlyList<string>> (see Choices.cs)
//   FaceMaskAreaSet, FaceMaskRegionSet, AudioTypeSet, ImageTypeSet, VideoTypeSet,
//   BenchmarkSet, ExecutionProviderSet, DownloadProviderSet, LogLevelSet               -> see Choices.cs
//   ModelOptions : Dict[str, Any]                        -> IReadOnlyDictionary<string, object?>
//   ModelSet : Dict[str, ModelOptions]                   -> IReadOnlyDictionary<string, IReadOnlyDictionary<string, object?>>
//   InferenceProvider : Any                              -> object
//   InferenceOptionSet : Dict[str, Any]                  -> IReadOnlyDictionary<string, object?>
//   InferencePool : Dict[str, InferenceSession]          -> IReadOnlyDictionary<string, object> (ONNX Runtime type owned by FaceFusion.Inference)
//   InferencePoolSet : Dict[AppContext, Dict[str, InferencePool]] -> IReadOnlyDictionary<AppContext, IReadOnlyDictionary<string, IReadOnlyDictionary<string, object>>>
//   DownloadSet : Dict[str, Download]                    -> IReadOnlyDictionary<string, Download>
//   JobSet : Dict[str, Job]                              -> IReadOnlyDictionary<string, Job>
//   JobOutputSet : Dict[str, List[str]]                  -> IReadOnlyDictionary<string, IReadOnlyList<string>>
//   FrameSet : Dict[int, str]                            -> IReadOnlyDictionary<int, string>
//   StateSet : Dict[AppContext, State]                   -> IReadOnlyDictionary<AppContext, State>
//   TableContent : Any                                   -> object?
//
// ErrorCode = Literal[0, 1, 2, 3, 4] is a plain process exit code, not a closed enumeration
// with named meanings in the Python source (see facefusion/exit_helper.py) — represented as
// `int` at call sites rather than as an enum.
