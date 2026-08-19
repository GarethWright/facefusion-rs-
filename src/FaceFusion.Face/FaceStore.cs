using System.Runtime.InteropServices;
using FaceFusion.Core;
using FaceFusion.Types;
using VisionHelper = FaceFusion.Vision.Vision;
using OpenCvSharp;

namespace FaceFusion.Face;

/// <summary>
/// Port of <c>facefusion/face_store.py</c> — a process-wide cache of already-detected
/// <see cref="Types.Face"/> lists, keyed by a CRC32 hash of the raw frame bytes, plus a
/// per-frame lock used by <c>face_creator.get_static_faces</c> to serialize the
/// detect-and-cache race for a given frame across threads.
///
/// <para>
/// <b>Deviation — instance class, not a module global.</b> Python holds
/// <c>FACE_STORE : FaceStore = {}</c> as a bare module dict guarded ad hoc (each accessor
/// re-hashes and does its own <c>dict.setdefault</c>; there is no single lock protecting the
/// dict itself, only the per-entry <c>threading.Lock</c> stashed inside each entry). Per
/// PORT_CONVENTIONS.md rule 5 / DOTNET_PORT_PLAN.md §3 ("no global mutable state"), this is an
/// instance class with the backing dictionary guarded by a private <see cref="_lock"/>,
/// consistent with how <c>ProcessManager</c>, <c>Logger</c> and
/// <c>FaceFusion.Inference.InferenceManager</c> were ported. Callers that want Python's
/// process-global sharing behaviour should hold and share one <see cref="FaceStore"/> instance
/// (e.g. as a DI singleton). Unlike Python, the top-level dictionary access itself
/// (create-entry-if-absent) is now genuinely thread-safe rather than racy, which is a
/// strengthening, not a behavioural change any caller can observe.
/// </para>
///
/// <para>
/// <b>No separate "face track" store here.</b> The current <c>facefusion/face_store.py</c>
/// contains only the frame-hash → faces cache described above; the <c>FaceTrack</c> alias
/// (<c>Dict[int, Face]</c>, keyed by track id rather than frame hash) lives in
/// <c>facefusion/face_tracker.py</c>, which is a different module, out of this assignment's
/// scope (and explicitly off-limits — see the port instructions). If a "face track" store was
/// expected here, it is not present in the Python source being ported; this port stays
/// faithful to what <c>face_store.py</c> actually contains.
/// </para>
/// </summary>
public sealed class FaceStore
{
    private sealed class Entry
    {
        public readonly object Lock = new();
        public IReadOnlyList<Types.Face>? Faces;
    }

    private readonly object _lock = new();
    private readonly Dictionary<string, Entry> _faceStore = new(StringComparer.Ordinal);

    /// <summary>
    /// Python: <c>get_faces</c>. Returns <see langword="null"/> when <paramref name="visionFrame"/>
    /// is not a vision frame, or no faces have been cached for it yet. Does not take ownership
    /// of <paramref name="visionFrame"/>.
    /// </summary>
    public IReadOnlyList<Types.Face>? GetFaces(Mat? visionFrame)
    {
        if (!VisionHelper.IsVisionFrame(visionFrame))
        {
            return null;
        }

        var visionHash = ComputeVisionHash(visionFrame!);

        lock (_lock)
        {
            // Python: `if FACE_STORE.get(vision_hash): return FACE_STORE.get(vision_hash).get('faces')`
            // — an entry created only via resolve_lock() (no 'faces' key yet) is falsy-checked
            // away in Python only because an empty dict is falsy; here an Entry always exists
            // once created, so the equivalent check is "Faces was actually set".
            if (_faceStore.TryGetValue(visionHash, out var entry) && entry.Faces is not null)
            {
                return entry.Faces;
            }
        }

        return null;
    }

    /// <summary>
    /// Python: <c>set_faces</c>. No-op when <paramref name="visionFrame"/> is not a vision
    /// frame. Does not take ownership of <paramref name="visionFrame"/>.
    /// </summary>
    public void SetFaces(Mat? visionFrame, IReadOnlyList<Types.Face> faces)
    {
        if (!VisionHelper.IsVisionFrame(visionFrame))
        {
            return;
        }

        var visionHash = ComputeVisionHash(visionFrame!);

        lock (_lock)
        {
            var entry = GetOrCreateEntryLocked(visionHash);
            entry.Faces = faces;
        }
    }

    /// <summary>
    /// Python: <c>resolve_lock</c>. Returns a lock object scoped to this frame's hash — callers
    /// take a CLR <c>lock</c> on the returned object to serialize a detect-and-cache race for
    /// the same frame, the same role Python's <c>threading.Lock</c> plays under
    /// <c>with face_store.resolve_lock(vision_frame):</c>. Returns a fresh, unshared object
    /// (never stored) when <paramref name="visionFrame"/> is not a vision frame, matching
    /// Python's <c>return threading.Lock()</c> fallback. Does not take ownership of
    /// <paramref name="visionFrame"/>.
    /// </summary>
    public object ResolveLock(Mat? visionFrame)
    {
        if (!VisionHelper.IsVisionFrame(visionFrame))
        {
            return new object();
        }

        var visionHash = ComputeVisionHash(visionFrame!);

        lock (_lock)
        {
            return GetOrCreateEntryLocked(visionHash).Lock;
        }
    }

    /// <summary>Python: <c>clear_faces</c>.</summary>
    public void ClearFaces()
    {
        lock (_lock)
        {
            _faceStore.Clear();
        }
    }

    private Entry GetOrCreateEntryLocked(string visionHash)
    {
        if (!_faceStore.TryGetValue(visionHash, out var entry))
        {
            entry = new Entry();
            _faceStore[visionHash] = entry;
        }

        return entry;
    }

    /// <summary>
    /// Python: <c>hash_helper.create_hash(vision_frame.tobytes())</c>. <c>Mat.tobytes()</c>
    /// (numpy) always yields the C-contiguous row-major byte layout; vision frames created by
    /// <c>FaceFusion.Vision.Vision</c> (and by OpenCV generally) are already C-contiguous, so
    /// this only defensively clones when a caller hands in a non-contiguous view (e.g. a
    /// <c>Mat</c> ROI), to keep the byte sequence identical to what numpy would produce either
    /// way.
    /// </summary>
    private static string ComputeVisionHash(Mat visionFrame)
    {
        var continuous = visionFrame.IsContinuous() ? visionFrame : visionFrame.Clone();

        try
        {
            var length = checked((int)(continuous.Total() * continuous.ElemSize()));
            var bytes = new byte[length];
            Marshal.Copy(continuous.Data, bytes, 0, length);
            return HashHelper.CreateHash(bytes);
        }
        finally
        {
            if (!ReferenceEquals(continuous, visionFrame))
            {
                continuous.Dispose();
            }
        }
    }
}
