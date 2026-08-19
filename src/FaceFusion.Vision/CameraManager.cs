using OpenCvSharp;

namespace FaceFusion.Vision;

/// <summary>
/// Port of <c>facefusion/camera_manager.py</c>.
///
/// <para>
/// <b>Not a static pool (PORT_CONVENTIONS.md rule 5).</b> Python keeps <c>CAMERA_POOL_SET</c>
/// as a module global and every function reaches into it; here the pool is an instance the
/// caller owns and disposes, the same shape <c>ContentAnalyser</c> already uses for its own
/// Python module-global counter. A camera is an exclusive OS resource — a process-wide pool
/// that nothing owns is exactly how a device ends up held open after the UI stops streaming.
/// </para>
/// </summary>
public sealed class CameraManager : IDisposable
{
    private readonly Dictionary<string, VideoCapture> _captures = new(StringComparer.Ordinal);
    private readonly object _lock = new();

    /// <summary>Python: <c>get_local_camera_capture</c>. Returns null when the device will not
    /// open, matching Python returning None from the dict lookup that was never populated.</summary>
    public VideoCapture? GetLocalCameraCapture(int cameraId) => GetCapture(cameraId.ToString(), () => new VideoCapture(cameraId));

    /// <summary>Python: <c>get_remote_camera_capture</c>. The "url" is anything OpenCV's
    /// VideoCapture accepts as a string source — an RTSP/HTTP stream, and also a plain file
    /// path, which is what makes this port's streaming testable without a physical device.</summary>
    public VideoCapture? GetRemoteCameraCapture(string cameraUrl) => GetCapture(cameraUrl, () => new VideoCapture(cameraUrl));

    private VideoCapture? GetCapture(string key, Func<VideoCapture> open)
    {
        lock (_lock)
        {
            if (_captures.TryGetValue(key, out var existing))
            {
                return existing;
            }

            var capture = open();

            if (capture.IsOpened())
            {
                _captures[key] = capture;
                return capture;
            }

            // Python leaks the unopened VideoCapture to the garbage collector; disposing it is
            // the same behaviour with the native handle released promptly.
            capture.Dispose();
            return null;
        }
    }

    /// <summary>Python: <c>clear_camera_pool</c> — releases every capture and empties the pool.</summary>
    public void ClearCameraPool()
    {
        lock (_lock)
        {
            foreach (var capture in _captures.Values)
            {
                capture.Release();
                capture.Dispose();
            }

            _captures.Clear();
        }
    }

    /// <summary>
    /// Python: <c>detect_local_camera_ids(id_start, id_end)</c> — the half-open range
    /// <c>range(id_start, id_end)</c>. Every id that opens stays in the pool, exactly as in
    /// Python (it calls <c>get_local_camera_capture</c>, which caches).
    /// </summary>
    public IReadOnlyList<int> DetectLocalCameraIds(int idStart, int idEnd)
    {
        var cameraIds = new List<int>();

        for (var cameraId = idStart; cameraId < idEnd; cameraId++)
        {
            if (GetLocalCameraCapture(cameraId) is not null)
            {
                cameraIds.Add(cameraId);
            }
        }

        return cameraIds;
    }

    public void Dispose() => ClearCameraPool();
}
