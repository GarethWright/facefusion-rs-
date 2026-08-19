using System.Globalization;
using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Face;
using FaceFusion.Types;
using FaceFusion.Vision;
using FaceFusion.Workflows;
using OpenCvSharp;

namespace FaceFusion.Ui;

/// <summary>
/// Port of <c>facefusion/uis/components/webcam.py</c> plus <c>webcam_options.py</c> — the live
/// path's UI half. The processing itself is <see cref="Streamer"/>; this owns the capture, the
/// worker loop and the frame the page displays.
///
/// <para>
/// <b>The camera is read on the server, not in the browser.</b> Plan §6 assumed the webcam
/// layout would need <c>getUserMedia</c> and a frame-upload channel. It does not: Python's
/// webcam layout opens a <i>local</i> device with <c>cv2.VideoCapture(device_id)</c> in the
/// server process, so following it keeps the two implementations identical and needs no JS
/// interop at all. Frames travel the other way, server to browser, exactly as the preview's do.
/// </para>
///
/// <para>
/// <b>Addition: a remote source.</b> Python ships <c>get_remote_camera_capture(camera_url)</c>
/// but no UI that reaches it. This exposes it, because a URL or file path is the only way to
/// exercise the live path on a machine with no camera — including this one, which is where the
/// port's streaming was verified.
/// </para>
/// </summary>
public sealed class UiWebcam : IDisposable
{
    private readonly UiState _state;
    private readonly UiTerminal _terminal;
    private readonly object _lock = new();

    private CameraManager? _cameraManager;
    private CancellationTokenSource? _cancellation;
    private Task? _streamTask;

    public UiWebcam(UiState state, UiTerminal terminal)
    {
        _state = state;
        _terminal = terminal;
    }

    /// <summary>Python: <c>uis/choices.py</c>'s <c>webcam_modes</c>.</summary>
    public static readonly IReadOnlyList<string> WebcamModes = new[] { "inline", "udp", "v4l2" };

    /// <summary>Python: <c>uis/choices.py</c>'s <c>webcam_resolutions</c>.</summary>
    public static readonly IReadOnlyList<string> WebcamResolutions = new[]
    {
        "320x240", "640x480", "800x600", "1024x768", "1280x720", "1280x960", "1920x1080",
    };

    public int DeviceId { get; set; }

    /// <summary>Empty means "use <see cref="DeviceId"/>"; anything else is passed to
    /// <c>CameraManager.GetRemoteCameraCapture</c>.</summary>
    public string RemoteSource { get; set; } = string.Empty;

    public string Mode { get; set; } = "inline";

    public string Resolution { get; set; } = "640x480";

    /// <summary>Python: the fps slider, 1..30, defaulting to 30.</summary>
    public int Fps { get; set; } = 30;

    public bool IsStreaming { get; private set; }

    public string? ImageDataUri { get; private set; }

    public string? LastError { get; private set; }

    public event Action? Changed;

    /// <summary>Python: <c>detect_local_camera_ids(0, 10)</c>. Opening ten devices is slow on a
    /// machine with none, so the result is cached for the process's life — Python re-detects on
    /// every render, but its pool makes the repeat calls cheap in the same way.</summary>
    public IReadOnlyList<int> DetectLocalCameraIds()
    {
        lock (_lock)
        {
            _detectedCameraIds ??= new CameraManager().DetectLocalCameraIds(0, 10);
            return _detectedCameraIds;
        }
    }

    private IReadOnlyList<int>? _detectedCameraIds;

    /// <summary>Python: <c>start()</c>.</summary>
    public void Start()
    {
        lock (_lock)
        {
            if (IsStreaming)
            {
                return;
            }

            IsStreaming = true;
            LastError = null;
        }

        // Python: state_manager.init_item('face_selector_mode', 'one'). A live stream has no
        // reference frame to select against, so reference mode would never resolve a face.
        _state.SetString("face_selector_mode", "one");

        Changed?.Invoke();

        _cancellation = new CancellationTokenSource();
        var token = _cancellation.Token;
        _streamTask = Task.Run(() => Run(token), token);
    }

    /// <summary>Python: <c>stop()</c> — <c>clear_camera_pool()</c> and blank the image.</summary>
    public void Stop()
    {
        CancellationTokenSource? cancellation;

        lock (_lock)
        {
            if (!IsStreaming)
            {
                return;
            }

            cancellation = _cancellation;
        }

        cancellation?.Cancel();
    }

    private void Run(CancellationToken cancellationToken)
    {
        var logger = new Logger(_terminal);
        logger.Init(EnumNames.FromWireName<FaceFusion.Types.LogLevel>(_state.GetString("log_level") ?? "info"));

        var args = _state.BuildArgs();
        var processorNames = _state.GetList("processors");
        var built = new List<ProcessorStepFactory.BuiltStep>();
        FacePipelineFactory.Resources? faceResources = null;
        System.Diagnostics.Process? stream = null;

        try
        {
            foreach (var name in processorNames)
            {
                if (!ProcessorStepFactory.PreCheck(name, args))
                {
                    LastError = $"processor '{name}' pre-check failed — its model files are missing or unreadable";
                    return;
                }
            }

            if (processorNames.Any(FacePipelineFactory.Requires))
            {
                faceResources = FacePipelineFactory.Build(args);
            }

            foreach (var name in processorNames)
            {
                built.Add(ProcessorStepFactory.Build(name, args, faceResources));
            }

            _cameraManager = new CameraManager();

            var capture = string.IsNullOrWhiteSpace(RemoteSource)
                ? _cameraManager.GetLocalCameraCapture(DeviceId)
                : _cameraManager.GetRemoteCameraCapture(RemoteSource);

            if (capture is null)
            {
                LastError = string.IsNullOrWhiteSpace(RemoteSource)
                    ? $"could not open camera device {DeviceId}"
                    : $"could not open '{RemoteSource}'";
                return;
            }

            var resolution = FaceFusion.Vision.Vision.UnpackResolution(Resolution);
            capture.Set(VideoCaptureProperties.FrameWidth, resolution.Width);
            capture.Set(VideoCaptureProperties.FrameHeight, resolution.Height);
            capture.Set(VideoCaptureProperties.Fps, Fps);

            if (Mode is "udp" or "v4l2")
            {
                stream = Streamer.OpenStream(EnumNames.FromWireName<StreamMode>(Mode), Resolution, Fps, logger);
            }

            foreach (var frame in Streamer.MultiProcessCapture(
                capture,
                Fps,
                built.Select(b => b.Step).ToArray(),
                _state.GetList("source_paths"),
                _state.GetInt("execution_thread_count"),
                new ContentAnalyser(),
                HeadlessRunner.ResolveModelsDirectory(),
                _state.GetList("execution_device_ids").Select(id => int.Parse(id, CultureInfo.InvariantCulture)).ToArray(),
                ReadExecutionProviders(),
                cancellationToken,
                logger))
            {
                using (frame)
                {
                    using var fitted = FaceFusion.Vision.Vision.FitCoverFrame(frame, resolution);

                    if (Mode == "inline")
                    {
                        Cv2.ImEncode(".jpg", fitted, out var jpeg, new[] { (int)ImwriteFlags.JpegQuality, 80 });
                        ImageDataUri = "data:image/jpeg;base64," + Convert.ToBase64String(jpeg);
                        Changed?.Invoke();
                    }

                    if (stream is not null)
                    {
                        try
                        {
                            // Python: stream.stdin.write(capture_vision_frame.data) — the raw
                            // pixel bytes, which is what ffmpeg's rawvideo input expects.
                            var byteCount = (int)(fitted.Total() * fitted.ElemSize());
                            var bytes = new byte[byteCount];
                            System.Runtime.InteropServices.Marshal.Copy(fitted.Data, bytes, 0, byteCount);
                            stream.StandardInput.BaseStream.Write(bytes, 0, byteCount);
                        }
                        catch (Exception)
                        {
                            // Python swallows this too: a stream sink that has gone away must
                            // not take the capture loop down with it.
                        }
                    }
                }
            }
        }
        catch (OperationCanceledException)
        {
            // Stop() was pressed — the normal way a stream ends.
        }
        catch (Exception exception)
        {
            LastError = $"{exception.GetType().Name}: {exception.Message}";
            logger.Debug(exception.ToString(), "facefusion.uis.webcam");
        }
        finally
        {
            try
            {
                stream?.StandardInput.Close();
                stream?.WaitForExit(5000);
            }
            catch (Exception)
            {
                // The ffmpeg process is already gone.
            }

            stream?.Dispose();

            foreach (var step in built)
            {
                step.Resource.Dispose();
            }

            faceResources?.Dispose();

            // Python: clear_camera_pool().
            _cameraManager?.Dispose();
            _cameraManager = null;

            lock (_lock)
            {
                IsStreaming = false;
            }

            Changed?.Invoke();
        }
    }

    private IReadOnlyList<ExecutionProvider> ReadExecutionProviders()
    {
        var names = _state.GetList("execution_providers");

        return names.Count > 0
            ? names.Select(EnumNames.FromWireName<ExecutionProvider>).ToArray()
            : new[] { ExecutionProvider.Cpu };
    }

    public void Dispose()
    {
        Stop();

        try
        {
            _streamTask?.Wait(TimeSpan.FromSeconds(10));
        }
        catch (Exception)
        {
            // Shutting down; a stream that will not stop cleanly must not block the process.
        }

        _cancellation?.Dispose();
        _cameraManager?.Dispose();
    }
}
