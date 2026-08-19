using FaceFusion.Types;

namespace FaceFusion.Core;

/// <summary>
/// Port of <c>facefusion/process_manager.py</c>.
///
/// Deviation from Python: the Python module keeps <c>PROCESS_STATE</c> as a module-level
/// global. Per port convention rule 5 (no global mutable state), this is an instance
/// class with the state held in a private field, guarded by a lock so it is safe to use
/// from multiple threads and independently testable. Callers that want module-global
/// behaviour should share a single instance (e.g. via DI).
/// </summary>
public sealed class ProcessManager
{
    private readonly object _lock = new();
    private ProcessState _processState = ProcessState.Pending;

    /// <summary>Python: <c>get_process_state</c>.</summary>
    public ProcessState GetProcessState()
    {
        lock (_lock)
        {
            return _processState;
        }
    }

    /// <summary>Python: <c>set_process_state</c>.</summary>
    public void SetProcessState(ProcessState processState)
    {
        lock (_lock)
        {
            _processState = processState;
        }
    }

    /// <summary>Python: <c>is_checking</c>.</summary>
    public bool IsChecking() => GetProcessState() == ProcessState.Checking;

    /// <summary>Python: <c>is_processing</c>.</summary>
    public bool IsProcessing() => GetProcessState() == ProcessState.Processing;

    /// <summary>Python: <c>is_stopping</c>.</summary>
    public bool IsStopping() => GetProcessState() == ProcessState.Stopping;

    /// <summary>Python: <c>is_pending</c>.</summary>
    public bool IsPending() => GetProcessState() == ProcessState.Pending;

    /// <summary>Python: <c>check</c>.</summary>
    public void Check() => SetProcessState(ProcessState.Checking);

    /// <summary>Python: <c>start</c>.</summary>
    public void Start() => SetProcessState(ProcessState.Processing);

    /// <summary>Python: <c>stop</c>.</summary>
    public void Stop() => SetProcessState(ProcessState.Stopping);

    /// <summary>Python: <c>end</c>.</summary>
    public void End() => SetProcessState(ProcessState.Pending);

    /// <summary>
    /// Convenience scope not present verbatim in the Python module (which has no
    /// context manager of its own), added so callers can bracket processing work with
    /// <c>Start</c>/<c>End</c> via <c>using</c>. Disposing calls <see cref="End"/>.
    /// </summary>
    public IDisposable Manage()
    {
        Start();
        return new ManageScope(this);
    }

    private sealed class ManageScope : IDisposable
    {
        private readonly ProcessManager _processManager;
        private bool _disposed;

        public ManageScope(ProcessManager processManager) => _processManager = processManager;

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            _processManager.End();
        }
    }
}
