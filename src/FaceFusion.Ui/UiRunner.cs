using FaceFusion.Cli;
using FaceFusion.Types;
using FaceFusion.Core;
using FaceFusion.Jobs;

namespace FaceFusion.Ui;

/// <summary>
/// Port of <c>facefusion/uis/components/instant_runner.py</c> — the START button. Python calls
/// <c>core.conditional_process()</c> straight from the click handler; this hands the same flat
/// args bag to <see cref="HeadlessRunner.ProcessHeadless"/>, so a UI run and the equivalent
/// <c>headless-run</c> command execute the same code path with the same values. There is no
/// UI-only processing path to drift.
/// </summary>
public sealed class UiRunner
{
    private readonly UiState _state;
    private readonly UiTerminal _terminal;
    private readonly object _lock = new();

    public UiRunner(UiState state, UiTerminal terminal)
    {
        _state = state;
        _terminal = terminal;
    }

    public bool IsRunning { get; private set; }

    /// <summary>The output path of the last successful run, for the output preview pane.</summary>
    public string? LastOutputPath { get; private set; }

    public event Action? StateChanged;

    /// <summary>
    /// Python: <c>start()</c>. Runs off the request thread so the SignalR circuit stays
    /// responsive — Gradio does the same by running the handler in a worker.
    /// </summary>
    public async Task StartAsync()
    {
        lock (_lock)
        {
            if (IsRunning)
            {
                return;
            }

            IsRunning = true;
        }

        StateChanged?.Invoke();

        try
        {
            var args = _state.BuildArgs();
            var outputPath = _state.GetString("output_path");

            await Task.Run(() =>
            {
                var logger = new Logger(_terminal);
                logger.Init(EnumNames.FromWireName<FaceFusion.Types.LogLevel>(_state.GetString("log_level") ?? "info"));

                var jobsPath = _state.GetString("jobs_path");
                var jobManager = new JobManager(string.IsNullOrWhiteSpace(jobsPath) ? ".jobs" : jobsPath);
                jobManager.InitJobs();

                var errorCode = HeadlessRunner.ProcessHeadless(args, jobManager, logger);

                if (errorCode == 0 && !string.IsNullOrWhiteSpace(outputPath) && File.Exists(outputPath))
                {
                    LastOutputPath = outputPath;
                }
            }).ConfigureAwait(false);
        }
        catch (Exception exception)
        {
            // Same reasoning as HeadlessRunner's own catch: a UI that goes quiet on failure is
            // worse to use than one that shows the exception.
            _terminal.WriteLine($"[FACEFUSION.UI] {exception.GetType().Name}: {exception.Message}");
        }
        finally
        {
            lock (_lock)
            {
                IsRunning = false;
            }

            StateChanged?.Invoke();
        }
    }
}
