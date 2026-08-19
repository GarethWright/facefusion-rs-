using FaceFusion.Cli;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>route_job_manager</c> / <c>route_job_runner</c> in
/// <c>facefusion/core.py</c>. Exercises the exit codes, which scripts branch on, without
/// needing models or media.
/// </summary>
public sealed class JobRouterTests : IDisposable
{
    private readonly string _jobsPath;
    private readonly JobManager _jobManager;
    private readonly JobRouter _router;

    public JobRouterTests()
    {
        _jobsPath = Path.Combine(Path.GetTempPath(), "facefusion-router-" + Guid.NewGuid().ToString("N"));
        _jobManager = new JobManager(_jobsPath);
        _jobManager.InitJobs();
        _router = new JobRouter(_jobManager, new Logger());
    }

    public void Dispose()
    {
        if (Directory.Exists(_jobsPath))
        {
            Directory.Delete(_jobsPath, recursive: true);
        }
    }

    private int Route(string command, string? jobId = null, int? stepIndex = null,
        JobStatus jobStatus = JobStatus.Drafted, bool haltOnError = false,
        IReadOnlyDictionary<string, object?>? stepArgs = null)
        => _router.RouteJobManager(command, jobId, stepIndex, jobStatus, haltOnError,
            stepArgs ?? new Dictionary<string, object?>());

    [Fact]
    public void CreateThenSubmitReturnsZero()
    {
        Assert.Equal(0, Route("job-create", "job-alpha"));
        Assert.Equal(0, Route("job-add-step", "job-alpha", stepArgs: new Dictionary<string, object?> { ["target_path"] = "t.jpg" }));
        Assert.Equal(0, Route("job-submit", "job-alpha"));
    }

    [Fact]
    public void CreatingTheSameJobTwiceReturnsOne()
    {
        Assert.Equal(0, Route("job-create", "job-beta"));
        Assert.Equal(1, Route("job-create", "job-beta"));
    }

    /// <summary>Submitting a job with no steps fails, matching job_manager.submit_job.</summary>
    [Fact]
    public void SubmittingAnEmptyJobReturnsOne()
    {
        Assert.Equal(0, Route("job-create", "job-empty"));
        Assert.Equal(1, Route("job-submit", "job-empty"));
    }

    [Fact]
    public void DeleteReturnsZeroThenOne()
    {
        Assert.Equal(0, Route("job-create", "job-gamma"));
        Assert.Equal(0, Route("job-delete", "job-gamma"));
        Assert.Equal(1, Route("job-delete", "job-gamma"));
    }

    /// <summary>job-list returns 1 when there is nothing to list, not 0.</summary>
    [Fact]
    public void ListReturnsOneWhenEmptyAndZeroWhenPopulated()
    {
        Assert.Equal(1, Route("job-list", jobStatus: JobStatus.Drafted));

        Route("job-create", "job-delta");
        Assert.Equal(0, Route("job-list", jobStatus: JobStatus.Drafted));
    }

    [Fact]
    public void StepCommandsReturnZeroThenOneOnMissingStep()
    {
        Route("job-create", "job-steps");
        var args = new Dictionary<string, object?> { ["target_path"] = "t.jpg" };

        Assert.Equal(0, Route("job-add-step", "job-steps", stepArgs: args));
        Assert.Equal(0, Route("job-insert-step", "job-steps", stepIndex: 0, stepArgs: args));
        Assert.Equal(0, Route("job-remix-step", "job-steps", stepIndex: 0, stepArgs: args));
        Assert.Equal(0, Route("job-remove-step", "job-steps", stepIndex: 0));
        Assert.Equal(1, Route("job-remove-step", "job-steps", stepIndex: 99));
    }

    /// <summary>
    /// Python's route_job_manager falls through to `return 1` for an unknown command,
    /// while route_job_runner returns 2. The asymmetry is deliberate, so it is pinned.
    /// </summary>
    [Fact]
    public void UnknownCommandReturnsOneFromManagerAndTwoFromRunner()
    {
        Assert.Equal(1, Route("job-does-not-exist"));

        var runnerCode = _router.RouteJobRunner(
            "job-does-not-exist", null, false,
            (_, _, _) => true,
            (_, _) => true);

        Assert.Equal(2, runnerCode);
    }

    /// <summary>
    /// A step whose process callback returns true still fails the job when it produced no
    /// output file, because finalize_steps cannot collect and move a non-existent output.
    ///
    /// This is verified Python behaviour, not an assumption — the equivalent script
    /// (create, add_step, submit, then run_job with a lambda returning True) returns
    /// False in Python too. An earlier version of this test asserted 0 and was wrong.
    /// </summary>
    [Fact]
    public void RunJobFailsWhenStepsProduceNoOutput()
    {
        Route("job-create", "job-run-ok");
        Route("job-add-step", "job-run-ok", stepArgs: new Dictionary<string, object?> { ["target_path"] = "t.jpg" });
        Route("job-submit", "job-run-ok");

        var code = _router.RouteJobRunner("job-run", "job-run-ok", false, (_, _, _) => true, (_, _) => true);

        Assert.Equal(1, code);
    }

    [Fact]
    public void RunJobReturnsOneWhenAStepFails()
    {
        Route("job-create", "job-run-bad");
        Route("job-add-step", "job-run-bad", stepArgs: new Dictionary<string, object?> { ["target_path"] = "t.jpg" });
        Route("job-submit", "job-run-bad");

        var code = _router.RouteJobRunner("job-run", "job-run-bad", false, (_, _, _) => false, (_, _) => true);

        Assert.Equal(1, code);
    }
}
