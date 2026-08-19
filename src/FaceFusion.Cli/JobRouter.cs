using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Types;

namespace FaceFusion.Cli;

/// <summary>
/// Port of <c>route_job_manager</c> and <c>route_job_runner</c> from
/// <c>facefusion/core.py</c>.
///
/// Returns the process error code rather than calling <c>hard_exit</c>: only
/// <c>Program.Main</c> exits, per <see cref="ExitHelper"/>. The codes themselves are
/// reproduced exactly — scripts branch on them.
/// </summary>
public sealed class JobRouter
{
    private readonly JobManager _jobManager;
    private readonly Logger _logger;

    public JobRouter(JobManager jobManager, Logger logger)
    {
        _jobManager = jobManager;
        _logger = logger;
    }

    /// <summary>Python: <c>route_job_manager</c>. Returns 0 on success, 1 on failure.</summary>
    public int RouteJobManager(
        string command,
        string? jobId,
        int? stepIndex,
        JobStatus jobStatus,
        bool haltOnError,
        IReadOnlyDictionary<string, object?> stepArgs)
    {
        switch (command)
        {
            case "job-list":
            {
                var (headers, contents) = JobList.ComposeJobList(_jobManager, jobStatus);

                if (contents.Count > 0)
                {
                    foreach (var line in CliHelper.ComposeTable(headers, contents))
                    {
                        Console.WriteLine(line);
                    }

                    return 0;
                }

                return 1;
            }

            case "job-create":
                return Report(_jobManager.CreateJob(Require(jobId)), "job_created", "job_not_created", jobId);

            case "job-submit":
                return Report(_jobManager.SubmitJob(Require(jobId)), "job_submitted", "job_not_submitted", jobId);

            case "job-submit-all":
                return Report(_jobManager.SubmitJobs(haltOnError), "job_all_submitted", "job_all_not_submitted", null);

            case "job-delete":
                return Report(_jobManager.DeleteJob(Require(jobId)), "job_deleted", "job_not_deleted", jobId);

            case "job-delete-all":
                return Report(_jobManager.DeleteJobs(haltOnError), "job_all_deleted", "job_all_not_deleted", null);

            case "job-add-step":
                return Report(_jobManager.AddStep(Require(jobId), stepArgs), "job_step_added", "job_step_not_added", jobId);

            case "job-remix-step":
                return Report(
                    _jobManager.RemixStep(Require(jobId), RequireIndex(stepIndex), stepArgs),
                    "job_remix_step_added", "job_remix_step_not_added", jobId);

            case "job-insert-step":
                return Report(
                    _jobManager.InsertStep(Require(jobId), RequireIndex(stepIndex), stepArgs),
                    "job_step_inserted", "job_step_not_inserted", jobId);

            case "job-remove-step":
                return Report(
                    _jobManager.RemoveStep(Require(jobId), RequireIndex(stepIndex)),
                    "job_step_removed", "job_step_not_removed", jobId);

            default:
                // Python falls through every branch and returns 1.
                return 1;
        }
    }

    /// <summary>
    /// Python: <c>route_job_runner</c>. Returns 0 on success, 1 on a processing failure,
    /// and 2 for an unrecognised command — note the 2, which differs from
    /// <see cref="RouteJobManager"/>'s fall-through of 1.
    /// </summary>
    public int RouteJobRunner(
        string command,
        string? jobId,
        bool haltOnError,
        ProcessStep processStep,
        ConcatVideoStep concatVideo)
    {
        switch (command)
        {
            case "job-run":
                _logger.Info(Message("running_job", jobId), nameof(JobRouter));
                return ReportRun(JobRunner.RunJob(_jobManager, Require(jobId), processStep, concatVideo), jobId);

            case "job-run-all":
                _logger.Info(Message("running_jobs", null), nameof(JobRouter));
                return ReportRunAll(JobRunner.RunJobs(_jobManager, processStep, concatVideo, haltOnError));

            case "job-retry":
                _logger.Info(Message("retrying_job", jobId), nameof(JobRouter));
                return ReportRun(JobRunner.RetryJob(_jobManager, Require(jobId), processStep, concatVideo), jobId);

            case "job-retry-all":
                _logger.Info(Message("retrying_jobs", null), nameof(JobRouter));
                return ReportRunAll(JobRunner.RetryJobs(_jobManager, processStep, concatVideo, haltOnError));

            default:
                return 2;
        }
    }

    private int Report(bool succeeded, string successKey, string failureKey, string? jobId)
    {
        if (succeeded)
        {
            _logger.Info(Message(successKey, jobId), nameof(JobRouter));
            return 0;
        }

        _logger.Error(Message(failureKey, jobId), nameof(JobRouter));
        return 1;
    }

    // Python logs both the success and failure of a run at info level, not error.
    private int ReportRun(bool succeeded, string? jobId)
    {
        _logger.Info(Message(succeeded ? "processing_job_succeeded" : "processing_job_failed", jobId), nameof(JobRouter));
        return succeeded ? 0 : 1;
    }

    private int ReportRunAll(bool succeeded)
    {
        _logger.Info(Message(succeeded ? "processing_jobs_succeeded" : "processing_jobs_failed", null), nameof(JobRouter));
        return succeeded ? 0 : 1;
    }

    private static string Message(string key, string? jobId)
    {
        var template = Translator.Get(key) ?? key;
        return jobId is null ? template : Translator.Format(template, ("job_id", jobId));
    }

    private static string Require(string? jobId) =>
        jobId ?? throw new ArgumentException("this command requires --job-id");

    private static int RequireIndex(int? stepIndex) =>
        stepIndex ?? throw new ArgumentException("this command requires --step-index");
}
