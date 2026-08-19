using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.Jobs;

/// <summary>
/// Delegate matching Python's <c>facefusion.ffmpeg.concat_video</c>, as used by
/// <c>job_runner.finalize_steps</c> to stitch a job step's per-step temp video outputs
/// back into one file. <c>FaceFusion.Jobs</c> does not (and per the solution layout in
/// docs/DOTNET_PORT_PLAN.md §3 should not) reference <c>FaceFusion.Media</c>, so the
/// concat operation is injected as a delegate — the same treatment
/// <see cref="ProcessStep"/> already gets from the Python source, and consistent with
/// "take it as a parameter instead" from port convention rule 5.
/// </summary>
public delegate bool ConcatVideoStep(string outputPath, IReadOnlyList<string> tempOutputPaths);

/// <summary>
/// Port of <c>facefusion/jobs/job_runner.py</c>.
/// </summary>
public static class JobRunner
{
    /// <summary>Python: <c>run_job</c>.</summary>
    public static bool RunJob(JobManager jobManager, string jobId, ProcessStep processStep, ConcatVideoStep concatVideo)
    {
        var queuedJobIds = jobManager.FindJobIds(JobStatus.Queued);

        if (queuedJobIds.Contains(jobId))
        {
            if (RunSteps(jobManager, jobId, processStep) && FinalizeSteps(jobManager, jobId, concatVideo))
            {
                CleanSteps(jobManager, jobId);
                return jobManager.MoveJobFile(jobId, JobStatus.Completed);
            }

            CleanSteps(jobManager, jobId);
            jobManager.MoveJobFile(jobId, JobStatus.Failed);
        }

        return false;
    }

    /// <summary>Python: <c>run_jobs</c>.</summary>
    public static bool RunJobs(JobManager jobManager, ProcessStep processStep, ConcatVideoStep concatVideo, bool haltOnError)
    {
        var queuedJobIds = jobManager.FindJobIds(JobStatus.Queued);
        var hasError = false;

        if (queuedJobIds.Count > 0)
        {
            foreach (var jobId in queuedJobIds)
            {
                if (!RunJob(jobManager, jobId, processStep, concatVideo))
                {
                    hasError = true;
                    if (haltOnError)
                    {
                        return false;
                    }
                }
            }

            return !hasError;
        }

        return false;
    }

    /// <summary>Python: <c>retry_job</c>.</summary>
    public static bool RetryJob(JobManager jobManager, string jobId, ProcessStep processStep, ConcatVideoStep concatVideo)
    {
        var failedJobIds = jobManager.FindJobIds(JobStatus.Failed);

        if (failedJobIds.Contains(jobId))
        {
            return jobManager.SetStepsStatus(jobId, JobStepStatus.Queued)
                && jobManager.MoveJobFile(jobId, JobStatus.Queued)
                && RunJob(jobManager, jobId, processStep, concatVideo);
        }

        return false;
    }

    /// <summary>Python: <c>retry_jobs</c>.</summary>
    public static bool RetryJobs(JobManager jobManager, ProcessStep processStep, ConcatVideoStep concatVideo, bool haltOnError)
    {
        var failedJobIds = jobManager.FindJobIds(JobStatus.Failed);
        var hasError = false;

        if (failedJobIds.Count > 0)
        {
            foreach (var jobId in failedJobIds)
            {
                if (!RetryJob(jobManager, jobId, processStep, concatVideo))
                {
                    hasError = true;
                    if (haltOnError)
                    {
                        return false;
                    }
                }
            }

            return !hasError;
        }

        return false;
    }

    /// <summary>Python: <c>run_step</c>.</summary>
    public static bool RunStep(JobManager jobManager, string jobId, int stepIndex, JobStep step, ProcessStep processStep)
    {
        var stepArgs = step.Args;

        if (jobManager.SetStepStatus(jobId, stepIndex, JobStepStatus.Started) && processStep(jobId, stepIndex, stepArgs))
        {
            var outputPath = stepArgs.TryGetValue("output_path", out var value) ? value as string : null;
            var stepOutputPath = JobHelper.GetStepOutputPath(jobId, stepIndex, outputPath);

            return FileSystem.MoveFile(outputPath, stepOutputPath) && jobManager.SetStepStatus(jobId, stepIndex, JobStepStatus.Completed);
        }

        jobManager.SetStepStatus(jobId, stepIndex, JobStepStatus.Failed);
        return false;
    }

    /// <summary>Python: <c>run_steps</c>.</summary>
    public static bool RunSteps(JobManager jobManager, string jobId, ProcessStep processStep)
    {
        var steps = jobManager.GetSteps(jobId);

        if (steps.Count > 0)
        {
            for (var index = 0; index < steps.Count; index++)
            {
                if (!RunStep(jobManager, jobId, index, steps[index], processStep))
                {
                    return false;
                }
            }

            return true;
        }

        return false;
    }

    /// <summary>Python: <c>finalize_steps</c>.</summary>
    public static bool FinalizeSteps(JobManager jobManager, string jobId, ConcatVideoStep concatVideo)
    {
        var outputSet = CollectOutputSet(jobManager, jobId);

        foreach (var (outputPath, tempOutputPaths) in outputSet)
        {
            if (FileSystem.AreVideos(tempOutputPaths))
            {
                if (!concatVideo(outputPath, tempOutputPaths))
                {
                    return false;
                }
            }

            if (FileSystem.AreImages(tempOutputPaths))
            {
                foreach (var tempOutputPath in tempOutputPaths)
                {
                    if (!FileSystem.MoveFile(tempOutputPath, outputPath))
                    {
                        return false;
                    }
                }
            }
        }

        return true;
    }

    /// <summary>Python: <c>clean_steps</c>.</summary>
    public static bool CleanSteps(JobManager jobManager, string jobId)
    {
        var outputSet = CollectOutputSet(jobManager, jobId);

        foreach (var tempOutputPaths in outputSet.Values)
        {
            foreach (var tempOutputPath in tempOutputPaths)
            {
                if (!FileSystem.RemoveFile(tempOutputPath))
                {
                    return false;
                }
            }
        }

        return true;
    }

    /// <summary>Python: <c>collect_output_set</c>.</summary>
    public static IReadOnlyDictionary<string, IReadOnlyList<string>> CollectOutputSet(JobManager jobManager, string jobId)
    {
        var steps = jobManager.GetSteps(jobId);
        var jobOutputSet = new Dictionary<string, List<string>>();

        for (var index = 0; index < steps.Count; index++)
        {
            var outputPath = steps[index].Args.TryGetValue("output_path", out var value) ? value as string : null;

            if (!string.IsNullOrEmpty(outputPath))
            {
                var stepOutputPath = JobHelper.GetStepOutputPath(jobId, index, outputPath);

                if (!jobOutputSet.TryGetValue(outputPath, out var tempOutputPaths))
                {
                    tempOutputPaths = new List<string>();
                    jobOutputSet[outputPath] = tempOutputPaths;
                }

                if (stepOutputPath is not null)
                {
                    tempOutputPaths.Add(stepOutputPath);
                }
            }
        }

        return jobOutputSet.ToDictionary(pair => pair.Key, pair => (IReadOnlyList<string>)pair.Value);
    }
}
