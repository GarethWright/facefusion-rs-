using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.Jobs;

/// <summary>
/// Port of <c>facefusion/jobs/job_list.py</c>.
/// </summary>
public static class JobList
{
    /// <summary>Python: <c>compose_job_list</c>.</summary>
    public static (IReadOnlyList<string> Headers, IReadOnlyList<IReadOnlyList<object?>> Contents) ComposeJobList(JobManager jobManager, JobStatus jobStatus)
    {
        var jobs = jobManager.FindJobs(jobStatus);
        IReadOnlyList<string> jobHeaders = new[] { "job id", "steps", "date created", "date updated", "job status" };
        var jobContents = new List<IReadOnlyList<object?>>();

        foreach (var jobId in jobs.Keys)
        {
            if (jobManager.ValidateJob(jobId))
            {
                var job = jobs[jobId];
                var stepTotal = jobManager.CountStepTotal(jobId);
                var dateCreated = PrepareDescribeDateTime(job.DateCreated);
                var dateUpdated = PrepareDescribeDateTime(job.DateUpdated);

                jobContents.Add(new object?[] { jobId, stepTotal, dateCreated, dateUpdated, jobStatus.ToWireName() });
            }
        }

        return (jobHeaders, jobContents);
    }

    /// <summary>Python: <c>prepare_describe_datetime</c>.</summary>
    public static string? PrepareDescribeDateTime(string? dateTime)
    {
        if (!string.IsNullOrEmpty(dateTime) && DateTimeOffset.TryParse(dateTime, System.Globalization.CultureInfo.InvariantCulture, System.Globalization.DateTimeStyles.None, out var parsed))
        {
            return TimeHelper.DescribeTimeAgo(parsed.LocalDateTime);
        }

        return null;
    }
}
