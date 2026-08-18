using FaceFusion.Jobs;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_job_list.py</c>.
/// </summary>
public sealed class JobListTests
{
    private static string GetTestJobsDirectory() => Path.Combine(Path.GetTempPath(), "facefusion-test-jobs");

    private readonly JobManager _jobManager;

    /// <summary>xUnit constructs a fresh instance per test, matching pytest's
    /// function-scoped autouse <c>before_each</c> fixture.</summary>
    public JobListTests()
    {
        _jobManager = new JobManager(GetTestJobsDirectory());
        _jobManager.ClearJobs(GetTestJobsDirectory());
        _jobManager.InitJobs();
    }

    [Fact]
    public void TestComposeJobList()
    {
        _jobManager.CreateJob("job-test-compose-job-list-1");
        Thread.Sleep(500);
        _jobManager.CreateJob("job-test-compose-job-list-2");

        var (jobHeaders, jobContents) = JobList.ComposeJobList(_jobManager, JobStatus.Drafted);

        Assert.Equal(new[] { "job id", "steps", "date created", "date updated", "job status" }, jobHeaders);
        Assert.Equal(new object?[] { "job-test-compose-job-list-1", 0, "just now", null, "drafted" }, jobContents[0]);
        Assert.Equal(new object?[] { "job-test-compose-job-list-2", 0, "just now", null, "drafted" }, jobContents[1]);
    }
}
