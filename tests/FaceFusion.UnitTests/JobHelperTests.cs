using FaceFusion.Jobs;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_job_helper.py</c>.
/// </summary>
public sealed class JobHelperTests
{
    [Fact]
    public void TestGetStepOutputPath()
    {
        Assert.Equal("test-test-job-0.mp4", JobHelper.GetStepOutputPath("test-job", 0, "test.mp4"));
        Assert.Equal(Path.Combine("test", "test-test-job-0.mp4"), JobHelper.GetStepOutputPath("test-job", 0, "test/test.mp4"));
        Assert.Null(JobHelper.GetStepOutputPath("test-job", 0, "invalid"));
    }
}
