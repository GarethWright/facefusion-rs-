using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Media;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_job_runner.py</c>.
/// </summary>
public sealed class JobRunnerTests
{
    private static string GetTestJobsDirectory() => Path.Combine(Path.GetTempPath(), "facefusion-test-jobs");

    private readonly JobManager _jobManager;

    public JobRunnerTests()
    {
        _jobManager = new JobManager(GetTestJobsDirectory());
        _jobManager.ClearJobs(GetTestJobsDirectory());
        _jobManager.InitJobs();
        TestHelper.PrepareTestOutputDirectory();
    }

    private static Dictionary<string, object?> Args(string sourcePath, string targetPath, string outputPath)
        => new()
        {
            ["source_path"] = sourcePath,
            ["target_path"] = targetPath,
            ["output_path"] = outputPath
        };

    /// <summary>Python: <c>process_step</c> test fixture.</summary>
    private static bool ProcessStep(string jobId, int stepIndex, IReadOnlyDictionary<string, object?> stepArgs)
        => FileSystem.CopyFile(stepArgs.TryGetValue("target_path", out var target) ? target as string : null,
                                stepArgs.TryGetValue("output_path", out var output) ? output as string : null);

    private static bool ConcatVideo(string outputPath, IReadOnlyList<string> tempOutputPaths)
        => Ffmpeg.ConcatVideo(outputPath, tempOutputPaths);

    // -----------------------------------------------------------------
    // Runtime skip: these tests need the downloaded example media
    // (target-240p.mp4 / target-240p.jpg / source.jpg) and a real ffmpeg on
    // PATH, neither of which is guaranteed in this environment. Per
    // PORT_CONVENTIONS.md rule 2 they are ported rather than dropped, and
    // skip with a clear reason instead of failing.
    // -----------------------------------------------------------------

    [AttributeUsage(AttributeTargets.Method)]
    private sealed class MediaFactAttribute : FactAttribute
    {
        public MediaFactAttribute()
        {
            if (!TestHelper.ExamplesAvailable)
            {
                Skip = TestHelper.MissingMediaMessage;
            }
            else if (!TestHelper.HasFfmpeg || !TestHelper.HasFfprobe)
            {
                Skip = TestHelper.MissingMediaMessage;
            }
            else if (!File.Exists(TestHelper.GetTestExampleFile("target-240p.jpg")))
            {
                Skip = "requires target-240p.jpg (a single extracted frame of target-240p.mp4) " +
                       "in the example media directory — the Python suite's before_all fixture " +
                       "extracts it with ffmpeg -vframes 1";
            }
        }
    }

    [MediaFact]
    public void TestRunJob()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));

        Assert.False(JobRunner.RunJob(_jobManager, "job-invalid", ProcessStep, ConcatVideo));

        _jobManager.CreateJob("job-test-run-job");
        _jobManager.AddStep("job-test-run-job", args1);
        _jobManager.AddStep("job-test-run-job", args2);
        _jobManager.AddStep("job-test-run-job", args2);
        _jobManager.AddStep("job-test-run-job", args3);

        Assert.False(JobRunner.RunJob(_jobManager, "job-test-run-job", ProcessStep, ConcatVideo));

        _jobManager.SubmitJob("job-test-run-job");

        Assert.True(JobRunner.RunJob(_jobManager, "job-test-run-job", ProcessStep, ConcatVideo));
    }

    [MediaFact]
    public void TestRunJobs()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));
        var haltOnError = true;

        Assert.False(JobRunner.RunJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));

        _jobManager.CreateJob("job-test-run-jobs-1");
        _jobManager.CreateJob("job-test-run-jobs-2");
        _jobManager.AddStep("job-test-run-jobs-1", args1);
        _jobManager.AddStep("job-test-run-jobs-1", args1);
        _jobManager.AddStep("job-test-run-jobs-2", args2);
        // Faithful port of a Python test quirk: this add_step targets a job id
        // ('job-test-run-jobs-3') that was never created, so it is a silent no-op.
        _jobManager.AddStep("job-test-run-jobs-3", args3);

        Assert.False(JobRunner.RunJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));

        _jobManager.SubmitJobs(haltOnError);

        Assert.True(JobRunner.RunJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));
    }

    [MediaFact]
    public void TestRetryJob()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));

        Assert.False(JobRunner.RetryJob(_jobManager, "job-invalid", ProcessStep, ConcatVideo));

        _jobManager.CreateJob("job-test-retry-job");
        _jobManager.AddStep("job-test-retry-job", args1);
        _jobManager.SubmitJob("job-test-retry-job");

        Assert.False(JobRunner.RetryJob(_jobManager, "job-test-retry-job", ProcessStep, ConcatVideo));

        _jobManager.MoveJobFile("job-test-retry-job", JobStatus.Failed);

        Assert.True(JobRunner.RetryJob(_jobManager, "job-test-retry-job", ProcessStep, ConcatVideo));
    }

    [MediaFact]
    public void TestRetryJobs()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));
        var haltOnError = true;

        Assert.False(JobRunner.RetryJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));

        _jobManager.CreateJob("job-test-retry-jobs-1");
        _jobManager.CreateJob("job-test-retry-jobs-2");
        _jobManager.AddStep("job-test-retry-jobs-1", args1);
        _jobManager.AddStep("job-test-retry-jobs-1", args1);
        _jobManager.AddStep("job-test-retry-jobs-2", args2);
        _jobManager.AddStep("job-test-retry-jobs-3", args3);

        Assert.False(JobRunner.RetryJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));

        _jobManager.MoveJobFile("job-test-retry-jobs-1", JobStatus.Failed);
        _jobManager.MoveJobFile("job-test-retry-jobs-2", JobStatus.Failed);

        Assert.True(JobRunner.RetryJobs(_jobManager, ProcessStep, ConcatVideo, haltOnError));
    }

    [MediaFact]
    public void TestRunSteps()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));

        Assert.False(JobRunner.RunSteps(_jobManager, "job-invalid", ProcessStep));

        _jobManager.CreateJob("job-test-run-steps");
        _jobManager.AddStep("job-test-run-steps", args1);
        _jobManager.AddStep("job-test-run-steps", args1);
        _jobManager.AddStep("job-test-run-steps", args2);
        _jobManager.AddStep("job-test-run-steps", args3);

        Assert.True(JobRunner.RunSteps(_jobManager, "job-test-run-steps", ProcessStep));
    }

    [MediaFact]
    public void TestFinalizeSteps()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));

        _jobManager.CreateJob("job-test-finalize-steps");
        _jobManager.AddStep("job-test-finalize-steps", args1);
        _jobManager.AddStep("job-test-finalize-steps", args1);
        _jobManager.AddStep("job-test-finalize-steps", args2);
        _jobManager.AddStep("job-test-finalize-steps", args3);

        FileSystem.CopyFile((string?)args1["target_path"], TestHelper.GetTestOutputFile("output-1-job-test-finalize-steps-0.mp4"));
        FileSystem.CopyFile((string?)args1["target_path"], TestHelper.GetTestOutputFile("output-1-job-test-finalize-steps-1.mp4"));
        FileSystem.CopyFile((string?)args2["target_path"], TestHelper.GetTestOutputFile("output-2-job-test-finalize-steps-2.mp4"));
        FileSystem.CopyFile((string?)args3["target_path"], TestHelper.GetTestOutputFile("output-3-job-test-finalize-steps-3.jpg"));

        Assert.True(JobRunner.FinalizeSteps(_jobManager, "job-test-finalize-steps", ConcatVideo));
        Assert.True(File.Exists(TestHelper.GetTestOutputFile("output-1.mp4")));
        Assert.True(File.Exists(TestHelper.GetTestOutputFile("output-2.mp4")));
        Assert.True(File.Exists(TestHelper.GetTestOutputFile("output-3.jpg")));
    }

    [Fact]
    public void TestCollectOutputSet()
    {
        var args1 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-1.mp4"));
        var args2 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.mp4"), TestHelper.GetTestOutputFile("output-2.mp4"));
        var args3 = Args(TestHelper.GetTestExampleFile("source.jpg"), TestHelper.GetTestExampleFile("target-240p.jpg"), TestHelper.GetTestOutputFile("output-3.jpg"));

        _jobManager.CreateJob("job-test-collect-output-set");
        _jobManager.AddStep("job-test-collect-output-set", args1);
        _jobManager.AddStep("job-test-collect-output-set", args1);
        _jobManager.AddStep("job-test-collect-output-set", args2);
        _jobManager.AddStep("job-test-collect-output-set", args3);

        var expected = new Dictionary<string, IReadOnlyList<string>>
        {
            [TestHelper.GetTestOutputFile("output-1.mp4")] = new[]
            {
                TestHelper.GetTestOutputFile("output-1-job-test-collect-output-set-0.mp4"),
                TestHelper.GetTestOutputFile("output-1-job-test-collect-output-set-1.mp4")
            },
            [TestHelper.GetTestOutputFile("output-2.mp4")] = new[]
            {
                TestHelper.GetTestOutputFile("output-2-job-test-collect-output-set-2.mp4")
            },
            [TestHelper.GetTestOutputFile("output-3.jpg")] = new[]
            {
                TestHelper.GetTestOutputFile("output-3-job-test-collect-output-set-3.jpg")
            }
        };

        var outputSet = JobRunner.CollectOutputSet(_jobManager, "job-test-collect-output-set");

        Assert.Equal(expected.Keys.OrderBy(k => k), outputSet.Keys.OrderBy(k => k));
        foreach (var key in expected.Keys)
        {
            Assert.Equal(expected[key], outputSet[key]);
        }
    }
}
