using System.Diagnostics;
using System.Text;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_job_manager.py</c>, plus the byte-compatibility round trip
/// required by docs/DOTNET_PORT_PLAN.md §9.3: job files written by one implementation
/// must be readable by the other.
/// </summary>
public sealed class JobManagerTests
{
    private static string GetTestJobsDirectory() => Path.Combine(Path.GetTempPath(), "facefusion-test-jobs");

    private readonly JobManager _jobManager;

    public JobManagerTests()
    {
        _jobManager = new JobManager(GetTestJobsDirectory());
        _jobManager.ClearJobs(GetTestJobsDirectory());
        _jobManager.InitJobs();
    }

    private static Dictionary<string, object?> Args(string sourcePath, string targetPath, string outputPath)
        => new()
        {
            ["source_path"] = sourcePath,
            ["target_path"] = targetPath,
            ["output_path"] = outputPath
        };

    [Fact]
    public void TestCreateJob()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");

        Assert.True(_jobManager.CreateJob("job-test-create-job"));
        Assert.False(_jobManager.CreateJob("job-test-create-job"));

        // Faithful port of a Python test quirk: this add_step/submit_job target a job
        // id ('job-test-submit-job') that was never created in this test, so both
        // calls are silent no-ops.
        _jobManager.AddStep("job-test-submit-job", args1);
        _jobManager.SubmitJob("job-test-create-job");

        Assert.False(_jobManager.CreateJob("job-test-create-job"));
    }

    [Fact]
    public void TestSubmitJob()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");

        Assert.False(_jobManager.SubmitJob("job-invalid"));

        _jobManager.CreateJob("job-test-submit-job");

        Assert.False(_jobManager.SubmitJob("job-test-submit-job"));

        _jobManager.AddStep("job-test-submit-job", args1);

        Assert.True(_jobManager.SubmitJob("job-test-submit-job"));
        Assert.False(_jobManager.SubmitJob("job-test-submit-job"));
    }

    [Fact]
    public void TestSubmitJobs()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");
        var haltOnError = true;

        Assert.False(_jobManager.SubmitJobs(haltOnError));

        _jobManager.CreateJob("job-test-submit-jobs-1");
        _jobManager.CreateJob("job-test-submit-jobs-2");

        Assert.False(_jobManager.SubmitJobs(haltOnError));

        _jobManager.AddStep("job-test-submit-jobs-1", args1);
        _jobManager.AddStep("job-test-submit-jobs-2", args2);

        Assert.True(_jobManager.SubmitJobs(haltOnError));
        Assert.False(_jobManager.SubmitJobs(haltOnError));
    }

    [Fact]
    public void TestDeleteJob()
    {
        Assert.False(_jobManager.DeleteJob("job-invalid"));

        _jobManager.CreateJob("job-test-delete-job");

        Assert.True(_jobManager.DeleteJob("job-test-delete-job"));
        Assert.False(_jobManager.DeleteJob("job-test-delete-job"));
    }

    [Fact]
    public void TestDeleteJobs()
    {
        var haltOnError = true;

        Assert.False(_jobManager.DeleteJobs(haltOnError));

        _jobManager.CreateJob("job-test-delete-jobs-1");
        _jobManager.CreateJob("job-test-delete-jobs-2");

        Assert.True(_jobManager.DeleteJobs(haltOnError));
    }

    [Fact]
    public void TestFindJobs()
    {
        _jobManager.CreateJob("job-test-find-jobs-1");
        Thread.Sleep(500);
        _jobManager.CreateJob("job-test-find-jobs-2");

        Assert.Contains("job-test-find-jobs-1", _jobManager.FindJobs(JobStatus.Drafted).Keys);
        Assert.Contains("job-test-find-jobs-2", _jobManager.FindJobs(JobStatus.Drafted).Keys);
        Assert.Empty(_jobManager.FindJobs(JobStatus.Queued));

        _jobManager.MoveJobFile("job-test-find-jobs-1", JobStatus.Queued);

        Assert.Contains("job-test-find-jobs-2", _jobManager.FindJobs(JobStatus.Drafted).Keys);
        Assert.Contains("job-test-find-jobs-1", _jobManager.FindJobs(JobStatus.Queued).Keys);
    }

    [Fact]
    public void TestFindJobIds()
    {
        _jobManager.CreateJob("job-test-find-job-ids-1");
        Thread.Sleep(500);
        _jobManager.CreateJob("job-test-find-job-ids-2");
        Thread.Sleep(500);
        _jobManager.CreateJob("job-test-find-job-ids-3");

        Assert.Equal(new[] { "job-test-find-job-ids-1", "job-test-find-job-ids-2", "job-test-find-job-ids-3" }, _jobManager.FindJobIds(JobStatus.Drafted));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Queued));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Completed));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Failed));

        _jobManager.MoveJobFile("job-test-find-job-ids-1", JobStatus.Queued);
        _jobManager.MoveJobFile("job-test-find-job-ids-2", JobStatus.Queued);
        _jobManager.MoveJobFile("job-test-find-job-ids-3", JobStatus.Queued);

        Assert.Empty(_jobManager.FindJobIds(JobStatus.Drafted));
        Assert.Equal(new[] { "job-test-find-job-ids-1", "job-test-find-job-ids-2", "job-test-find-job-ids-3" }, _jobManager.FindJobIds(JobStatus.Queued));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Completed));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Failed));

        _jobManager.MoveJobFile("job-test-find-job-ids-1", JobStatus.Completed);

        Assert.Empty(_jobManager.FindJobIds(JobStatus.Drafted));
        Assert.Equal(new[] { "job-test-find-job-ids-2", "job-test-find-job-ids-3" }, _jobManager.FindJobIds(JobStatus.Queued));
        Assert.Equal(new[] { "job-test-find-job-ids-1" }, _jobManager.FindJobIds(JobStatus.Completed));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Failed));

        _jobManager.MoveJobFile("job-test-find-job-ids-2", JobStatus.Failed);

        Assert.Empty(_jobManager.FindJobIds(JobStatus.Drafted));
        Assert.Equal(new[] { "job-test-find-job-ids-3" }, _jobManager.FindJobIds(JobStatus.Queued));
        Assert.Equal(new[] { "job-test-find-job-ids-1" }, _jobManager.FindJobIds(JobStatus.Completed));
        Assert.Equal(new[] { "job-test-find-job-ids-2" }, _jobManager.FindJobIds(JobStatus.Failed));

        _jobManager.MoveJobFile("job-test-find-job-ids-3", JobStatus.Completed);

        Assert.Empty(_jobManager.FindJobIds(JobStatus.Drafted));
        Assert.Empty(_jobManager.FindJobIds(JobStatus.Queued));
        Assert.Equal(new[] { "job-test-find-job-ids-1", "job-test-find-job-ids-3" }, _jobManager.FindJobIds(JobStatus.Completed));
        Assert.Equal(new[] { "job-test-find-job-ids-2" }, _jobManager.FindJobIds(JobStatus.Failed));
    }

    [Fact]
    public void TestAddStep()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");

        Assert.False(_jobManager.AddStep("job-invalid", args1));

        _jobManager.CreateJob("job-test-add-step");

        Assert.True(_jobManager.AddStep("job-test-add-step", args1));
        Assert.True(_jobManager.AddStep("job-test-add-step", args2));

        var steps = _jobManager.GetSteps("job-test-add-step");

        Assert.Equal(args1, steps[0].Args);
        Assert.Equal(args2, steps[1].Args);
        Assert.Equal(2, _jobManager.CountStepTotal("job-test-add-step"));
    }

    [Fact]
    public void TestRemixStep()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");

        Assert.False(_jobManager.RemixStep("job-invalid", 0, args1));

        _jobManager.CreateJob("job-test-remix-step");
        _jobManager.AddStep("job-test-remix-step", args1);
        _jobManager.AddStep("job-test-remix-step", args2);

        Assert.False(_jobManager.RemixStep("job-test-remix-step", 99, args1));
        Assert.True(_jobManager.RemixStep("job-test-remix-step", 0, args2));
        Assert.True(_jobManager.RemixStep("job-test-remix-step", -1, args2));

        var steps = _jobManager.GetSteps("job-test-remix-step");

        Assert.Equal(args1, steps[0].Args);
        Assert.Equal(args2, steps[1].Args);
        Assert.Equal(args2["source_path"], steps[2].Args["source_path"]);
        Assert.Equal(JobHelper.GetStepOutputPath("job-test-remix-step", 0, (string?)args1["output_path"]), steps[2].Args["target_path"]);
        Assert.Equal(args2["output_path"], steps[2].Args["output_path"]);
        Assert.Equal(args2["source_path"], steps[3].Args["source_path"]);
        Assert.Equal(JobHelper.GetStepOutputPath("job-test-remix-step", 2, (string?)args2["output_path"]), steps[3].Args["target_path"]);
        Assert.Equal(args2["output_path"], steps[3].Args["output_path"]);
        Assert.Equal(4, _jobManager.CountStepTotal("job-test-remix-step"));
    }

    [Fact]
    public void TestInsertStep()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");
        var args3 = Args("source-3.jpg", "target-3.jpg", "output-3.jpg");

        Assert.False(_jobManager.InsertStep("job-invalid", 0, args1));

        _jobManager.CreateJob("job-test-insert-step");
        _jobManager.AddStep("job-test-insert-step", args1);
        _jobManager.AddStep("job-test-insert-step", args1);

        Assert.False(_jobManager.InsertStep("job-test-insert-step", 99, args1));
        Assert.True(_jobManager.InsertStep("job-test-insert-step", 0, args2));
        Assert.True(_jobManager.InsertStep("job-test-insert-step", -1, args3));

        var steps = _jobManager.GetSteps("job-test-insert-step");

        Assert.Equal(args2, steps[0].Args);
        Assert.Equal(args1, steps[1].Args);
        Assert.Equal(args3, steps[2].Args);
        Assert.Equal(args1, steps[3].Args);
        Assert.Equal(4, _jobManager.CountStepTotal("job-test-insert-step"));
    }

    [Fact]
    public void TestRemoveStep()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");
        var args3 = Args("source-3.jpg", "target-3.jpg", "output-3.jpg");

        Assert.False(_jobManager.RemoveStep("job-invalid", 0));

        _jobManager.CreateJob("job-test-remove-step");
        _jobManager.AddStep("job-test-remove-step", args1);
        _jobManager.AddStep("job-test-remove-step", args2);
        _jobManager.AddStep("job-test-remove-step", args1);
        _jobManager.AddStep("job-test-remove-step", args3);

        Assert.False(_jobManager.RemoveStep("job-test-remove-step", 99));
        Assert.True(_jobManager.RemoveStep("job-test-remove-step", 0));
        Assert.True(_jobManager.RemoveStep("job-test-remove-step", -1));

        var steps = _jobManager.GetSteps("job-test-remove-step");

        Assert.Equal(args2, steps[0].Args);
        Assert.Equal(args1, steps[1].Args);
        Assert.Equal(2, _jobManager.CountStepTotal("job-test-remove-step"));
    }

    [Fact]
    public void TestGetSteps()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");

        Assert.Empty(_jobManager.GetSteps("job-invalid"));

        _jobManager.CreateJob("job-test-get-steps");
        _jobManager.AddStep("job-test-get-steps", args1);
        _jobManager.AddStep("job-test-get-steps", args2);

        var steps = _jobManager.GetSteps("job-test-get-steps");

        Assert.Equal(args1, steps[0].Args);
        Assert.Equal(args2, steps[1].Args);
        Assert.Equal(2, _jobManager.CountStepTotal("job-test-get-steps"));
    }

    [Fact]
    public void TestSetStepStatus()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");

        Assert.False(_jobManager.SetStepStatus("job-invalid", 0, JobStepStatus.Completed));

        _jobManager.CreateJob("job-test-set-step-status");
        _jobManager.AddStep("job-test-set-step-status", args1);
        _jobManager.AddStep("job-test-set-step-status", args2);

        Assert.False(_jobManager.SetStepStatus("job-test-set-step-status", 99, JobStepStatus.Completed));
        Assert.True(_jobManager.SetStepStatus("job-test-set-step-status", 0, JobStepStatus.Completed));
        Assert.True(_jobManager.SetStepStatus("job-test-set-step-status", 1, JobStepStatus.Failed));

        var steps = _jobManager.GetSteps("job-test-set-step-status");

        Assert.Equal(JobStepStatus.Completed, steps[0].Status);
        Assert.Equal(JobStepStatus.Failed, steps[1].Status);
        Assert.Equal(2, _jobManager.CountStepTotal("job-test-set-step-status"));
    }

    [Fact]
    public void TestSetStepsStatus()
    {
        var args1 = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");
        var args2 = Args("source-2.jpg", "target-2.jpg", "output-2.jpg");

        Assert.False(_jobManager.SetStepsStatus("job-invalid", JobStepStatus.Queued));

        _jobManager.CreateJob("job-test-set-steps-status");
        _jobManager.AddStep("job-test-set-steps-status", args1);
        _jobManager.AddStep("job-test-set-steps-status", args2);

        Assert.True(_jobManager.SetStepsStatus("job-test-set-steps-status", JobStepStatus.Queued));

        var steps = _jobManager.GetSteps("job-test-set-steps-status");

        Assert.Equal(JobStepStatus.Queued, steps[0].Status);
        Assert.Equal(JobStepStatus.Queued, steps[1].Status);
        Assert.Equal(2, _jobManager.CountStepTotal("job-test-set-steps-status"));
    }

    // -----------------------------------------------------------------
    // §9.3 byte-compatibility round trip: job files written by one
    // implementation must be readable by the other.
    // -----------------------------------------------------------------

    private static bool PythonAvailable
    {
        get
        {
            try
            {
                var psi = new ProcessStartInfo("python3", "-c \"import facefusion.jobs.job_manager\"")
                {
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    WorkingDirectory = FindRepoRoot()
                };
                using var process = Process.Start(psi);
                process!.WaitForExit(10000);
                return process.ExitCode == 0;
            }
            catch
            {
                return false;
            }
        }
    }

    private static string FindRepoRoot()
    {
        var directory = new DirectoryInfo(System.AppContext.BaseDirectory);

        while (directory is not null && !File.Exists(Path.Combine(directory.FullName, "facefusion", "jobs", "job_manager.py")))
        {
            directory = directory.Parent;
        }

        return directory?.FullName ?? System.AppContext.BaseDirectory;
    }

    private static int RunPython(string script, string workingDirectory, out string stdout, out string stderr)
    {
        var psi = new ProcessStartInfo("python3", "-c " + EscapeShellArg(script))
        {
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            WorkingDirectory = workingDirectory
        };

        using var process = Process.Start(psi)!;
        stdout = process.StandardOutput.ReadToEnd();
        stderr = process.StandardError.ReadToEnd();
        process.WaitForExit(20000);
        return process.ExitCode;
    }

    private static string EscapeShellArg(string script)
        => "\"" + script.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("$", "\\$") + "\"";

    [Fact]
    public void CSharpWrittenJob_IsReadableByPython()
    {
        if (!PythonAvailable)
        {
            return; // environment without a facefusion-importable python3; nothing to check.
        }

        var jobsDirectory = Path.Combine(Path.GetTempPath(), "facefusion-jobs-roundtrip-cs-to-py-" + Guid.NewGuid().ToString("N"));
        var repoRoot = FindRepoRoot();

        try
        {
            var jobManager = new JobManager(jobsDirectory);
            jobManager.InitJobs();

            var args = Args("source-1.jpg", "target-1.jpg", "output-1.jpg");

            Assert.True(jobManager.CreateJob("job-roundtrip"));
            Assert.True(jobManager.AddStep("job-roundtrip", args));
            Assert.True(jobManager.SubmitJob("job-roundtrip"));

            var jobPath = jobManager.FindJobPath("job-roundtrip");
            Assert.NotNull(jobPath);

            var script =
                "from facefusion.jobs import job_manager\n" +
                $"job_manager.init_jobs({PyStr(jobsDirectory)})\n" +
                "assert job_manager.validate_job('job-roundtrip'), 'python could not validate the C#-written job'\n" +
                "steps = job_manager.get_steps('job-roundtrip')\n" +
                "assert len(steps) == 1, steps\n" +
                "args = steps[0].get('args')\n" +
                "assert args.get('source_path') == 'source-1.jpg', args\n" +
                "assert args.get('target_path') == 'target-1.jpg', args\n" +
                "assert args.get('output_path') == 'output-1.jpg', args\n" +
                "assert steps[0].get('status') == 'queued', steps[0]\n" +
                "job_ids = job_manager.find_job_ids('queued')\n" +
                "assert job_ids == ['job-roundtrip'], job_ids\n" +
                "print('OK')\n";

            var exitCode = RunPython(script, repoRoot, out var stdout, out var stderr);

            Assert.True(exitCode == 0 && stdout.Contains("OK", StringComparison.Ordinal), $"python round trip failed.\nstdout: {stdout}\nstderr: {stderr}");
        }
        finally
        {
            FileSystem.RemoveDirectory(jobsDirectory);
        }
    }

    [Fact]
    public void PythonWrittenJob_IsReadableByCSharp()
    {
        if (!PythonAvailable)
        {
            return;
        }

        var jobsDirectory = Path.Combine(Path.GetTempPath(), "facefusion-jobs-roundtrip-py-to-cs-" + Guid.NewGuid().ToString("N"));
        var repoRoot = FindRepoRoot();

        try
        {
            var script =
                "from facefusion.jobs import job_manager\n" +
                $"job_manager.init_jobs({PyStr(jobsDirectory)})\n" +
                "job_manager.create_job('job-roundtrip')\n" +
                "job_manager.add_step('job-roundtrip', { 'source_path': 'source-1.jpg', 'target_path': 'target-1.jpg', 'output_path': 'output-1.jpg' })\n" +
                "job_manager.submit_job('job-roundtrip')\n" +
                "print('OK')\n";

            var exitCode = RunPython(script, repoRoot, out var stdout, out var stderr);
            Assert.True(exitCode == 0 && stdout.Contains("OK", StringComparison.Ordinal), $"python setup failed.\nstdout: {stdout}\nstderr: {stderr}");

            var jobManager = new JobManager(jobsDirectory);

            Assert.True(jobManager.ValidateJob("job-roundtrip"));

            var jobIds = jobManager.FindJobIds(JobStatus.Queued);
            Assert.Equal(new[] { "job-roundtrip" }, jobIds);

            var steps = jobManager.GetSteps("job-roundtrip");
            Assert.Single(steps);
            Assert.Equal("source-1.jpg", steps[0].Args["source_path"]);
            Assert.Equal("target-1.jpg", steps[0].Args["target_path"]);
            Assert.Equal("output-1.jpg", steps[0].Args["output_path"]);
            Assert.Equal(JobStepStatus.Queued, steps[0].Status);

            // Now write back through the C# implementation and confirm Python still reads it.
            Assert.True(jobManager.SetStepStatus("job-roundtrip", 0, JobStepStatus.Started));

            var verifyScript =
                "from facefusion.jobs import job_manager\n" +
                $"job_manager.init_jobs({PyStr(jobsDirectory)})\n" +
                "steps = job_manager.get_steps('job-roundtrip')\n" +
                "assert steps[0].get('status') == 'started', steps[0]\n" +
                "print('OK')\n";

            var verifyExitCode = RunPython(verifyScript, repoRoot, out var verifyStdout, out var verifyStderr);
            Assert.True(verifyExitCode == 0 && verifyStdout.Contains("OK", StringComparison.Ordinal), $"python re-read of the C#-updated job failed.\nstdout: {verifyStdout}\nstderr: {verifyStderr}");
        }
        finally
        {
            FileSystem.RemoveDirectory(jobsDirectory);
        }
    }

    private static string PyStr(string value)
    {
        var builder = new StringBuilder();
        builder.Append('\'');
        foreach (var c in value)
        {
            if (c == '\'' || c == '\\')
            {
                builder.Append('\\');
            }
            builder.Append(c);
        }
        builder.Append('\'');
        return builder.ToString();
    }
}
