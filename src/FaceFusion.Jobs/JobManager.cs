using System.Globalization;
using System.Text.Json;
using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.Jobs;

/// <summary>
/// Port of <c>facefusion/jobs/job_manager.py</c>.
///
/// Deviation from Python: the Python module keeps <c>JOBS_PATH</c> as a module-level
/// global set by <c>init_jobs</c>. Per port convention rule 5 (no global mutable state)
/// and consistent with how <c>ProcessManager</c>, <c>Logger</c>, and
/// <c>InferenceManager</c> were ported, this is an instance class that takes the jobs
/// path in its constructor instead. <c>clear_jobs</c> already took an explicit
/// <c>jobs_path</c> parameter in Python (it never read the global), so its signature is
/// unchanged here.
///
/// Job JSON must stay byte-compatible with the Python implementation (see
/// docs/DOTNET_PORT_PLAN.md §9.3): job files are read and written by both sides during
/// the transition. <see cref="Job"/>/<see cref="JobStep"/> use PascalCase C# property
/// names that do not match the snake_case keys on the wire, so this class never
/// serializes those records directly — it builds/reads plain
/// <c>Dictionary&lt;string, object?&gt;</c> graphs (matching the pattern already pinned
/// by JsonPythonParityTests) and converts them to/from the typed records at the API
/// boundary, using <see cref="FaceFusion.Core.Json"/> for all I/O.
/// </summary>
public sealed class JobManager
{
    private readonly string _jobsPath;

    public JobManager(string jobsPath)
    {
        _jobsPath = jobsPath;
    }

    /// <summary>Python: <c>init_jobs</c>.</summary>
    public bool InitJobs()
    {
        var jobStatusPaths = Enum.GetValues<JobStatus>()
            .Select(jobStatus => Path.Combine(_jobsPath, jobStatus.ToWireName()))
            .ToList();

        foreach (var jobStatusPath in jobStatusPaths)
        {
            FileSystem.CreateDirectory(jobStatusPath);
        }

        return jobStatusPaths.All(FileSystem.IsDirectory);
    }

    /// <summary>Python: <c>clear_jobs</c>. Takes an explicit path, matching Python.</summary>
    public bool ClearJobs(string jobsPath) => FileSystem.RemoveDirectory(jobsPath);

    /// <summary>Python: <c>create_job</c>.</summary>
    public bool CreateJob(string jobId)
    {
        var job = new Job("1", GetCurrentDateTimeIso(), null, Array.Empty<JobStep>());

        return CreateJobFile(jobId, job);
    }

    /// <summary>Python: <c>submit_job</c>.</summary>
    public bool SubmitJob(string jobId)
    {
        var draftedJobIds = FindJobIds(JobStatus.Drafted);
        var steps = GetSteps(jobId);

        if (draftedJobIds.Contains(jobId) && steps.Count > 0)
        {
            return SetStepsStatus(jobId, JobStepStatus.Queued) && MoveJobFile(jobId, JobStatus.Queued);
        }

        return false;
    }

    /// <summary>Python: <c>submit_jobs</c>.</summary>
    public bool SubmitJobs(bool haltOnError)
    {
        var draftedJobIds = FindJobIds(JobStatus.Drafted);
        var hasError = false;

        if (draftedJobIds.Count > 0)
        {
            foreach (var jobId in draftedJobIds)
            {
                if (!SubmitJob(jobId))
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

    /// <summary>Python: <c>delete_job</c>.</summary>
    public bool DeleteJob(string jobId) => DeleteJobFile(jobId);

    /// <summary>Python: <c>delete_jobs</c>.</summary>
    public bool DeleteJobs(bool haltOnError)
    {
        var jobIds = FindJobIds(JobStatus.Drafted)
            .Concat(FindJobIds(JobStatus.Queued))
            .Concat(FindJobIds(JobStatus.Failed))
            .Concat(FindJobIds(JobStatus.Completed))
            .ToList();
        var hasError = false;

        if (jobIds.Count > 0)
        {
            foreach (var jobId in jobIds)
            {
                if (!DeleteJob(jobId))
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

    /// <summary>Python: <c>find_jobs</c>.</summary>
    public IReadOnlyDictionary<string, Job> FindJobs(JobStatus jobStatus)
    {
        var jobIds = FindJobIds(jobStatus);
        var jobSet = new Dictionary<string, Job>();

        foreach (var jobId in jobIds)
        {
            var job = ReadJobFile(jobId);

            // Python assigns job_set[job_id] = read_job_file(job_id) unconditionally,
            // even when the read failed (None). A dict value here must be a Job, so a
            // failed read (job file vanished between the listing and the read) is
            // dropped instead of stored as null; this cannot happen in the tested
            // flows, which never race the filesystem.
            if (job is not null)
            {
                jobSet[jobId] = job;
            }
        }

        return jobSet;
    }

    /// <summary>Python: <c>find_job_ids</c>.</summary>
    public IReadOnlyList<string> FindJobIds(JobStatus jobStatus)
    {
        var jobPattern = Path.Combine(_jobsPath, jobStatus.ToWireName(), "*.json");
        // Python: job_paths.sort(key = os.path.getmtime) — a stable sort by
        // modification time over the (already alphabetically sorted) glob result.
        // OrderBy is a stable sort in .NET, matching Python's list.sort/sorted.
        var jobPaths = FileSystem.ResolveFilePattern(jobPattern)
            .OrderBy(File.GetLastWriteTimeUtc)
            .ToList();
        var jobIds = new List<string>();

        foreach (var jobPath in jobPaths)
        {
            var jobId = FileSystem.GetFileName(jobPath);

            if (jobId is not null)
            {
                jobIds.Add(jobId);
            }
        }

        return jobIds;
    }

    /// <summary>Python: <c>validate_job</c>.</summary>
    public bool ValidateJob(string jobId)
    {
        var jobPath = FindJobPath(jobId);
        var element = Json.ReadJson(jobPath);

        return element is { ValueKind: JsonValueKind.Object } job
            && job.TryGetProperty("version", out _)
            && job.TryGetProperty("date_created", out _)
            && job.TryGetProperty("date_updated", out _)
            && job.TryGetProperty("steps", out _);
    }

    /// <summary>Python: <c>has_step</c>.</summary>
    public bool HasStep(string jobId, int stepIndex)
    {
        var stepTotal = CountStepTotal(jobId);

        // Python: step_index in range(step_total) — true for 0 <= step_index < step_total.
        return stepIndex >= 0 && stepIndex < stepTotal;
    }

    /// <summary>Python: <c>add_step</c>.</summary>
    public bool AddStep(string jobId, IReadOnlyDictionary<string, object?> stepArgs)
    {
        var job = ReadJobFile(jobId);

        if (job is not null)
        {
            var newSteps = job.Steps.Append(new JobStep(stepArgs, JobStepStatus.Drafted)).ToList();
            var updatedJob = job with { Steps = newSteps };

            return UpdateJobFile(jobId, updatedJob);
        }

        return false;
    }

    /// <summary>Python: <c>remix_step</c>.</summary>
    public bool RemixStep(string jobId, int stepIndex, IReadOnlyDictionary<string, object?> stepArgs)
    {
        var steps = GetSteps(jobId);
        var args = new Dictionary<string, object?>(stepArgs);

        // Python: `if step_index and step_index < 0:` — step_index == 0 can never be
        // < 0, so the `step_index and` guard is a no-op here; kept as a plain
        // negative check.
        if (stepIndex < 0)
        {
            stepIndex = CountStepTotal(jobId) - 1;
        }

        if (HasStep(jobId, stepIndex))
        {
            var outputPath = steps[stepIndex].Args.TryGetValue("output_path", out var value) ? value as string : null;

            args["target_path"] = JobHelper.GetStepOutputPath(jobId, stepIndex, outputPath);

            return AddStep(jobId, args);
        }

        return false;
    }

    /// <summary>Python: <c>insert_step</c>.</summary>
    public bool InsertStep(string jobId, int stepIndex, IReadOnlyDictionary<string, object?> stepArgs)
    {
        var job = ReadJobFile(jobId);
        var args = new Dictionary<string, object?>(stepArgs);

        if (stepIndex < 0)
        {
            stepIndex = CountStepTotal(jobId) - 1;
        }

        if (job is not null && HasStep(jobId, stepIndex))
        {
            var newSteps = new List<JobStep>(job.Steps);
            newSteps.Insert(stepIndex, new JobStep(args, JobStepStatus.Drafted));
            var updatedJob = job with { Steps = newSteps };

            return UpdateJobFile(jobId, updatedJob);
        }

        return false;
    }

    /// <summary>Python: <c>remove_step</c>.</summary>
    public bool RemoveStep(string jobId, int stepIndex)
    {
        var job = ReadJobFile(jobId);

        if (stepIndex < 0)
        {
            stepIndex = CountStepTotal(jobId) - 1;
        }

        if (job is not null && HasStep(jobId, stepIndex))
        {
            var newSteps = new List<JobStep>(job.Steps);
            newSteps.RemoveAt(stepIndex);
            var updatedJob = job with { Steps = newSteps };

            return UpdateJobFile(jobId, updatedJob);
        }

        return false;
    }

    /// <summary>Python: <c>get_steps</c>.</summary>
    public IReadOnlyList<JobStep> GetSteps(string jobId)
    {
        var job = ReadJobFile(jobId);

        return job?.Steps ?? Array.Empty<JobStep>();
    }

    /// <summary>Python: <c>count_step_total</c>.</summary>
    public int CountStepTotal(string jobId) => GetSteps(jobId).Count;

    /// <summary>Python: <c>set_step_status</c>.</summary>
    public bool SetStepStatus(string jobId, int stepIndex, JobStepStatus stepStatus)
    {
        var job = ReadJobFile(jobId);

        if (job is not null && HasStep(jobId, stepIndex))
        {
            var newSteps = new List<JobStep>(job.Steps);
            newSteps[stepIndex] = newSteps[stepIndex] with { Status = stepStatus };
            var updatedJob = job with { Steps = newSteps };

            return UpdateJobFile(jobId, updatedJob);
        }

        return false;
    }

    /// <summary>Python: <c>set_steps_status</c>.</summary>
    public bool SetStepsStatus(string jobId, JobStepStatus stepStatus)
    {
        var job = ReadJobFile(jobId);

        if (job is not null)
        {
            var newSteps = job.Steps.Select(step => step with { Status = stepStatus }).ToList();
            var updatedJob = job with { Steps = newSteps };

            return UpdateJobFile(jobId, updatedJob);
        }

        return false;
    }

    /// <summary>Python: <c>read_job_file</c>.</summary>
    public Job? ReadJobFile(string jobId)
    {
        var jobPath = FindJobPath(jobId);
        var element = Json.ReadJson(jobPath);

        return JobFromJsonElement(element);
    }

    /// <summary>Python: <c>create_job_file</c>.</summary>
    public bool CreateJobFile(string jobId, Job job)
    {
        var jobPath = FindJobPath(jobId);

        if (!FileSystem.IsFile(jobPath))
        {
            var jobCreatePath = SuggestJobPath(jobId, JobStatus.Drafted);

            return jobCreatePath is not null && Json.WriteJson(jobCreatePath, JobToJsonObject(job));
        }

        return false;
    }

    /// <summary>Python: <c>update_job_file</c>.</summary>
    public bool UpdateJobFile(string jobId, Job job)
    {
        var jobPath = FindJobPath(jobId);

        if (FileSystem.IsFile(jobPath))
        {
            var updatedJob = job with { DateUpdated = GetCurrentDateTimeIso() };

            return Json.WriteJson(jobPath!, JobToJsonObject(updatedJob));
        }

        return false;
    }

    /// <summary>Python: <c>move_job_file</c>.</summary>
    public bool MoveJobFile(string jobId, JobStatus jobStatus)
    {
        var jobPath = FindJobPath(jobId);
        var jobMovePath = SuggestJobPath(jobId, jobStatus);

        return FileSystem.MoveFile(jobPath, jobMovePath);
    }

    /// <summary>Python: <c>delete_job_file</c>.</summary>
    public bool DeleteJobFile(string jobId) => FileSystem.RemoveFile(FindJobPath(jobId));

    /// <summary>Python: <c>suggest_job_path</c>.</summary>
    public string? SuggestJobPath(string jobId, JobStatus jobStatus)
    {
        var jobFileName = GetJobFileName(jobId);

        return jobFileName is not null ? Path.Combine(_jobsPath, jobStatus.ToWireName(), jobFileName) : null;
    }

    /// <summary>Python: <c>find_job_path</c>.</summary>
    public string? FindJobPath(string jobId)
    {
        var jobFileName = GetJobFileName(jobId);

        if (jobFileName is not null)
        {
            foreach (var jobStatus in Enum.GetValues<JobStatus>())
            {
                var jobPattern = Path.Combine(_jobsPath, jobStatus.ToWireName(), jobFileName);
                var jobPaths = FileSystem.ResolveFilePattern(jobPattern);

                foreach (var jobPath in jobPaths)
                {
                    return jobPath;
                }
            }
        }

        return null;
    }

    /// <summary>Python: <c>get_job_file_name</c>.</summary>
    public string? GetJobFileName(string jobId)
    {
        if (!string.IsNullOrEmpty(jobId))
        {
            var sanitizedJobId = Sanitizer.SanitizeJobId(jobId);

            return sanitizedJobId + ".json";
        }

        return null;
    }

    /// <summary>
    /// Python: <c>get_current_date_time().isoformat()</c> as used by
    /// <c>create_job</c>/<c>update_job_file</c>. <c>get_current_date_time</c> is
    /// <c>datetime.now().astimezone()</c> — a timezone-aware local time; the .NET
    /// equivalent is a <see cref="DateTimeOffset"/> built from <c>DateTime.Now</c> (the
    /// local offset is attached automatically), formatted with microsecond precision
    /// and an explicit zone offset to match Python's <c>isoformat()</c> output shape
    /// exactly (e.g. <c>2026-08-18T12:00:00.123456-07:00</c>).
    /// </summary>
    private static string GetCurrentDateTimeIso()
        => new DateTimeOffset(TimeHelper.GetCurrentDateTime()).ToString("yyyy-MM-ddTHH:mm:ss.ffffffzzz", CultureInfo.InvariantCulture);

    private static Dictionary<string, object?> JobToJsonObject(Job job)
        => new()
        {
            ["version"] = job.Version,
            ["date_created"] = job.DateCreated,
            ["date_updated"] = job.DateUpdated,
            ["steps"] = job.Steps.Select(JobStepToJsonObject).ToList()
        };

    private static Dictionary<string, object?> JobStepToJsonObject(JobStep step)
        => new()
        {
            ["args"] = step.Args,
            ["status"] = step.Status.ToWireName()
        };

    private static Job? JobFromJsonElement(JsonElement? elementOpt)
    {
        if (elementOpt is not { ValueKind: JsonValueKind.Object } element)
        {
            return null;
        }

        var version = element.TryGetProperty("version", out var versionElement) && versionElement.ValueKind == JsonValueKind.String
            ? versionElement.GetString()!
            : string.Empty;
        var dateCreated = element.TryGetProperty("date_created", out var dateCreatedElement) && dateCreatedElement.ValueKind == JsonValueKind.String
            ? dateCreatedElement.GetString()!
            : string.Empty;
        var dateUpdated = element.TryGetProperty("date_updated", out var dateUpdatedElement) && dateUpdatedElement.ValueKind == JsonValueKind.String
            ? dateUpdatedElement.GetString()
            : null;
        var steps = new List<JobStep>();

        if (element.TryGetProperty("steps", out var stepsElement) && stepsElement.ValueKind == JsonValueKind.Array)
        {
            foreach (var stepElement in stepsElement.EnumerateArray())
            {
                steps.Add(JobStepFromJsonElement(stepElement));
            }
        }

        return new Job(version, dateCreated, dateUpdated, steps);
    }

    private static JobStep JobStepFromJsonElement(JsonElement stepElement)
    {
        var args = stepElement.TryGetProperty("args", out var argsElement)
            ? ArgsFromJsonElement(argsElement)
            : new Dictionary<string, object?>();
        var statusName = stepElement.TryGetProperty("status", out var statusElement) && statusElement.ValueKind == JsonValueKind.String
            ? statusElement.GetString()
            : null;
        var status = statusName is not null && EnumNames.TryFromWireName<JobStepStatus>(statusName, out var parsedStatus)
            ? parsedStatus
            : JobStepStatus.Drafted;

        return new JobStep(args, status);
    }

    private static Dictionary<string, object?> ArgsFromJsonElement(JsonElement argsElement)
    {
        var args = new Dictionary<string, object?>();

        if (argsElement.ValueKind == JsonValueKind.Object)
        {
            foreach (var property in argsElement.EnumerateObject())
            {
                args[property.Name] = JsonElementToObject(property.Value);
            }
        }

        return args;
    }

    /// <summary>
    /// Converts a parsed <see cref="JsonElement"/> into plain CLR values (string,
    /// long/double, bool, null, List, Dictionary) matching what Python's
    /// <c>json.load</c> would hand back for the same document, so job-step args read
    /// off disk compare naturally (e.g. against a plain <c>string</c>) rather than
    /// staying boxed as <see cref="JsonElement"/>.
    /// </summary>
    private static object? JsonElementToObject(JsonElement element)
        => element.ValueKind switch
        {
            JsonValueKind.String => element.GetString(),
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.Null => null,
            JsonValueKind.Number => element.TryGetInt64(out var longValue) ? longValue : element.GetDouble(),
            JsonValueKind.Array => element.EnumerateArray().Select(JsonElementToObject).ToList(),
            JsonValueKind.Object => element.EnumerateObject().ToDictionary(p => p.Name, p => JsonElementToObject(p.Value)),
            _ => null
        };
}
