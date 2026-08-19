namespace FaceFusion.Jobs;

/// <summary>
/// Port of <c>facefusion/jobs/job_store.py</c>.
///
/// Deviation from Python: the Python module keeps <c>JOB_STORE</c> as a module-level
/// global dict. Per port convention rule 5 (no global mutable state), this is an
/// instance class with the two key lists held in private fields. Callers that want
/// module-global behaviour should share a single instance (e.g. via DI).
/// </summary>
public sealed class JobStore
{
    private readonly List<string> _jobKeys = new();
    private readonly List<string> _stepKeys = new();

    /// <summary>Python: <c>get_job_keys</c>.</summary>
    public IReadOnlyList<string> GetJobKeys() => _jobKeys;

    /// <summary>Python: <c>get_step_keys</c>.</summary>
    public IReadOnlyList<string> GetStepKeys() => _stepKeys;

    /// <summary>Python: <c>register_job_keys</c>.</summary>
    public void RegisterJobKeys(IReadOnlyList<string> jobKeys)
    {
        foreach (var jobKey in jobKeys)
        {
            _jobKeys.Add(jobKey);
        }
    }

    /// <summary>Python: <c>register_step_keys</c>.</summary>
    public void RegisterStepKeys(IReadOnlyList<string> stepKeys)
    {
        foreach (var stepKey in stepKeys)
        {
            _stepKeys.Add(stepKey);
        }
    }
}
