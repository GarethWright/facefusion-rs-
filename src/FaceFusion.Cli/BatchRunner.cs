using System.Globalization;
using FaceFusion.Core;
using FaceFusion.Jobs;

namespace FaceFusion.Cli;

/// <summary>
/// Port of <c>facefusion/core.py</c>'s <c>process_batch</c> — expands <c>--source-pattern</c>/
/// <c>--target-pattern</c> globs into one job step per source-x-target pair (Python:
/// <c>itertools.product(source_paths, target_paths)</c>), or one step per target when there are
/// no sources, formatting each step's <c>--output-pattern</c> with <c>{index}</c>,
/// <c>{source_name}</c>, <c>{target_name}</c>, <c>{target_extension}</c>.
/// </summary>
public static class BatchRunner
{
	/// <summary>Python: <c>process_batch(args)</c>. Returns Python's <c>ErrorCode</c> (0
	/// success, 1 failure — including an unresolvable <c>--output-pattern</c> placeholder,
	/// which Python's own <c>except KeyError: return 1</c> also turns into a plain failure
	/// rather than letting the exception propagate).</summary>
	public static int ProcessBatch(IReadOnlyDictionary<string, object?> args, JobManager jobManager, Logger logger)
	{
		var jobId = JobHelper.SuggestJobId("batch");
		var outputPattern = StepArgsReader.GetStringOrNull(args, "output_pattern") ?? string.Empty;
		var sourcePaths = FileSystem.ResolveFilePattern(StepArgsReader.GetStringOrNull(args, "source_pattern"));
		var targetPaths = FileSystem.ResolveFilePattern(StepArgsReader.GetStringOrNull(args, "target_pattern"));

		if (!jobManager.CreateJob(jobId))
		{
			return 1;
		}

		if (sourcePaths.Count > 0 && targetPaths.Count > 0)
		{
			var index = 0;

			foreach (var sourcePath in sourcePaths)
			{
				foreach (var targetPath in targetPaths)
				{
					var stepArgs = new Dictionary<string, object?>(args, StringComparer.Ordinal)
					{
						["source_paths"] = new[] { sourcePath },
						["target_path"] = targetPath,
					};

					string outputPath;

					try
					{
						outputPath = StepArgsReader.FormatPattern(outputPattern, new Dictionary<string, string?>(StringComparer.Ordinal)
						{
							["index"] = index.ToString(CultureInfo.InvariantCulture),
							["source_name"] = FileSystem.GetFileName(sourcePath),
							["target_name"] = FileSystem.GetFileName(targetPath),
							["target_extension"] = FileSystem.GetFileExtension(targetPath),
						});
					}
					catch (FormatUnknownPlaceholderException)
					{
						return 1;
					}

					stepArgs["output_path"] = outputPath;

					if (!jobManager.AddStep(jobId, stepArgs))
					{
						return 1;
					}

					index++;
				}
			}

			if (jobManager.SubmitJob(jobId)
				&& JobRunner.RunJob(jobManager, jobId, (id, stepIndex, stepArgs) => HeadlessRunner.ProcessStep(id, stepIndex, stepArgs, jobManager, logger), HeadlessRunner.ConcatVideoStep))
			{
				return 0;
			}
		}

		if (sourcePaths.Count == 0 && targetPaths.Count > 0)
		{
			var index = 0;

			foreach (var targetPath in targetPaths)
			{
				var stepArgs = new Dictionary<string, object?>(args, StringComparer.Ordinal)
				{
					["target_path"] = targetPath,
				};

				string outputPath;

				try
				{
					outputPath = StepArgsReader.FormatPattern(outputPattern, new Dictionary<string, string?>(StringComparer.Ordinal)
					{
						["index"] = index.ToString(CultureInfo.InvariantCulture),
						["target_name"] = FileSystem.GetFileName(targetPath),
						["target_extension"] = FileSystem.GetFileExtension(targetPath),
					});
				}
				catch (FormatUnknownPlaceholderException)
				{
					return 1;
				}

				stepArgs["output_path"] = outputPath;

				if (!jobManager.AddStep(jobId, stepArgs))
				{
					return 1;
				}

				index++;
			}

			if (jobManager.SubmitJob(jobId)
				&& JobRunner.RunJob(jobManager, jobId, (id, stepIndex, stepArgs) => HeadlessRunner.ProcessStep(id, stepIndex, stepArgs, jobManager, logger), HeadlessRunner.ConcatVideoStep))
			{
				return 0;
			}
		}

		return 1;
	}
}
