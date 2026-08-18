using System.Globalization;
using FaceFusion.Core;

namespace FaceFusion.Jobs;

/// <summary>
/// Port of <c>facefusion/jobs/job_helper.py</c>.
/// </summary>
public static class JobHelper
{
    /// <summary>Python: <c>get_step_output_path</c>.</summary>
    public static string? GetStepOutputPath(string jobId, int stepIndex, string? outputPath)
    {
        if (!string.IsNullOrEmpty(outputPath))
        {
            // Python: os.path.split(output_path) — split on the last path separator,
            // matching the semantics FileSystem's private GetDirectoryName/GetBaseName
            // helpers already implement for other filesystem.py ports (they are not
            // exposed publicly, so the split is reproduced here directly).
            var separatorIndex = outputPath.LastIndexOfAny(new[] { '/', Path.DirectorySeparatorChar });
            var outputDirectoryPath = separatorIndex < 0 ? string.Empty : outputPath[..separatorIndex];
            var outputFilePath = separatorIndex < 0 ? outputPath : outputPath[(separatorIndex + 1)..];

            var outputFileName = FileSystem.GetFileName(outputFilePath);
            var outputFileExtension = FileSystem.GetFileExtension(outputFilePath);

            if (!string.IsNullOrEmpty(outputFileName) && !string.IsNullOrEmpty(outputFileExtension))
            {
                var fileName = outputFileName + "-" + jobId + "-" + stepIndex.ToString(CultureInfo.InvariantCulture) + outputFileExtension;

                return string.IsNullOrEmpty(outputDirectoryPath) ? fileName : Path.Combine(outputDirectoryPath, fileName);
            }
        }

        return null;
    }

    /// <summary>Python: <c>suggest_job_id</c>.</summary>
    public static string SuggestJobId(string jobPrefix = "job")
        => jobPrefix + "-" + DateTime.Now.ToString("yyyy-MM-dd-HH-mm-ss", CultureInfo.InvariantCulture);
}
