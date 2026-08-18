using System.CommandLine;
using FaceFusion.Core;
using FaceFusion.Jobs;
using FaceFusion.Types;

namespace FaceFusion.Cli;

/// <summary>
/// Entry point, port of <c>facefusion.py</c> plus <c>program.py</c>'s command surface.
///
/// Only this class exits the process; every layer beneath returns an error code, unlike
/// Python's <c>hard_exit</c>/<c>fatal_exit</c> which call <c>sys.exit</c>/<c>os._exit</c>
/// from library code.
/// </summary>
public static class Program
{
    public static int Main(string[] args)
    {
        var root = CommandFactory.CreateRootCommand();
        return root.Parse(args).Invoke();
    }
}

/// <summary>Builds the command tree. Flag names come from the generated <see cref="CliOptions"/>.</summary>
public static class CommandFactory
{
    private static readonly string[] JobManagerCommands =
    {
        "job-list", "job-create", "job-submit", "job-submit-all", "job-delete",
        "job-delete-all", "job-add-step", "job-remix-step", "job-insert-step", "job-remove-step"
    };

    private static readonly string[] JobRunnerCommands =
    {
        "job-run", "job-run-all", "job-retry", "job-retry-all"
    };

    public static RootCommand CreateRootCommand()
    {
        var root = new RootCommand("FaceFusion — industry leading face manipulation platform");

        var jobsPath = new Option<string>("--jobs-path") { Description = "specify the directory containing jobs" };
        jobsPath.DefaultValueFactory = _ => ".jobs";

        var jobId = new Option<string?>("--job-id") { Description = "specify the job id" };
        var stepIndex = new Option<int?>("--step-index") { Description = "specify the step index" };
        var haltOnError = new Option<bool>("--halt-on-error") { Description = "stop as soon as a job fails" };
        var jobStatus = new Option<string>("--job-status") { Description = "specify the job status" };
        jobStatus.DefaultValueFactory = _ => "drafted";

        foreach (var name in JobManagerCommands)
        {
            var command = new Command(name, DescribeCommand(name)) { jobsPath };

            if (NeedsJobId(name))
            {
                command.Add(jobId);
            }

            if (NeedsStepIndex(name))
            {
                command.Add(stepIndex);
            }

            if (name is "job-submit-all" or "job-delete-all")
            {
                command.Add(haltOnError);
            }

            if (name == "job-list")
            {
                command.Add(jobStatus);
            }

            // Step-carrying commands accept the whole processing option surface, exactly
            // as program.py attaches the step arguments to these subcommands.
            var stepOptions = name is "job-add-step" or "job-remix-step" or "job-insert-step"
                ? AttachStepOptions(command, name)
                : new Dictionary<string, (Option Option, CliValueKind Kind)>();

            command.SetAction(result =>
            {
                var manager = new JobManager(result.GetValue(jobsPath) ?? ".jobs");

                if (!manager.InitJobs())
                {
                    return 1;
                }

                var router = new JobRouter(manager, new Logger());
                var status = EnumNames.TryFromWireName<JobStatus>(result.GetValue(jobStatus) ?? "drafted", out var parsed)
                    ? parsed
                    : JobStatus.Drafted;

                return router.RouteJobManager(
                    name,
                    NeedsJobId(name) ? result.GetValue(jobId) : null,
                    NeedsStepIndex(name) ? result.GetValue(stepIndex) : null,
                    status,
                    result.GetValue(haltOnError),
                    CollectStepArgs(result, stepOptions));
            });

            root.Add(command);
        }

        foreach (var name in JobRunnerCommands)
        {
            var command = new Command(name, DescribeCommand(name)) { jobsPath };

            if (name is "job-run" or "job-retry")
            {
                command.Add(jobId);
            }

            if (name is "job-run-all" or "job-retry-all")
            {
                command.Add(haltOnError);
            }

            command.SetAction(result =>
            {
                var manager = new JobManager(result.GetValue(jobsPath) ?? ".jobs");

                if (!manager.InitJobs())
                {
                    return 1;
                }

                var router = new JobRouter(manager, new Logger());

                // Running a step needs the full processing pipeline, which is assembled by
                // the headless path. Until that is wired, report rather than pretend.
                return router.RouteJobRunner(
                    name,
                    result.GetValue(jobId),
                    result.GetValue(haltOnError),
                    (_, _, _) =>
                    {
                        Console.Error.WriteLine(
                            "running job steps requires the processing pipeline, which is not wired to the CLI yet");
                        return false;
                    },
                    (outputPath, tempOutputPaths) =>
                        FaceFusion.Media.Ffmpeg.ConcatVideo(outputPath, tempOutputPaths));
            });

            root.Add(command);
        }

        root.Add(CreateUiPlaceholderCommand());
        return root;
    }

    /// <summary>
    /// <c>run</c> launches the Gradio UI in Python; the Blazor UI is Phase 7. Reports
    /// clearly rather than crashing, and uses exit code 2 to match core.py's
    /// pre-check failure code rather than inventing a new one.
    /// </summary>
    private static Command CreateUiPlaceholderCommand()
    {
        var command = new Command("run", "launch the user interface");

        command.SetAction(_ =>
        {
            Console.Error.WriteLine("the user interface is not ported yet (plan phase 7); use headless-run");
            return 2;
        });

        return command;
    }

    private static Dictionary<string, (Option Option, CliValueKind Kind)> AttachStepOptions(Command command, string commandName)
    {
        var map = new Dictionary<string, (Option Option, CliValueKind Kind)>(StringComparer.Ordinal);

        // Each subcommand accepts its own subset, taken from the real Python --help rather
        // than attaching the whole 106-flag surface to everything.
        var allowed = CliCommands.FlagsByCommand.TryGetValue(commandName, out var flags)
            ? new HashSet<string>(flags, StringComparer.Ordinal)
            : new HashSet<string>(CliOptions.All.Select(option => option.Flag), StringComparer.Ordinal);

        foreach (var definition in CliOptions.All)
        {
            if (!allowed.Contains(definition.Flag))
            {
                continue;
            }

            // These are already attached as command-level options above.
            if (definition.Flag is "--jobs-path" or "--job-id" or "--step-index" or "--halt-on-error" or "--job-status")
            {
                continue;
            }

            // Aliases matter: -s/-t/-o are what most user scripts actually type.
            var names = definition.Alias is null
                ? new[] { definition.Flag }
                : new[] { definition.Flag, definition.Alias };

            Option option = definition.Kind switch
            {
                CliValueKind.Int => new Option<int?>(names[0], names),
                CliValueKind.Float => new Option<double?>(names[0], names),
                CliValueKind.Flag => new Option<bool>(names[0], names),
                CliValueKind.IntList => new Option<int[]>(names[0], names) { AllowMultipleArgumentsPerToken = true },
                CliValueKind.StringList => new Option<string[]>(names[0], names) { AllowMultipleArgumentsPerToken = true },
                _ => new Option<string?>(names[0], names)
            };

            command.Add(option);
            map[definition.StateKey] = (option, definition.Kind);
        }

        return map;
    }

    private static IReadOnlyDictionary<string, object?> CollectStepArgs(
        ParseResult result,
        IReadOnlyDictionary<string, (Option Option, CliValueKind Kind)> stepOptions)
    {
        var args = new Dictionary<string, object?>(StringComparer.Ordinal);

        foreach (var (stateKey, (option, kind)) in stepOptions)
        {
            // Only options the user actually supplied become step args; Python likewise
            // only writes what argparse produced for this invocation.
            if (result.GetResult(option) is null)
            {
                continue;
            }

            // GetValue needs the concrete Option<T> to infer T, so dispatch on the kind
            // recorded alongside the option rather than on the erased base type.
            args[stateKey] = kind switch
            {
                CliValueKind.Int => result.GetValue((Option<int?>)option),
                CliValueKind.Float => result.GetValue((Option<double?>)option),
                CliValueKind.Flag => result.GetValue((Option<bool>)option),
                CliValueKind.IntList => result.GetValue((Option<int[]>)option),
                CliValueKind.StringList => result.GetValue((Option<string[]>)option),
                _ => result.GetValue((Option<string?>)option)
            };
        }

        return args;
    }

    private static bool NeedsJobId(string command) =>
        command is not ("job-list" or "job-submit-all" or "job-delete-all");

    private static bool NeedsStepIndex(string command) =>
        command is "job-remix-step" or "job-insert-step" or "job-remove-step";

    private static string DescribeCommand(string command) => command switch
    {
        "job-list" => "list jobs by status",
        "job-create" => "create a drafted job",
        "job-submit" => "submit a drafted job to become a queued job",
        "job-submit-all" => "submit all drafted jobs to become queued jobs",
        "job-delete" => "delete a job",
        "job-delete-all" => "delete all jobs",
        "job-add-step" => "add a step to a drafted job",
        "job-remix-step" => "remix a previous step from a drafted job",
        "job-insert-step" => "insert a step to a drafted job",
        "job-remove-step" => "remove a step from a drafted job",
        "job-run" => "run a queued job",
        "job-run-all" => "run all queued jobs",
        "job-retry" => "retry a failed job",
        "job-retry-all" => "retry all failed jobs",
        _ => command
    };
}
