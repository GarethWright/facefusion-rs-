#!/usr/bin/env python3
"""
Generates src/FaceFusion.Cli/UiOptionDescriptors.cs from the REAL argparse parser
facefusion/program.py builds.

Why introspect the parser rather than regex the source (which
generate_cli_options.py does for flag names): the UI needs each option's *default*
and its *choice list*, and both are computed at runtime — defaults come from
config.get_*_value(...) reading facefusion.ini, and choices come from expressions
like facefusion.choices.face_detector_models or a create_int_range(...) call. A
regex would have to re-implement all of that. argparse already holds the answers.

Run with the repo root as the working directory:
    python3 tools/parity/generate_ui_options.py
"""
import argparse
import io
import os
import sys

sys.path.insert(0, os.getcwd())

import facefusion.program  # noqa: E402

OUTPUT_PATH = 'src/FaceFusion.Cli/UiOptionDescriptors.cs'

# argparse's own help action and the batch/pattern flags, which the UI has no control
# for: it always drives a single run, not a glob batch (batch-run stays a CLI command).
SKIPPED_DESTS = {'help', 'source_pattern', 'target_pattern', 'output_pattern'}


def resolve_kind(action):
	type_name = getattr(action.type, '__name__', None)

	if isinstance(action, argparse._StoreTrueAction):
		return 'Flag'
	if action.nargs == '+':
		# sanitize_int_range is a functools.partial, so it has no __name__; every
		# multi-value numeric flag in program.py goes through it or through plain int.
		return 'IntList' if (type_name == 'int' or type_name is None and action.type is not None) else 'StringList'
	if type_name == 'int':
		return 'Int'
	if type_name == 'float':
		return 'Float'
	return 'String'


def csharp_string(value):
	if value is None:
		return 'null'
	escaped = str(value).replace('\\', '\\\\').replace('"', '\\"')
	return f'"{escaped}"'


def csharp_default(action, kind):
	default = action.default

	if default is None:
		return 'null'
	if kind == 'Flag':
		return '"true"' if default else '"false"'
	if isinstance(default, (list, tuple)):
		return csharp_string(' '.join(str(item) for item in default))
	return csharp_string(default)


def csharp_choices(action):
	if not action.choices:
		return 'Array.Empty<string>()'

	items = ', '.join(csharp_string(choice) for choice in action.choices)
	return f'new[] {{ {items} }}'


def main():
	program = facefusion.program.create_program()
	subparsers = program._actions[2].choices
	run_parser = subparsers['run']

	buffer = io.StringIO()
	buffer.write('''namespace FaceFusion.Cli;

/// <summary>
/// One option as the UI needs to render it: the state key it feeds, how its value is
/// parsed, the default the Python parser resolves (after facefusion.ini is read), and the
/// full choice list when argparse restricts one.
/// </summary>
/// <param name="Default">Written the way the CLI would receive it, i.e. a list default is
/// space-joined, so one string round-trips every <see cref="CliValueKind"/>.</param>
public sealed record UiOptionDescriptor(
	string StateKey,
	string Flag,
	CliValueKind Kind,
	string? Default,
	IReadOnlyList<string> Choices);

/// <summary>
/// GENERATED from the real argparse parser facefusion/program.py builds, by
/// tools/parity/generate_ui_options.py. Defaults and choice lists are computed at runtime
/// in Python (defaults read facefusion.ini through config.get_*_value; choices come from
/// facefusion.choices and create_*_range calls), so they are introspected off the
/// constructed parser rather than transcribed — the UI would otherwise drift from the CLI
/// silently, and a UI default that disagrees with the CLI's is exactly the kind of
/// divergence this port is trying to avoid.
///
/// Regenerate with: python3 tools/parity/generate_ui_options.py
/// </summary>
public static class UiOptionDescriptors
{
	public static readonly IReadOnlyList<UiOptionDescriptor> All = new UiOptionDescriptor[]
	{
''')

	count = 0

	for action in run_parser._actions:
		if not action.option_strings or action.dest in SKIPPED_DESTS:
			continue

		kind = resolve_kind(action)
		buffer.write(
			f'\t\tnew("{action.dest}", "{action.option_strings[-1]}", CliValueKind.{kind}, '
			f'{csharp_default(action, kind)}, {csharp_choices(action)}),\n')
		count += 1

	buffer.write('''	};

	private static readonly Dictionary<string, UiOptionDescriptor> ByStateKey =
		All.ToDictionary(descriptor => descriptor.StateKey, StringComparer.Ordinal);

	/// <summary>Throws when the key is unknown — a typo in a Razor component should fail
	/// loudly at render time rather than silently binding to nothing.</summary>
	public static UiOptionDescriptor Get(string stateKey)
		=> ByStateKey.TryGetValue(stateKey, out var descriptor)
			? descriptor
			: throw new KeyNotFoundException($"no UI option descriptor for state key '{stateKey}'.");

	public static bool Has(string stateKey) => ByStateKey.ContainsKey(stateKey);
}
''')

	with open(OUTPUT_PATH, 'w') as file:
		file.write(buffer.getvalue())

	print(f'wrote {OUTPUT_PATH} with {count} descriptors')


if __name__ == '__main__':
	main()
