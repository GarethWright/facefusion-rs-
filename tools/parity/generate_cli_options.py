"""
Generate src/FaceFusion.Cli/CliOptions.cs from facefusion/program.py.

Every CLI flag appears in job JSON and in user scripts, so a renamed or retyped option is
a breaking change. Generating the C# table from the Python source means the two cannot
drift; CliOptionsTests asserts the generated table still matches program.py.

Usage: python3 tools/parity/generate_cli_options.py
"""

import re

PROGRAM_PATH = 'facefusion/program.py'
OUTPUT_PATH = 'src/FaceFusion.Cli/CliOptions.cs'


def parse_options():
	source = open(PROGRAM_PATH).read()
	rows, seen = [], set()

	for match in re.finditer(r"add_argument\(\s*'(--[a-z0-9-]+)'(.*?)\)\n", source, re.S):
		flag, rest = match.group(1), match.group(2)

		if flag in seen:
			continue
		seen.add(flag)

		argument_type = (re.search(r"type\s*=\s*([\w.]+)", rest) or [ None, 'str' ])[1]
		action = (re.search(r"action\s*=\s*'([\w_]+)'", rest) or [ None, None ])[1]
		nargs = (re.search(r"nargs\s*=\s*'([^']+)'", rest) or [ None, None ])[1]

		if action == 'store_true':
			kind = 'Flag'
		elif nargs == '+':
			# functools.partial(...) wraps a numeric caster in every case it is used.
			kind = 'IntList' if argument_type in ('int', 'partial') else 'StringList'
		elif argument_type == 'int':
			kind = 'Int'
		elif argument_type == 'float':
			kind = 'Float'
		else:
			kind = 'String'

		rows.append((flag, flag[2:].replace('-', '_'), kind))

	return rows


def main():
	rows = parse_options()
	entries = ',\n'.join('        new("%s", "%s", CliValueKind.%s)' % row for row in rows)
	header = open(OUTPUT_PATH).read().split('public static class CliOptions')[0]
	open(OUTPUT_PATH, 'w').write(
		header
		+ 'public static class CliOptions\n{\n'
		+ '    public static readonly IReadOnlyList<CliOption> All = new CliOption[]\n    {\n'
		+ entries + '\n    };\n}\n')
	print('generated', len(rows), 'options')


if __name__ == '__main__':
	main()
