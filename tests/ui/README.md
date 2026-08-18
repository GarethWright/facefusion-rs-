# UI end-to-end test

Drives the Blazor UI in a real browser, because the failure mode this UI actually has is not a
compile error. An earlier version of the layout compiled, served a correct page, and ran jobs
correctly — while every conditional panel showed stale markup, so ticking a processor left its
options hidden and a valid target path never showed as found. The state was right and the
screen was wrong. Only a browser sees that.

## What it checks

1. The default layout renders, with only the selected processor's options block visible.
2. Ticking and unticking a processor shows and hides its options block (Python does this with
   an explicit `.change()` handler on the processors checkbox group).
3. Changing the face-swapper model resets pixel boost to the new model's first choice
   (Python: `update_face_swapper_model`).
4. The reference-face controls appear only in `reference` selector mode.
5. A real run: sets the target and output paths, clicks START, and waits for the terminal to
   report success and the output file to exist.
6. The preview renders a frame.

## Running it

    # from the repo root, with the .NET SDK on PATH
    dotnet build src/FaceFusion.Ui -c Release
    ./src/FaceFusion.Ui/bin/Release/net8.0/FaceFusion.Ui &

    cd tests/ui
    npm install            # playwright only; the browser is expected to be already installed
    node ui-test.mjs

`CHROMIUM_PATH` overrides the browser binary. The test needs the example media in
`/tmp/facefusion-test-examples` (`tools/parity/fetch_examples.sh`) and the `ddcolor` model in
`.assets/models`, the same prerequisites the parity tests have.

## Output parity

The video this test produces through the UI was compared against the same run driven from the
CLI: **byte-identical** (PSNR inf, max difference 0). That is the point of `UiState.BuildArgs`
handing `HeadlessRunner.ProcessHeadless` the same flat argument bag the CLI builds from argv —
there is no UI-only processing path that could drift.
