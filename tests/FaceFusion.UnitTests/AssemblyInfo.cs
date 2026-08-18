using Xunit;

// xunit runs different test classes in parallel by default. Several classes in this
// project drive real ffmpeg subprocesses against shared, name-derived temp directories
// (FaceFusion.Core.TempHelper.GetTempDirectoryPath keys purely off the target file's
// name, not the test class), and some (ProcessHelperTests, FfmpegTests, FfprobeTests,
// FfmpegPipelineTests) mutate/depend on the resolved ffmpeg/ffprobe binary path via the
// ffmpegPath/ffprobePath override added for deterministic "binary not found" testing —
// none of that is safe under cross-class parallelism. Disabling parallelization keeps the
// whole assembly deterministic; the suite is still fast (small 240p fixtures, ultrafast
// preset).
[assembly: CollectionBehavior(DisableTestParallelization = true)]
