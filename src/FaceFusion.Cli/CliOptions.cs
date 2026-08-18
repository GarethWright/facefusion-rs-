namespace FaceFusion.Cli;

/// <summary>How a CLI value is parsed. Mirrors argparse type/nargs/action in program.py.</summary>
public enum CliValueKind
{
    String,
    Int,
    Float,
    Flag,
    StringList,
    IntList
}

/// <summary>One command-line option: its wire flag and the state key it feeds.</summary>
public sealed record CliOption(string Flag, string StateKey, CliValueKind Kind);

/// <summary>
/// The full option surface, GENERATED from facefusion/program.py by
/// tools/parity/generate_cli_options.py so flag names and value kinds cannot drift
/// from the Python. Every flag here appears in job JSON and in user scripts, so a
/// renamed or retyped option is a breaking change, not a cosmetic one.
///
/// Regenerate with: python3 tools/parity/generate_cli_options.py
/// </summary>
public static class CliOptions
{
    public static readonly IReadOnlyList<CliOption> All = new CliOption[]
    {
        new("--config-path", "config_path", CliValueKind.String),
        new("--temp-path", "temp_path", CliValueKind.String),
        new("--jobs-path", "jobs_path", CliValueKind.String),
        new("--face-detector-model", "face_detector_model", CliValueKind.String),
        new("--face-detector-size", "face_detector_size", CliValueKind.String),
        new("--face-detector-margin", "face_detector_margin", CliValueKind.IntList),
        new("--face-detector-angles", "face_detector_angles", CliValueKind.IntList),
        new("--face-detector-score", "face_detector_score", CliValueKind.Float),
        new("--face-landmarker-model", "face_landmarker_model", CliValueKind.String),
        new("--face-landmarker-score", "face_landmarker_score", CliValueKind.Float),
        new("--face-selector-mode", "face_selector_mode", CliValueKind.String),
        new("--face-selector-order", "face_selector_order", CliValueKind.String),
        new("--face-selector-age-start", "face_selector_age_start", CliValueKind.Int),
        new("--face-selector-age-end", "face_selector_age_end", CliValueKind.Int),
        new("--face-selector-gender", "face_selector_gender", CliValueKind.String),
        new("--face-selector-race", "face_selector_race", CliValueKind.String),
        new("--reference-face-position", "reference_face_position", CliValueKind.Int),
        new("--reference-face-distance", "reference_face_distance", CliValueKind.Float),
        new("--reference-frame-number", "reference_frame_number", CliValueKind.Int),
        new("--face-tracker-score", "face_tracker_score", CliValueKind.Float),
        new("--face-occluder-model", "face_occluder_model", CliValueKind.String),
        new("--face-parser-model", "face_parser_model", CliValueKind.String),
        new("--face-mask-types", "face_mask_types", CliValueKind.StringList),
        new("--face-mask-areas", "face_mask_areas", CliValueKind.StringList),
        new("--face-mask-regions", "face_mask_regions", CliValueKind.StringList),
        new("--face-mask-blur", "face_mask_blur", CliValueKind.Float),
        new("--face-mask-padding", "face_mask_padding", CliValueKind.IntList),
        new("--voice-extractor-model", "voice_extractor_model", CliValueKind.String),
        new("--trim-frame-start", "trim_frame_start", CliValueKind.Int),
        new("--trim-frame-end", "trim_frame_end", CliValueKind.Int),
        new("--temp-frame-format", "temp_frame_format", CliValueKind.String),
        new("--temp-pixel-format", "temp_pixel_format", CliValueKind.String),
        new("--target-frame-amount", "target_frame_amount", CliValueKind.Int),
        new("--output-image-quality", "output_image_quality", CliValueKind.Int),
        new("--output-image-scale", "output_image_scale", CliValueKind.Float),
        new("--output-audio-encoder", "output_audio_encoder", CliValueKind.String),
        new("--output-audio-quality", "output_audio_quality", CliValueKind.Int),
        new("--output-audio-volume", "output_audio_volume", CliValueKind.Int),
        new("--output-video-encoder", "output_video_encoder", CliValueKind.String),
        new("--output-video-preset", "output_video_preset", CliValueKind.String),
        new("--output-video-quality", "output_video_quality", CliValueKind.Int),
        new("--output-video-scale", "output_video_scale", CliValueKind.Float),
        new("--output-video-fps", "output_video_fps", CliValueKind.Float),
        new("--workflow-mode", "workflow_mode", CliValueKind.String),
        new("--workflow-strategy", "workflow_strategy", CliValueKind.String),
        new("--processors", "processors", CliValueKind.StringList),
        new("--open-browser", "open_browser", CliValueKind.Flag),
        new("--ui-layouts", "ui_layouts", CliValueKind.StringList),
        new("--ui-workflow", "ui_workflow", CliValueKind.String),
        new("--download-providers", "download_providers", CliValueKind.StringList),
        new("--download-scope", "download_scope", CliValueKind.String),
        new("--benchmark-mode", "benchmark_mode", CliValueKind.String),
        new("--benchmark-resolutions", "benchmark_resolutions", CliValueKind.StringList),
        new("--benchmark-cycle-count", "benchmark_cycle_count", CliValueKind.Int),
        new("--execution-device-ids", "execution_device_ids", CliValueKind.IntList),
        new("--execution-providers", "execution_providers", CliValueKind.StringList),
        new("--execution-thread-count", "execution_thread_count", CliValueKind.Int),
        new("--video-memory-strategy", "video_memory_strategy", CliValueKind.String),
        new("--log-level", "log_level", CliValueKind.String),
        new("--halt-on-error", "halt_on_error", CliValueKind.Flag)
    };
}
