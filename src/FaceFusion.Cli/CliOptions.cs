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
/// <param name="Alias">Short form such as <c>-t</c>, or null. Python declares these for
/// --source-paths, --target-path and --output-path; scripts use them, so they are part of
/// the contract rather than a convenience.</param>
public sealed record CliOption(string Flag, string StateKey, CliValueKind Kind, string? Alias = null);

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
        new("--config-path", "config_path", CliValueKind.String, null),
        new("--temp-path", "temp_path", CliValueKind.String, null),
        new("--jobs-path", "jobs_path", CliValueKind.String, null),
        new("--source-paths", "source_paths", CliValueKind.StringList, "-s"),
        new("--target-path", "target_path", CliValueKind.String, "-t"),
        new("--output-path", "output_path", CliValueKind.String, "-o"),
        new("--source-pattern", "source_pattern", CliValueKind.String, "-s"),
        new("--target-pattern", "target_pattern", CliValueKind.String, "-t"),
        new("--output-pattern", "output_pattern", CliValueKind.String, "-o"),
        new("--face-detector-model", "face_detector_model", CliValueKind.String, null),
        new("--face-detector-size", "face_detector_size", CliValueKind.String, null),
        new("--face-detector-margin", "face_detector_margin", CliValueKind.IntList, null),
        new("--face-detector-angles", "face_detector_angles", CliValueKind.IntList, null),
        new("--face-detector-score", "face_detector_score", CliValueKind.Float, null),
        new("--face-landmarker-model", "face_landmarker_model", CliValueKind.String, null),
        new("--face-landmarker-score", "face_landmarker_score", CliValueKind.Float, null),
        new("--face-selector-mode", "face_selector_mode", CliValueKind.String, null),
        new("--face-selector-order", "face_selector_order", CliValueKind.String, null),
        new("--face-selector-age-start", "face_selector_age_start", CliValueKind.Int, null),
        new("--face-selector-age-end", "face_selector_age_end", CliValueKind.Int, null),
        new("--face-selector-gender", "face_selector_gender", CliValueKind.String, null),
        new("--face-selector-race", "face_selector_race", CliValueKind.String, null),
        new("--reference-face-position", "reference_face_position", CliValueKind.Int, null),
        new("--reference-face-distance", "reference_face_distance", CliValueKind.Float, null),
        new("--reference-frame-number", "reference_frame_number", CliValueKind.Int, null),
        new("--face-tracker-score", "face_tracker_score", CliValueKind.Float, null),
        new("--face-occluder-model", "face_occluder_model", CliValueKind.String, null),
        new("--face-parser-model", "face_parser_model", CliValueKind.String, null),
        new("--face-mask-types", "face_mask_types", CliValueKind.StringList, null),
        new("--face-mask-areas", "face_mask_areas", CliValueKind.StringList, null),
        new("--face-mask-regions", "face_mask_regions", CliValueKind.StringList, null),
        new("--face-mask-blur", "face_mask_blur", CliValueKind.Float, null),
        new("--face-mask-padding", "face_mask_padding", CliValueKind.IntList, null),
        new("--voice-extractor-model", "voice_extractor_model", CliValueKind.String, null),
        new("--trim-frame-start", "trim_frame_start", CliValueKind.Int, null),
        new("--trim-frame-end", "trim_frame_end", CliValueKind.Int, null),
        new("--temp-frame-format", "temp_frame_format", CliValueKind.String, null),
        new("--temp-pixel-format", "temp_pixel_format", CliValueKind.String, null),
        new("--target-frame-amount", "target_frame_amount", CliValueKind.Int, null),
        new("--output-image-quality", "output_image_quality", CliValueKind.Int, null),
        new("--output-image-scale", "output_image_scale", CliValueKind.Float, null),
        new("--output-audio-encoder", "output_audio_encoder", CliValueKind.String, null),
        new("--output-audio-quality", "output_audio_quality", CliValueKind.Int, null),
        new("--output-audio-volume", "output_audio_volume", CliValueKind.Int, null),
        new("--output-video-encoder", "output_video_encoder", CliValueKind.String, null),
        new("--output-video-preset", "output_video_preset", CliValueKind.String, null),
        new("--output-video-quality", "output_video_quality", CliValueKind.Int, null),
        new("--output-video-scale", "output_video_scale", CliValueKind.Float, null),
        new("--output-video-fps", "output_video_fps", CliValueKind.Float, null),
        new("--workflow-mode", "workflow_mode", CliValueKind.String, null),
        new("--workflow-strategy", "workflow_strategy", CliValueKind.String, null),
        new("--processors", "processors", CliValueKind.StringList, null),
        new("--open-browser", "open_browser", CliValueKind.Flag, null),
        new("--ui-layouts", "ui_layouts", CliValueKind.StringList, null),
        new("--ui-workflow", "ui_workflow", CliValueKind.String, null),
        new("--download-providers", "download_providers", CliValueKind.StringList, null),
        new("--download-scope", "download_scope", CliValueKind.String, null),
        new("--benchmark-mode", "benchmark_mode", CliValueKind.String, null),
        new("--benchmark-resolutions", "benchmark_resolutions", CliValueKind.StringList, null),
        new("--benchmark-cycle-count", "benchmark_cycle_count", CliValueKind.Int, null),
        new("--execution-device-ids", "execution_device_ids", CliValueKind.IntList, null),
        new("--execution-providers", "execution_providers", CliValueKind.StringList, null),
        new("--execution-thread-count", "execution_thread_count", CliValueKind.Int, null),
        new("--video-memory-strategy", "video_memory_strategy", CliValueKind.String, null),
        new("--log-level", "log_level", CliValueKind.String, null),
        new("--halt-on-error", "halt_on_error", CliValueKind.Flag, null),
        new("--version", "version", CliValueKind.String, "-v"),
        new("--age-modifier-model", "age_modifier_model", CliValueKind.String, null),
        new("--age-modifier-direction", "age_modifier_direction", CliValueKind.Int, null),
        new("--background-remover-model", "background_remover_model", CliValueKind.String, null),
        new("--background-remover-fill-color", "background_remover_fill_color", CliValueKind.IntList, null),
        new("--background-remover-despill-color", "background_remover_despill_color", CliValueKind.IntList, null),
        new("--deep-swapper-model", "deep_swapper_model", CliValueKind.String, null),
        new("--deep-swapper-morph", "deep_swapper_morph", CliValueKind.Int, null),
        new("--expression-restorer-model", "expression_restorer_model", CliValueKind.String, null),
        new("--expression-restorer-factor", "expression_restorer_factor", CliValueKind.Int, null),
        new("--expression-restorer-areas", "expression_restorer_areas", CliValueKind.StringList, null),
        new("--face-debugger-items", "face_debugger_items", CliValueKind.StringList, null),
        new("--face-editor-model", "face_editor_model", CliValueKind.String, null),
        new("--face-editor-eyebrow-direction", "face_editor_eyebrow_direction", CliValueKind.Float, null),
        new("--face-editor-eye-gaze-horizontal", "face_editor_eye_gaze_horizontal", CliValueKind.Float, null),
        new("--face-editor-eye-gaze-vertical", "face_editor_eye_gaze_vertical", CliValueKind.Float, null),
        new("--face-editor-eye-open-ratio", "face_editor_eye_open_ratio", CliValueKind.Float, null),
        new("--face-editor-lip-open-ratio", "face_editor_lip_open_ratio", CliValueKind.Float, null),
        new("--face-editor-mouth-grim", "face_editor_mouth_grim", CliValueKind.Float, null),
        new("--face-editor-mouth-pout", "face_editor_mouth_pout", CliValueKind.Float, null),
        new("--face-editor-mouth-purse", "face_editor_mouth_purse", CliValueKind.Float, null),
        new("--face-editor-mouth-smile", "face_editor_mouth_smile", CliValueKind.Float, null),
        new("--face-editor-mouth-position-horizontal", "face_editor_mouth_position_horizontal", CliValueKind.Float, null),
        new("--face-editor-mouth-position-vertical", "face_editor_mouth_position_vertical", CliValueKind.Float, null),
        new("--face-editor-head-pitch", "face_editor_head_pitch", CliValueKind.Float, null),
        new("--face-editor-head-yaw", "face_editor_head_yaw", CliValueKind.Float, null),
        new("--face-editor-head-roll", "face_editor_head_roll", CliValueKind.Float, null),
        new("--face-enhancer-model", "face_enhancer_model", CliValueKind.String, null),
        new("--face-enhancer-blend", "face_enhancer_blend", CliValueKind.Int, null),
        new("--face-enhancer-weight", "face_enhancer_weight", CliValueKind.Float, null),
        new("--face-swapper-model", "face_swapper_model", CliValueKind.String, null),
        new("--face-swapper-pixel-boost", "face_swapper_pixel_boost", CliValueKind.String, null),
        new("--face-swapper-weight", "face_swapper_weight", CliValueKind.Float, null),
        new("--frame-colorizer-model", "frame_colorizer_model", CliValueKind.String, null),
        new("--frame-colorizer-size", "frame_colorizer_size", CliValueKind.String, null),
        new("--frame-colorizer-blend", "frame_colorizer_blend", CliValueKind.Int, null),
        new("--frame-enhancer-model", "frame_enhancer_model", CliValueKind.String, null),
        new("--frame-enhancer-blend", "frame_enhancer_blend", CliValueKind.Int, null),
        new("--lip-syncer-model", "lip_syncer_model", CliValueKind.String, null),
        new("--lip-syncer-weight", "lip_syncer_weight", CliValueKind.Float, null)
    };
}
