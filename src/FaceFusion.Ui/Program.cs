using System.Diagnostics;
using FaceFusion.Ui;
using FaceFusion.Ui.Components;

var builder = WebApplication.CreateBuilder(new WebApplicationOptions
{
    Args = args,
    // Without this the content root is the process's working directory, so wwwroot is only
    // found when the app happens to be started from its own project directory — running the
    // built binary from the repo root logged "The WebRootPath was not found" and served no CSS.
    ContentRootPath = AppContext.BaseDirectory,
});

builder.Services.AddRazorComponents().AddInteractiveServerComponents();

// Singleton, not scoped: Python's state_manager is process-global and one FaceFusion process
// serves one user's session (Gradio's launch() has the same shape). A scoped registration would
// silently give a second browser tab a second, independent set of settings, which is a
// behaviour change rather than a hardening.
builder.Services.AddSingleton<UiState>();
builder.Services.AddSingleton<UiTerminal>();
builder.Services.AddSingleton<UiRunner>();
builder.Services.AddSingleton<UiPreview>();
builder.Services.AddSingleton<UiWebcam>();

var app = builder.Build();

app.UseStaticFiles();
app.UseAntiforgery();
app.MapRazorComponents<App>().AddInteractiveServerRenderMode();

// Python: ui.launch(inbrowser = state_manager.get_item('open_browser')), i.e. --open-browser.
if (args.Contains("--open-browser", StringComparer.Ordinal))
{
    app.Lifetime.ApplicationStarted.Register(() =>
    {
        try
        {
            var url = app.Services.GetRequiredService<Microsoft.AspNetCore.Hosting.Server.IServer>()
                .Features.Get<Microsoft.AspNetCore.Hosting.Server.Features.IServerAddressesFeature>()
                ?.Addresses.FirstOrDefault() ?? "http://localhost:7860";

            Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
        }
        catch (Exception exception)
        {
            // A headless host has no browser to open; that is not a reason to fail the run.
            Console.WriteLine($"could not open a browser: {exception.Message}");
        }
    });
}

app.Run();
