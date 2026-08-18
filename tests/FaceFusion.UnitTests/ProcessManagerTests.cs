using FaceFusion.Core;
using FaceFusion.Types;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of <c>tests/test_process_manager.py</c>.
///
/// The Python module keeps process state on a module-level global reached through
/// free functions; here <see cref="ProcessManager"/> is an instance class (port
/// convention rule 5), so each test constructs its own instance instead of resetting a
/// shared global via <c>set_process_state('pending')</c>.
/// </summary>
public sealed class ProcessManagerTests
{
    [Fact]
    public void TestStart()
    {
        var processManager = new ProcessManager();
        processManager.SetProcessState(ProcessState.Pending);

        processManager.Start();

        Assert.True(processManager.IsProcessing());
    }

    [Fact]
    public void TestStop()
    {
        var processManager = new ProcessManager();
        processManager.SetProcessState(ProcessState.Processing);

        processManager.Stop();

        Assert.True(processManager.IsStopping());
    }

    [Fact]
    public void TestEnd()
    {
        var processManager = new ProcessManager();
        processManager.SetProcessState(ProcessState.Processing);

        processManager.End();

        Assert.True(processManager.IsPending());
    }

    [Fact]
    public void TestCheck()
    {
        var processManager = new ProcessManager();
        processManager.SetProcessState(ProcessState.Pending);

        processManager.Check();

        Assert.True(processManager.IsChecking());
    }

    [Fact]
    public void TestManageScopeStartsAndEnds()
    {
        var processManager = new ProcessManager();
        processManager.SetProcessState(ProcessState.Pending);

        using (processManager.Manage())
        {
            Assert.True(processManager.IsProcessing());
        }

        Assert.True(processManager.IsPending());
    }
}
