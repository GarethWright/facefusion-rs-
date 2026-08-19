namespace FaceFusion.UnitTests;

/// <summary>
/// Serialises the test classes that share the on-disk output directory
/// (<see cref="TestHelper.GetTestOutputsDirectory"/>).
///
/// Those classes call <c>PrepareTestOutputDirectory</c>, which deletes and recreates that
/// one directory. xunit runs test collections in parallel by default, so without this they
/// wipe each other's files mid-run and fail non-deterministically — passing in isolation
/// and failing in a full run, which is exactly the symptom that surfaced.
///
/// The Python suite does not need this because pytest runs sequentially.
/// </summary>
[CollectionDefinition("MediaOutput", DisableParallelization = true)]
public sealed class MediaOutputCollection
{
}
