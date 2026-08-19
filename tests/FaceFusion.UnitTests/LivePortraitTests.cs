using FaceFusion.Processors;

namespace FaceFusion.UnitTests;

/// <summary>
/// Port of the ground-truth checks for <c>facefusion/processors/live_portrait.py</c>. There is
/// no <c>tests/test_live_portrait.py</c> in the Python suite, so every case below was derived by
/// hand from the module's own literal constants/formulas, cross-checked against a live Python +
/// scipy 1.17.1 REPL (see each test's comment). Real cross-checking of
/// <see cref="LivePortrait.CreateRotation"/> against a live scipy process for a non-axis-aligned
/// angle triple lives in <c>tests/FaceFusion.ParityTests/ProcessorParityTests3.cs</c>, ground
/// truth captured by <c>tools/parity/dump_processors3.py</c>.
/// </summary>
public sealed class LivePortraitTests
{
    // -----------------------------------------------------------------
    // create_rotation
    // -----------------------------------------------------------------

    [Fact]
    public void CreateRotationForIdentityAnglesIsIdentity()
    {
        var rotation = LivePortrait.CreateRotation(0f, 0f, 0f);

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Assert.Equal(i == j ? 1.0 : 0.0, rotation[i, j], 6);
            }
        }
    }

    /// <summary>
    /// Ground truth: <c>Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix()</c> is a
    /// pure rotation about X, i.e. the standard elemental Rx(90) matrix
    /// <c>[[1,0,0],[0,0,-1],[0,1,0]]</c> (cos(90)=0, sin(90)=1) — verified interactively.
    /// </summary>
    [Fact]
    public void CreateRotationForPureXRotationMatchesElementalRxMatrix()
    {
        var rotation = LivePortrait.CreateRotation(90f, 0f, 0f);

        Assert.Equal(1.0, rotation[0, 0], 5);
        Assert.Equal(0.0, rotation[0, 1], 5);
        Assert.Equal(0.0, rotation[0, 2], 5);
        Assert.Equal(0.0, rotation[1, 0], 5);
        Assert.Equal(0.0, rotation[1, 1], 4);
        Assert.Equal(-1.0, rotation[1, 2], 5);
        Assert.Equal(0.0, rotation[2, 0], 5);
        Assert.Equal(1.0, rotation[2, 1], 5);
        Assert.Equal(0.0, rotation[2, 2], 4);
    }

    /// <summary>
    /// Ground truth: <c>Rotation.from_euler('xyz', [0, 90, 0], degrees=True).as_matrix()</c> is
    /// the standard elemental Ry(90) matrix <c>[[0,0,1],[0,1,0],[-1,0,0]]</c>.
    /// </summary>
    [Fact]
    public void CreateRotationForPureYRotationMatchesElementalRyMatrix()
    {
        var rotation = LivePortrait.CreateRotation(0f, 90f, 0f);

        Assert.Equal(0.0, rotation[0, 0], 4);
        Assert.Equal(0.0, rotation[0, 1], 5);
        Assert.Equal(1.0, rotation[0, 2], 5);
        Assert.Equal(0.0, rotation[1, 0], 5);
        Assert.Equal(1.0, rotation[1, 1], 5);
        Assert.Equal(0.0, rotation[1, 2], 5);
        Assert.Equal(-1.0, rotation[2, 0], 5);
        Assert.Equal(0.0, rotation[2, 1], 5);
        Assert.Equal(0.0, rotation[2, 2], 4);
    }

    /// <summary>
    /// Ground truth from a live Python + scipy 1.17.1 REPL, confirming the composition order is
    /// <c>Rz(roll) @ Ry(yaw) @ Rx(pitch)</c> and NOT <c>Rx(pitch) @ Ry(yaw) @ Rz(roll)</c> (the two
    /// only agree when at most one angle is non-zero, so this needs all three non-zero at once):
    /// <code>
    /// pitch, yaw, roll = 10.0, 20.0, 30.0
    /// Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
    /// # [[ 0.81379768, -0.44096961,  0.37852231],
    /// #  [ 0.46984631,  0.88256412,  0.01802831],
    /// #  [-0.34202014,  0.16317591,  0.92541658]]
    /// </code>
    /// </summary>
    [Fact]
    public void CreateRotationForAllThreeAxesMatchesVerifiedScipyComposition()
    {
        var rotation = LivePortrait.CreateRotation(10f, 20f, 30f);

        double[,] expected =
        {
            { 0.81379768, -0.44096961, 0.37852231 },
            { 0.46984631, 0.88256412, 0.01802831 },
            { -0.34202014, 0.16317591, 0.92541658 },
        };

        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                Assert.Equal(expected[i, j], rotation[i, j], 6);
            }
        }
    }

    [Fact]
    public void CreateRotationIsAlwaysOrthogonalWithUnitDeterminant()
    {
        var rotation = LivePortrait.CreateRotation(17.3f, -42.1f, 8.9f);

        // R @ R.T == identity for any proper rotation matrix.
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                double dot = 0;
                for (var k = 0; k < 3; k++)
                {
                    dot += (double)rotation[i, k] * rotation[j, k];
                }

                Assert.Equal((double)(i == j ? 1.0 : 0.0), (double)(dot), 4);
            }
        }

        var determinant =
            (rotation[0, 0] * ((rotation[1, 1] * rotation[2, 2]) - (rotation[1, 2] * rotation[2, 1]))) -
            (rotation[0, 1] * ((rotation[1, 0] * rotation[2, 2]) - (rotation[1, 2] * rotation[2, 0]))) +
            (rotation[0, 2] * ((rotation[1, 0] * rotation[2, 1]) - (rotation[1, 1] * rotation[2, 0])));

        Assert.Equal((double)(1.0), (double)(determinant), 4);
    }

    // -----------------------------------------------------------------
    // limit_expression
    // -----------------------------------------------------------------

    [Fact]
    public void LimitExpressionClipsToPerElementBounds()
    {
        var expression = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            // Far below every EXPRESSION_MIN entry and far above every EXPRESSION_MAX entry,
            // alternating by row, so the clip is exercised on both sides.
            expression[i, 0] = i % 2 == 0 ? -10f : 10f;
            expression[i, 1] = i % 2 == 0 ? -10f : 10f;
            expression[i, 2] = i % 2 == 0 ? -10f : 10f;
        }

        var limited = LivePortrait.LimitExpression(expression);

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                var expected = i % 2 == 0 ? LivePortrait.ExpressionMin[i, c] : LivePortrait.ExpressionMax[i, c];
                Assert.Equal((double)expected, (double)limited[i, c], 6);
            }
        }
    }

    [Fact]
    public void LimitExpressionLeavesInBoundsValuesUnchanged()
    {
        var expression = new float[21, 3];
        var limited = LivePortrait.LimitExpression(expression); // all zeros, within every bound

        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                Assert.Equal(0f, limited[i, c]);
            }
        }
    }

    // -----------------------------------------------------------------
    // calculate_euler_limits / limit_angle
    // -----------------------------------------------------------------

    [Fact]
    public void CalculateEulerLimitsReturnsDefaultBoundsForSmallAngles()
    {
        var (pitchMin, pitchMax, yawMin, yawMax, rollMin, rollMax) = LivePortrait.CalculateEulerLimits(5f, -5f, 5f);

        Assert.Equal((double)(-30.0), (double)(pitchMin), 5);
        Assert.Equal((double)(30.0), (double)(pitchMax), 5);
        Assert.Equal((double)(-60.0), (double)(yawMin), 5);
        Assert.Equal((double)(60.0), (double)(yawMax), 5);
        Assert.Equal((double)(-20.0), (double)(rollMin), 5);
        Assert.Equal((double)(20.0), (double)(rollMax), 5);
    }

    /// <summary>
    /// Python: <c>if pitch < 0: pitch_min = min(pitch, pitch_min) else: pitch_max = max(pitch,
    /// pitch_max)</c> — a positive angle outside the default range widens the *max* bound only
    /// (the min bound stays at its default), and vice versa for a negative angle.
    /// </summary>
    [Fact]
    public void CalculateEulerLimitsWidensOnlyTheExceededBoundForOutOfRangeAngles()
    {
        var (pitchMin, pitchMax, yawMin, yawMax, rollMin, rollMax) = LivePortrait.CalculateEulerLimits(45f, -70f, 25f);

        Assert.Equal((double)(-30.0), (double)(pitchMin), 5);
        Assert.Equal((double)(45.0), (double)(pitchMax), 5);
        Assert.Equal((double)(-70.0), (double)(yawMin), 5);
        Assert.Equal((double)(60.0), (double)(yawMax), 5);
        Assert.Equal((double)(-20.0), (double)(rollMin), 5);
        Assert.Equal((double)(25.0), (double)(rollMax), 5);
    }

    /// <summary>
    /// Ground truth from a live Python REPL using the real <c>limit_angle</c> with these exact
    /// inputs (also captured as a fixture by <c>tools/parity/dump_processors3.py</c> for the
    /// parity-test tier):
    /// <code>
    /// limit_angle(45.0, -70.0, 25.0, 10.0, -10.0, 5.0) == (10.0, -10.0, 5.0)
    /// </code>
    /// (every output value is already within the widened bounds computed above, so
    /// <c>limit_angle</c> is a no-op here — see <see cref="LimitAngleClampsOutOfBoundsOutputs"/>
    /// for the clamping case).
    /// </summary>
    [Fact]
    public void LimitAngleIsNoOpWhenOutputsAreAlreadyInBounds()
    {
        var (pitch, yaw, roll) = LivePortrait.LimitAngle(45f, -70f, 25f, 10f, -10f, 5f);

        Assert.Equal((double)(10.0), (double)(pitch), 5);
        Assert.Equal((double)(-10.0), (double)(yaw), 5);
        Assert.Equal((double)(5.0), (double)(roll), 5);
    }

    [Fact]
    public void LimitAngleClampsOutOfBoundsOutputs()
    {
        var (pitch, yaw, roll) = LivePortrait.LimitAngle(5f, 5f, 5f, 999f, -999f, 999f);

        Assert.Equal((double)(30.0), (double)(pitch), 5); // default pitch_max
        Assert.Equal((double)(-60.0), (double)(yaw), 5); // default yaw_min
        Assert.Equal((double)(20.0), (double)(roll), 5); // default roll_max
    }
}
