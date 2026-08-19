namespace FaceFusion.Processors;

/// <summary>
/// Port of <c>facefusion/processors/live_portrait.py</c> — shared support for the
/// <c>live_portrait</c>-family models used by <c>expression_restorer</c> (this assignment) and,
/// in a later phase, <c>face_editor</c>.
///
/// <para>
/// <b>Array shapes: the leading batch dimension of 1 is dropped throughout, matching every
/// other landmark-shaped array in this codebase.</b> Python's <c>LivePortraitExpression</c>/
/// <c>LivePortraitMotionPoints</c> are <c>(1, 21, 3)</c> ONNX outputs; here they are
/// <c>float[21, 3]</c>, the same convention <c>FaceHelper</c> already uses for
/// <c>float[,]</c> landmark arrays (e.g. <c>float[5,2]</c> for <c>FaceLandmark5</c>).
/// <c>LivePortraitRotation</c> has no batch dimension in Python either (<c>create_rotation</c>
/// returns a bare <c>(3, 3)</c> matrix) and stays <c>float[3,3]</c> here for the same reason.
/// <c>LivePortraitScale</c> is a <c>(1, 1)</c> ONNX output used only as a scalar multiplier —
/// represented as a bare <see cref="float"/>. <c>LivePortraitTranslation</c> is <c>(1, 3)</c> —
/// represented as <c>float[3]</c>. <c>LivePortraitPitch</c>/<c>Yaw</c>/<c>Roll</c> are already
/// <c>float</c> in Python's own <c>processors/types.py</c> (confirmed against the real
/// <c>live_portrait_motion_extractor.onnx</c>: those three outputs have scalar/0-d shape), so
/// they stay <see cref="float"/> here too.
/// </para>
///
/// <para>
/// <b><see cref="CreateRotation"/>'s Euler convention — verified against real scipy 1.17.1, not
/// assumed.</b> Python: <c>scipy.spatial.transform.Rotation.from_euler('xyz', [pitch, yaw,
/// roll], degrees=True).as_matrix()</c>. Lowercase axis letters mean *extrinsic* (fixed-frame)
/// rotations, applied in the listed order about the world axes — this composes as
/// <c>R = Rz(roll) @ Ry(yaw) @ Rx(pitch)</c> using the standard right-handed elemental rotation
/// matrices (no sign flip — this is the ordinary math convention, not a "camera" convention
/// that negates any angle). Verified interactively:
/// <code>
/// python3 -c "
/// import math, numpy as np
/// from scipy.spatial.transform import Rotation
/// pitch, yaw, roll = 10.0, 20.0, 30.0
/// print(Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix())
/// p, y, r = map(math.radians, [pitch, yaw, roll])
/// Rx = np.array([[1,0,0],[0,math.cos(p),-math.sin(p)],[0,math.sin(p),math.cos(p)]])
/// Ry = np.array([[math.cos(y),0,math.sin(y)],[0,1,0],[-math.sin(y),0,math.cos(y)]])
/// Rz = np.array([[math.cos(r),-math.sin(r),0],[math.sin(r),math.cos(r),0],[0,0,1]])
/// print(Rz @ Ry @ Rx)"
/// </code>
/// prints the identical 3x3 matrix from both computations (to full float64 precision); the
/// alternative composition order <c>Rx @ Ry @ Rz</c> (what an *intrinsic* 'XYZ' convention, or a
/// naive "apply pitch then yaw then roll" reading, would produce) does not match. This is also
/// cross-checked end to end against the real ONNX motion extractor's outputs in
/// <c>ProcessorParityTests3.TestCreateRotationMatchesPython</c> (ground truth from
/// <c>tools/parity/dump_processors3.py</c>'s <c>live_portrait/rotation_*</c> fixtures, an
/// arbitrary non-axis-aligned (pitch, yaw, roll) triple) — matched Python exactly
/// (rtol=atol=0; pure float32 arithmetic, no ONNX Runtime/OpenCV involved).
/// </para>
/// </summary>
public static class LivePortrait
{
    // -----------------------------------------------------------------
    // EXPRESSION_MIN / EXPRESSION_MAX (batch dim dropped — see class remarks)
    // -----------------------------------------------------------------

    /// <summary>Python: <c>EXPRESSION_MIN</c> (with the leading batch dimension dropped).</summary>
    public static readonly float[,] ExpressionMin =
    {
        { -2.88067125e-02f, -8.12731311e-02f, -1.70541159e-03f },
        { -4.88598682e-02f, -3.32196616e-02f, -1.67431499e-04f },
        { -6.75425082e-02f, -4.28681746e-02f, -1.98950816e-04f },
        { -7.23103955e-02f, -3.28503326e-02f, -7.31324719e-04f },
        { -3.87073644e-02f, -6.01546466e-02f, -5.50269964e-04f },
        { -6.38048723e-02f, -2.23840728e-01f, -7.13261834e-04f },
        { -3.02710701e-02f, -3.93195450e-02f, -8.24086510e-06f },
        { -2.95799859e-02f, -5.39318882e-02f, -1.74219604e-04f },
        { -2.92359516e-02f, -1.53050944e-02f, -6.30460854e-05f },
        { -5.56493877e-03f, -2.34344602e-02f, -1.26858242e-04f },
        { -4.37593013e-02f, -2.77768299e-02f, -2.70503685e-02f },
        { -1.76926646e-02f, -1.91676542e-02f, -1.15090821e-04f },
        { -8.34268332e-03f, -3.99775570e-03f, -3.27481248e-05f },
        { -3.40162888e-02f, -2.81868968e-02f, -1.96679524e-04f },
        { -2.91855410e-02f, -3.97511162e-02f, -2.81230678e-05f },
        { -1.50395725e-02f, -2.49494594e-02f, -9.42573533e-05f },
        { -1.67938769e-02f, -2.00953931e-02f, -4.00750607e-04f },
        { -1.86435618e-02f, -2.48535164e-02f, -2.74416432e-02f },
        { -4.61211195e-03f, -1.21660791e-02f, -2.93173041e-04f },
        { -4.10017073e-02f, -7.43824020e-02f, -4.42762971e-02f },
        { -1.90370996e-02f, -3.74363363e-02f, -1.34740388e-02f },
    };

    /// <summary>Python: <c>EXPRESSION_MAX</c> (with the leading batch dimension dropped).</summary>
    public static readonly float[,] ExpressionMax =
    {
        { 4.46682945e-02f, 7.08772913e-02f, 4.08344204e-04f },
        { 2.14308221e-02f, 6.15894832e-02f, 4.85319615e-05f },
        { 3.02363783e-02f, 4.45043296e-02f, 1.28298725e-05f },
        { 3.05869691e-02f, 3.79812494e-02f, 6.57040102e-04f },
        { 4.45670523e-02f, 3.97259220e-02f, 7.10966764e-04f },
        { 9.43699256e-02f, 9.85926315e-02f, 2.02551950e-04f },
        { 1.61131397e-02f, 2.92906128e-02f, 3.44733417e-06f },
        { 5.23825921e-02f, 1.07065082e-01f, 6.61510974e-04f },
        { 2.85718683e-03f, 8.32320191e-03f, 2.39314613e-04f },
        { 2.57947259e-02f, 1.60935968e-02f, 2.41853559e-05f },
        { 4.90833223e-02f, 3.43903080e-02f, 3.22353356e-02f },
        { 1.44766076e-02f, 3.39248963e-02f, 1.42291479e-04f },
        { 8.75749043e-04f, 6.82212645e-03f, 2.76097053e-05f },
        { 1.86958015e-02f, 3.84016186e-02f, 7.33085908e-05f },
        { 2.01714113e-02f, 4.90544215e-02f, 2.34028921e-05f },
        { 2.46518422e-02f, 3.29151377e-02f, 3.48571630e-05f },
        { 2.22457591e-02f, 1.21796541e-02f, 1.56396593e-04f },
        { 1.72109623e-02f, 3.01626958e-02f, 1.36556877e-02f },
        { 1.83460284e-02f, 1.61141958e-02f, 2.87440169e-04f },
        { 3.57594155e-02f, 1.80554688e-01f, 2.75554154e-02f },
        { 2.17450950e-02f, 8.66811201e-02f, 3.34241726e-02f },
    };

    /// <summary>Python: <c>limit_expression</c> — elementwise <c>numpy.clip(expression, EXPRESSION_MIN, EXPRESSION_MAX)</c>.</summary>
    public static float[,] LimitExpression(float[,] expression)
    {
        if (expression.GetLength(0) != 21 || expression.GetLength(1) != 3)
        {
            throw new ArgumentException("expression must be (21, 3).", nameof(expression));
        }

        var result = new float[21, 3];
        for (var i = 0; i < 21; i++)
        {
            for (var c = 0; c < 3; c++)
            {
                var value = expression[i, c];
                var min = ExpressionMin[i, c];
                var max = ExpressionMax[i, c];
                result[i, c] = value < min ? min : (value > max ? max : value);
            }
        }

        return result;
    }

    /// <summary>Python: <c>limit_angle</c>.</summary>
    public static (float Pitch, float Yaw, float Roll) LimitAngle(
        float targetPitch, float targetYaw, float targetRoll,
        float outputPitch, float outputYaw, float outputRoll)
    {
        var (pitchMin, pitchMax, yawMin, yawMax, rollMin, rollMax) = CalculateEulerLimits(targetPitch, targetYaw, targetRoll);

        var pitch = outputPitch < pitchMin ? pitchMin : (outputPitch > pitchMax ? pitchMax : outputPitch);
        var yaw = outputYaw < yawMin ? yawMin : (outputYaw > yawMax ? yawMax : outputYaw);
        var roll = outputRoll < rollMin ? rollMin : (outputRoll > rollMax ? rollMax : outputRoll);
        return (pitch, yaw, roll);
    }

    /// <summary>Python: <c>calculate_euler_limits</c>.</summary>
    public static (float PitchMin, float PitchMax, float YawMin, float YawMax, float RollMin, float RollMax) CalculateEulerLimits(
        float pitch, float yaw, float roll)
    {
        var pitchMin = -30.0f;
        var pitchMax = 30.0f;
        var yawMin = -60.0f;
        var yawMax = 60.0f;
        var rollMin = -20.0f;
        var rollMax = 20.0f;

        if (pitch < 0)
        {
            pitchMin = Math.Min(pitch, pitchMin);
        }
        else
        {
            pitchMax = Math.Max(pitch, pitchMax);
        }

        if (yaw < 0)
        {
            yawMin = Math.Min(yaw, yawMin);
        }
        else
        {
            yawMax = Math.Max(yaw, yawMax);
        }

        if (roll < 0)
        {
            rollMin = Math.Min(roll, rollMin);
        }
        else
        {
            rollMax = Math.Max(roll, rollMax);
        }

        return (pitchMin, pitchMax, yawMin, yawMax, rollMin, rollMax);
    }

    /// <summary>
    /// Python: <c>create_rotation</c>. See the class remarks for the verified scipy Euler
    /// convention this reproduces (<c>R = Rz(roll) @ Ry(yaw) @ Rx(pitch)</c>, extrinsic 'xyz',
    /// degrees). Computed in <see cref="double"/> (matching scipy's own float64 internals) and
    /// narrowed to <see cref="float"/> only at the end, matching Python's
    /// <c>rotation.astype(numpy.float32)</c> — the explicit final cast in <c>create_rotation</c>
    /// confirms scipy's own <c>as_matrix()</c> returns float64 regardless of the float32 input
    /// angles.
    /// </summary>
    public static float[,] CreateRotation(float pitch, float yaw, float roll)
    {
        var p = pitch * Math.PI / 180.0;
        var y = yaw * Math.PI / 180.0;
        var r = roll * Math.PI / 180.0;

        var cp = Math.Cos(p);
        var sp = Math.Sin(p);
        var cy = Math.Cos(y);
        var sy = Math.Sin(y);
        var cr = Math.Cos(r);
        var sr = Math.Sin(r);

        // Rx(pitch)
        double[,] rx =
        {
            { 1, 0, 0 },
            { 0, cp, -sp },
            { 0, sp, cp },
        };

        // Ry(yaw)
        double[,] ry =
        {
            { cy, 0, sy },
            { 0, 1, 0 },
            { -sy, 0, cy },
        };

        // Rz(roll)
        double[,] rz =
        {
            { cr, -sr, 0 },
            { sr, cr, 0 },
            { 0, 0, 1 },
        };

        // R = Rz @ Ry @ Rx (extrinsic 'xyz' composition — see class remarks).
        var ryRx = Multiply3X3(ry, rx);
        var rotation = Multiply3X3(rz, ryRx);

        var result = new float[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                result[i, j] = (float)rotation[i, j];
            }
        }

        return result;
    }

    private static double[,] Multiply3X3(double[,] a, double[,] b)
    {
        var result = new double[3, 3];
        for (var i = 0; i < 3; i++)
        {
            for (var j = 0; j < 3; j++)
            {
                double sum = 0;
                for (var k = 0; k < 3; k++)
                {
                    sum += a[i, k] * b[k, j];
                }

                result[i, j] = sum;
            }
        }

        return result;
    }
}
