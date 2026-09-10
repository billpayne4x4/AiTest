using System;

namespace AiModel_V1;

/// <summary>
/// Modular vision-ray configuration. The car "sees" by casting one ray per
/// entry, at an angle relative to its heading. Change the angles / count /
/// range to reconfigure perception without touching the brain or the network
/// (the network input size is derived from <see cref="Count"/>).
/// </summary>
public class VisionRays
{
    /// <summary>Ray angles in degrees, relative to the car's heading (0 = straight ahead).</summary>
    public float[] AnglesDeg { get; }

    /// <summary>Distance that maps to a normalized input of 1.0 (wall very close).</summary>
    public float MaxDist { get; }

    /// <summary>March step used when casting (smaller = more precise, slower).</summary>
    public float Step { get; }

    public int Count => AnglesDeg.Length;

    public VisionRays(float[] anglesDeg, float maxDist = 30f, float step = 0.5f)
    {
        if (anglesDeg is null || anglesDeg.Length == 0)
            throw new ArgumentException("Need at least one ray.");
        AnglesDeg = (float[])anglesDeg.Clone();
        MaxDist = maxDist;
        Step = step;
    }

    /// <summary>The default 5-ray fan: hard-left, left, ahead, right, hard-right.</summary>
    public static VisionRays Default5 => new(new[] { -90f, -45f, 0f, 45f, 90f }, 30f, 0.5f);

    /// <summary>A wider 7-ray fan for finer perception.</summary>
    public static VisionRays Wide7 => new(new[] { -120f, -75f, -30f, 0f, 30f, 75f, 120f }, 35f, 0.5f);

    public float AngleDeg(int i) => AnglesDeg[i];
}
