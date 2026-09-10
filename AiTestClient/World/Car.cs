using System;

namespace AiTestClient.World;

/// <summary>
/// Simple top-down car physics: a heading angle, a speed, and forward motion.
/// Steering changes the heading; throttle accelerates toward a target speed.
/// </summary>
public class Car
{
    public const float MaxSpeed = 16f;
    public const float Accel = 12f;
    public const float Brake = 20f;
    public const float Friction = 5f;
    public const float SteerRate = 2.4f; // rad/s at full lock

    public float X, Z;
    public float Heading; // radians; 0 = +X, increases toward +Z
    public float Speed;

    public Car(float x, float z, float heading)
    {
        X = x; Z = z; Heading = heading;
    }

    public void Reset(float x, float z, float heading)
    {
        X = x; Z = z; Heading = heading; Speed = 0f;
    }

    public (float dx, float dz) ForwardDir()
        => ((float)Math.Cos(Heading), (float)Math.Sin(Heading));

    public (float dx, float dz) RightDir()
        => ((float)-Math.Sin(Heading), (float)Math.Cos(Heading));

    /// <param name="steer">-1 (full left) .. +1 (full right)</param>
    /// <param name="throttle">0 .. 1</param>
    public void Apply(float steer, float throttle, float dt)
    {
        // steering (only effective while moving)
        float speedFactor = Math.Min(1f, Math.Abs(Speed) / 3f);
        Heading += steer * SteerRate * speedFactor * dt;

        // longitudinal
        float target = throttle * MaxSpeed;
        if (Speed < target) Speed = Math.Min(target, Speed + Accel * dt);
        else Speed = Math.Max(target, Speed - (Brake * throttle + Friction) * dt);

        // integrate position
        var (dx, dz) = ForwardDir();
        X += dx * Speed * dt;
        Z += dz * Speed * dt;
    }
}
