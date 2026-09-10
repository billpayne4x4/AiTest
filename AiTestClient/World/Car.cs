using System;

namespace AiTestClient.World;

/// <summary>
/// Kinematic-bicycle car physics with a friction-circle grip cap.
///
/// How it drives: yaw follows the classic bicycle formula
/// (yaw = speed / wheelbase * tan(steerAngle)), so normal cornering is
/// planted with essentially zero slide. Slide only appears when the demanded
/// lateral acceleration exceeds what the tires can hold (friction circle,
/// with mild downforce growth at speed): then the car drifts wide
/// (understeer) or, if the rear is provoked hard, steps out (oversteer) and
/// can spin. Longitudinal grip is shared with cornering, so flooring it
/// mid-corner or braking late genuinely upsets the car.
/// </summary>
public class Car
{
    public const float MaxSpeed = 24f;
    public const float Accel = 16f;
    public const float Brake = 34f;
    public const float Drag = 0.012f;      // quadratic aero factor
    public const float RollResistance = 1.2f;
    public const float WheelBase = 2.8f;
    public const float MaxSteerAngle = 0.55f; // rad at the wheels
    public const float GripBase = 26f;        // base lateral accel limit (m/s^2)
    public const float Downforce = 0.35f;     // extra grip per m/s of speed
    public const float SpinThreshold = 11f;   // lateral speed that triggers a spin

    public float X, Z;
    public float Heading; // radians; 0 = +X, increases toward +Z
    public float Speed;   // forward component (for HUD/reward)
    public float YawRate;
    public float SlipAngle;  // radians between velocity and heading
    public float LateralSlip; // lateral speed magnitude
    public bool Spinning;
    public bool Understeering;

    private float _vx, _vz;

    public Car(float x, float z, float heading)
    {
        X = x; Z = z; Heading = heading;
    }

    public void Reset(float x, float z, float heading)
    {
        X = x; Z = z; Heading = heading; Speed = 0f;
        YawRate = 0f; SlipAngle = 0f; LateralSlip = 0f;
        Spinning = false; Understeering = false;
        _vx = _vz = 0f;
    }

    public (float dx, float dz) ForwardDir()
        => ((float)Math.Cos(Heading), (float)Math.Sin(Heading));

    public (float dx, float dz) RightDir()
        => ((float)-Math.Sin(Heading), (float)Math.Cos(Heading));

    /// <param name="steer">-1 (full left) .. +1 (full right)</param>
    /// <param name="throttle">0 .. 1</param>
    /// <param name="brake">0 .. 1</param>
    public void Apply(float steer, float throttle, float brake, float dt, PhysicsConfig phys)
    {
        if (!phys.Realistic)
        {
            // Arcade fallback: heading follows steer, no slide model.
            float speedFactor = Math.Min(1f, Math.Abs(Speed) / 3f);
            Heading += steer * 2.6f * speedFactor * dt;
            var (adx, adz) = ForwardDir();
            float target = throttle * MaxSpeed;
            if (Speed < target) Speed = Math.Min(target, Speed + Accel * dt);
            else Speed = Math.Max(target, Speed - (Brake * throttle + RollResistance) * dt);
            _vx = adx * Speed; _vz = adz * Speed;
            X += _vx * dt; Z += _vz * dt;
            YawRate = 0f; SlipAngle = 0f; LateralSlip = 0f;
            Spinning = false; Understeering = false;
            return;
        }

        var (fx, fz) = ForwardDir();
        var (rx, rz) = RightDir();
        float vf = _vx * fx + _vz * fz; // forward speed
        float vl = _vx * rx + _vz * rz; // lateral (slide) speed

        // --- bicycle yaw demand ---
        float steerAngle = steer * phys.Steering;
        float yawDemand = vf / WheelBase * (float)Math.Tan(steerAngle);

        // --- friction circle: tires can only hold so much lateral accel ---
        float longDemand = throttle * Accel + brake * Brake;
        float maxLat = phys.Grip + phys.Downforce * Math.Abs(vf);
        float availLat = (float)Math.Sqrt(Math.Max(0f, maxLat * maxLat - Math.Min(longDemand, maxLat) * Math.Min(longDemand, maxLat) * 0.35f));
        float latDemand = Math.Abs(yawDemand * Math.Max(Math.Abs(vf), 0.1f));

        Understeering = phys.Understeer && latDemand > availLat && Math.Abs(steer) > 0.25f && Math.Abs(vf) > 6f;
        if (Understeering)
        {
            // Plow: cap yaw at what the front tires can hold.
            float cap = availLat / Math.Max(Math.Abs(vf), 0.1f) * Math.Sign(yawDemand);
            yawDemand = cap;
        }
        YawRate += (yawDemand - YawRate) * Math.Min(1f, 12f * dt);
        Heading += YawRate * dt;

        // --- lateral grip: kill slide hard while inside the circle ---
        (fx, fz) = ForwardDir();
        (rx, rz) = RightDir();
        float vf2 = _vx * fx + _vz * fz;
        float vl2 = _vx * rx + _vz * rz;
        float overLimit = Math.Max(0f, latDemand - availLat);
        if (overLimit <= 0f)
        {
            // Planted: lateral velocity decays almost instantly.
            vl2 *= (float)Math.Exp(-30f * dt);
        }
        else
        {
            // Past the limit: the excess becomes real sliding drift,
            // scaled by the Oversteer setting (0 = planted, 2 = drifty).
            float slideRate = Math.Min(1f, overLimit / maxLat) * phys.Oversteer;
            vl2 = vl2 * (float)Math.Exp(-6f * dt) + Math.Sign(steer != 0f ? steer : vl2) * slideRate * Math.Abs(vf2) * 0.35f * Math.Min(1f, 4f * dt);
        }

        // --- spin-out: huge slide kicks the yaw and scrubs speed ---
        Spinning = Math.Abs(vl2) > phys.SpinThreshold;
        if (Spinning)
        {
            YawRate += Math.Sign(vl2) * 5f * dt;
            vf2 *= 1f - 1.5f * dt;
            vl2 *= 1f - 1.2f * dt;
        }

        // --- longitudinal: engine (fading at speed) vs brakes vs aero ---
        float engine = throttle * Accel * Math.Max(0.35f, 1f - Math.Max(0f, vf2) / (MaxSpeed * 1.4f));
        vf2 += engine * dt;
        vf2 -= Math.Sign(vf2) * Math.Min(Math.Abs(vf2), brake * Brake * dt);
        vf2 -= vf2 * Math.Abs(vf2) * Drag * dt;
        vf2 -= Math.Sign(vf2) * Math.Min(Math.Abs(vf2), RollResistance * dt);
        vf2 = Math.Clamp(vf2, -4f, MaxSpeed);

        _vx = fx * vf2 + rx * vl2;
        _vz = fz * vf2 + rz * vl2;
        X += _vx * dt;
        Z += _vz * dt;

        Speed = vf2;
        LateralSlip = Math.Abs(vl2);
        float spd = Math.Max(1f, (float)Math.Sqrt(_vx * _vx + _vz * _vz));
        SlipAngle = (float)Math.Asin(Math.Clamp(vl2 / spd, -1f, 1f));
    }
}
