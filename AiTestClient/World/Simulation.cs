using System;
using AiModel_V1;

namespace AiTestClient.World;

/// <summary>
/// Ties the track, car, and AI brain together. Each <see cref="Step"/> advances
/// the world by one fixed tick: sense -> act -> move -> reward -> learn.
///
/// The ray casting lives here (in the world), not in the AI model: we cast the
/// brain's configured rays against the track, normalize the distances, and feed
/// them to the brain. This keeps the AI model decoupled from the world.
///
/// Reward shaping (what the car is learning to maximize):
///   + strong bonus for staying near the centerline
///   + bonus for forward progress around the track
///   + small bonus for speed
///   - large penalty when off the track
/// </summary>
public class Simulation
{
    public const float Dt = 1f / 60f;

    public Track Track { get; }
    public Car Car { get; }
    public CarBrain Brain { get; private set; }
    public PhysicsConfig PhysicsCfg { get; set; } = new();
    public RewardConfig RewardCfg { get; set; } = new();

    // latest step outputs (for the UI)
    public float[] Inputs { get; private set; } = Array.Empty<float>();
    public float[] RayDistances { get; private set; } = Array.Empty<float>();
    public float Steer { get; private set; }
    public float Throttle { get; private set; }
    public float Brake { get; private set; }
    public float Reward { get; private set; }
    public float[] NodeActivations { get; private set; } = Array.Empty<float>(); // flattened, layer-major
    public bool OffTrack { get; private set; }
    public bool NewGenerationOnOffTrack { get; set; }

    // stats
    public long Steps { get; private set; }
    public int Laps { get; private set; }
    public float BestLapProgress { get; private set; }
    public float TotalReward { get; private set; }
    public float CurrentLapProgress => Math.Clamp(_lapProgress, 0f, 1f);
    public int Generation { get; private set; } = 1;

    private float _lastProgress;
    private float _lapProgress;
    private int _offTrackStreak;
    private const int MaxOffTrackSteps = 90; // auto-recover after ~1.5s off track
    private const int KillGraceSteps = 30; // kill-mode: ~0.5s of off-track learning before the soft reset

    public Simulation(VisionRays? rays = null)
    {
        Track = new Track(5f);
        Brain = new CarBrain(rays, 2024, 32, 16);
        var (sx, sz) = Track.CenterAt(0f);
        var (tx, tz) = Track.TangentAt(0f);
        Car = new Car(sx, sz, (float)Math.Atan2(tz, tx));
        _lastProgress = Track.ProgressAt(sx, sz);
        _lapProgress = 0f;
    }

    public void ResetCar()
    {
        var (sx, sz) = Track.CenterAt(0f);
        var (tx, tz) = Track.TangentAt(0f);
        Car.Reset(sx, sz, (float)Math.Atan2(tz, tx));
        _lastProgress = Track.ProgressAt(sx, sz);
        Laps = 0;
        _offTrackStreak = 0;
        OffTrack = false;
    }

    public void RetrainBrain()
    {
        Generation++;
        Brain.Retrain(Environment.TickCount);
        TotalReward = 0f;
        Steps = 0;
        Laps = 0;
        ResetCar();
    }

    /// <summary>
    /// Restart after driving off track with kill mode on: the car is placed
    /// back on the centerline <i>where it went off</i> (not back at the start),
    /// stopped, with the reward baseline reset — but the brain keeps its
    /// weights and all stats (steps, reward, laps) keep accumulating, so lap
    /// flow and learning continue across the restart. By the time this runs,
    /// the brain has already absorbed <see cref="KillGraceSteps"/> off-track
    /// steps of -12 penalty + guided correction, so the mistake was learned
    /// from instead of being wiped by <see cref="CarBrain.Retrain"/>.
    /// (Teleporting to the start instead would force a full re-lap per mistake
    /// and reset the lap counter, which is why kill mode used to look stuck.)
    /// </summary>
    private void RestartAfterOffTrack()
    {
        Generation++;
        Brain.ResetLearningState();
        var (cx, cz, heading) = Track.SnapToCenterline(Car.X, Car.Z);
        Car.Reset(cx, cz, heading);
        _lastProgress = Track.ProgressAt(cx, cz);
        _offTrackStreak = 0;
        OffTrack = false;
    }

    public void ReconfigureBrain(int hiddenLayerCount, int hiddenNodeCount)
        => ReconfigureBrain(hiddenLayerCount, hiddenNodeCount, Brain.TrainConfig.Clone());

    public void ReconfigureBrain(int hiddenLayerCount, int hiddenNodeCount, AiModel_V1.TrainingConfig config)
    {
        Brain = new CarBrain(Brain.Rays, Environment.TickCount, hiddenLayerCount, hiddenNodeCount, config);
        Inputs = Array.Empty<float>();
        RayDistances = Array.Empty<float>();
        NodeActivations = Array.Empty<float>();
        Steer = Throttle = Reward = TotalReward = 0f;
        Steps = 0;
        BestLapProgress = 0f;
        ResetCar();
    }

    public void Step()
    {
        // 1) sense: cast the brain's rays against the track
        int rayCount = Brain.Rays.Count;
        RayDistances = new float[rayCount];
        for (int i = 0; i < rayCount; i++)
        {
            double a = Car.Heading + Math.PI * Brain.Rays.AngleDeg(i) / 180.0;
            float dx = (float)Math.Cos(a), dz = (float)Math.Sin(a);
            RayDistances[i] = Track.CastRay(Car.X, Car.Z, dx, dz, Brain.Rays.MaxDist, Brain.Rays.Step);
        }
        var (_, _, desiredHeading) = Track.SnapToCenterline(Car.X, Car.Z);
        float headingError = desiredHeading - Car.Heading;
        headingError = (float)Math.Atan2(Math.Sin(headingError), Math.Cos(headingError));
        Inputs = Brain.Normalize(RayDistances, (float)Math.Sin(headingError), (float)Math.Cos(headingError));

        // 2) act
        (Steer, Throttle, Brake) = Brain.Act(Inputs);

        // 3) move
        Car.Apply(Steer, Throttle, Brake, Dt, PhysicsCfg);

        // 4) reward
        OffTrack = !Track.IsOnTrack(Car.X, Car.Z);
        float lateral = Track.LateralOffset(Car.X, Car.Z);
        float centering = 1f - Math.Clamp(lateral / Track.HalfWidth, 0f, 1f);

        float progress = Track.ProgressAt(Car.X, Car.Z);
        float dProg = progress - _lastProgress;
        if (dProg > 0.5f) dProg -= 1f;
        if (dProg < -0.5f) dProg += 1f;
        _lapProgress += dProg;
        if (_lapProgress >= 1f)
        {
            Laps++;
            _lapProgress -= 1f;
        }
        else if (_lapProgress < 0f)
        {
            _lapProgress = 0f;
        }
        _lastProgress = progress;
        if (progress > BestLapProgress) BestLapProgress = progress;

        var (trackTx, trackTz) = Track.TangentAt(progress);
        var (carFx, carFz) = Car.ForwardDir();
        float alignment = carFx * trackTx + carFz * trackTz;
        float speed01 = Car.Speed / Car.MaxSpeed;
        float slip01 = Math.Clamp(Car.LateralSlip / PhysicsCfg.SpinThreshold, 0f, 1f);
        // All shaping weights live in RewardCfg (rewards menu, W).
        var rc = RewardCfg;
        float reward = rc.Speed * speed01
                     + rc.Centering * centering * speed01
                     + rc.Alignment * alignment * speed01
                     + rc.Progress * dProg
                     - (alignment < 0f ? rc.WrongWay * -alignment * speed01 : 0f)
                     - rc.Slide * slip01
                     - (Car.Spinning ? rc.Spin : 0f)
                     - (Car.Understeering ? rc.Understeer : 0f)
                     - rc.SteerEffort * Math.Abs(Steer)
                     - (OffTrack ? rc.OffTrack : 0f);
        Reward = reward;
        TotalReward += reward;

        // 5) learn
        float corner = Math.Clamp(Math.Abs(headingError), 0f, 1f);
        float targetSteer = Math.Clamp(headingError * 1.4f, -1f, 1f);
        // Fast on straights, lift into corners; brake hard when facing the
        // wrong way at speed.
        float targetThrottle = 0.95f - 0.55f * corner;
        float targetBrake = (corner > 0.45f && speed01 > 0.5f) ? 1f : 0f;
        Brain.LearnGuided(reward, Inputs, targetSteer, targetThrottle, targetBrake);

        // capture node activations for the UI (flattened, layer-major)
        CaptureActivations();

        Steps++;

        // Kill mode: give the brain a grace period of off-track steps first so
        // each of them applies its -12 penalty + guided correction via
        // LearnGuided above, then do a SOFT restart that keeps the weights.
        if (NewGenerationOnOffTrack)
        {
            if (OffTrack)
            {
                _offTrackStreak++;
                if (_offTrackStreak >= KillGraceSteps)
                {
                    RestartAfterOffTrack();
                    return;
                }
            }
            else
            {
                _offTrackStreak = 0;
            }
            return;
        }

        // 6) auto-recover if stuck off track
        if (OffTrack)
        {
            _offTrackStreak++;
            if (_offTrackStreak > MaxOffTrackSteps)
            {
                var (cx, cz, heading) = Track.SnapToCenterline(Car.X, Car.Z);
                Car.Reset(cx, cz, heading);
                _lastProgress = Track.ProgressAt(cx, cz);
                _offTrackStreak = 0;
            }
        }
        else
        {
            _offTrackStreak = 0;
        }
    }

    private void CaptureActivations()
    {
        int total = 0;
        foreach (var s in Brain.Net.Sizes) total += s;
        NodeActivations = new float[total];
        int idx = 0;
        for (int l = 0; l < Brain.Net.LayerCount; l++)
            for (int n = 0; n < Brain.Net.Sizes[l]; n++)
                NodeActivations[idx++] = Brain.Net.Activation(l, n);
    }
}
