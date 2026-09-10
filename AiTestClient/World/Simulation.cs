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
    public long OnTrackSteps { get; private set; }
    public long OffTrackSteps { get; private set; }
    public float BestReward { get; private set; }
    public float WorstReward { get; private set; }
    // Rolling window: ON TRACK % covers the last WindowSteps so sustained
    // clean driving can climb back to 100% (a lifetime average never could).
    public const int WindowSteps = 5000;
    private readonly bool[] _window = new bool[WindowSteps];
    private int _windowIdx;
    private int _windowCount;
    private int _windowOnTrack;
    public float OnTrackPct => _windowCount > 0
        ? Math.Clamp(100f * _windowOnTrack / _windowCount, 0f, 100f) : 100f;
    public float WorstOnTrackPct { get; private set; } = 100f;
    public float CurrentLapProgress => Math.Clamp(_lapProgress, 0f, 1f);
    public int Generation { get; private set; } = 1;

    private float _lastProgress;
    private float _lapProgress;
    private float _lastLateral;
    private int _offTrackStreak;
    private const int MaxOffTrackSteps = 90; // auto-recover after ~1.5s off track
    private const int KillGraceSteps = 30; // kill-mode: ~0.5s of off-track learning before the soft reset

    public Simulation(VisionRays? rays = null)
    {
        Track = new Track(5f);
        Brain = new CarBrain(rays ?? AiModel_V1.VisionRays.Default5, 2024, 2, 32);
        var (sx, sz) = Track.CenterAt(0f);
        var (tx, tz) = Track.TangentAt(0f);
        Car = new Car(sx, sz, (float)Math.Atan2(tz, tx));
        _lastProgress = Track.ProgressAt(sx, sz);
        _lapProgress = 0f;
    }

    public void ResetCar(bool resetLaps = true)
    {
        var (sx, sz) = Track.CenterAt(0f);
        var (tx, tz) = Track.TangentAt(0f);
        Car.Reset(sx, sz, (float)Math.Atan2(tz, tx));
        _lastProgress = Track.ProgressAt(sx, sz);
        if (resetLaps) Laps = 0;
        _offTrackStreak = 0;
        OffTrack = false;
    }

    public void RetrainBrain()
    {
        Generation++;
        Brain.Retrain(Environment.TickCount);
        _champion = null; // extinction event: forget the bloodline
        _championFitness = float.NegativeInfinity;
        BestGenDist = WorstGenDist = 0f;
        _hasGenDist = false;
        ChampCount = 0;
        ChampGen = 0;
        TotalReward = 0f;
        Steps = 0;
        Laps = 0;
        OnTrackSteps = OffTrackSteps = 0;
        Array.Clear(_window);
        _windowIdx = _windowCount = _windowOnTrack = 0;
        BestReward = WorstReward = 0f;
        WorstOnTrackPct = 100f;
        _genStartDist = 0f;
        _lapProgress = 0f;
        ResetCar();
    }

    // Champion bloodline for true generational evolution.
    private (float[][][] W, float[][] B)? _champion;
    private float _championFitness = float.NegativeInfinity;
    private float _genStartDist;
    private const float MutationScale = 0.05f;
    /// <summary>Best/worst distance covered in a single generation, in laps (can exceed 1).</summary>
    public float BestGenDist { get; private set; } = 0f;
    public float WorstGenDist { get; private set; } = 0f;
    private bool _hasGenDist;
    /// <summary>How many times a new champion has been crowned.</summary>
    public int ChampCount { get; private set; }
    /// <summary>Generation number that produced the current champion.</summary>
    public int ChampGen { get; private set; }

    /// <summary>
    /// A real generation: score the distance covered since the last restart,
    /// keep the genome if it's the best ever (elitism), otherwise roll back
    /// to the champion with a small mutation — then back to the start line.
    /// </summary>
    private void NewGeneration()
    {
        float fitness = (Laps + _lapProgress) - _genStartDist;
        if (!_hasGenDist || fitness > BestGenDist) BestGenDist = fitness;
        if (!_hasGenDist || fitness < WorstGenDist) WorstGenDist = fitness;
        _hasGenDist = true;
        Generation++;
        if (!_champion.HasValue || fitness > _championFitness)
        {
            _champion = Brain.SnapshotGenome();
            _championFitness = fitness;
            ChampCount++;
            ChampGen = Generation;
        }
        else
        {
            Brain.RestoreGenome(_champion.Value);
            Brain.MutateGenome(MutationScale);
        }
        Brain.ResetLearningState();
        ResetCar(resetLaps: false); // back to the starting line, laps keep counting
        _genStartDist = Laps + _lapProgress;
        _offTrackStreak = 0;
        OffTrack = false;
    }

    /// <summary>
    /// Restart after driving off track with kill mode on. Now a true
    /// generation (see <see cref="NewGeneration"/>): the bloodline keeps its
    /// best genome and the car goes back to the starting line.
    /// </summary>
    private void RestartAfterOffTrack() => NewGeneration();

    public void ReconfigureBrain(int hiddenLayerCount, int hiddenNodeCount)
        => ReconfigureBrain(hiddenLayerCount, hiddenNodeCount, Brain.TrainConfig.Clone());

    public void ReconfigureBrain(int hiddenLayerCount, int hiddenNodeCount, AiModel_V1.TrainingConfig config)
    {
        Brain = new CarBrain(Brain.Rays, Environment.TickCount, hiddenLayerCount, hiddenNodeCount, config);
        _champion = null; // new body, new bloodline
        _championFitness = float.NegativeInfinity;
        BestGenDist = WorstGenDist = 0f;
        _hasGenDist = false;
        ChampCount = 0;
        ChampGen = 0;
        _genStartDist = 0f;
        Inputs = Array.Empty<float>();
        RayDistances = Array.Empty<float>();
        NodeActivations = Array.Empty<float>();
        Steer = Throttle = Brake = Reward = TotalReward = 0f;
        Steps = 0;
        OnTrackSteps = OffTrackSteps = 0;
        Array.Clear(_window);
        _windowIdx = _windowCount = _windowOnTrack = 0;
        BestReward = WorstReward = 0f;
        WorstOnTrackPct = 100f;
        BestLapProgress = 0f;
        _lapProgress = 0f;
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

        // 2) act (NaN firewall: a diverged brain must stall the car, not crash the app)
        (Steer, Throttle, Brake) = Brain.Act(Inputs);
        if (!float.IsFinite(Steer)) Steer = 0f;
        if (!float.IsFinite(Throttle)) Throttle = 0f;
        if (!float.IsFinite(Brake)) Brake = 0f;

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
        // Live record: the current generation can raise BEST DIST mid-run,
        // so it climbs past 100%/200%/300% in real time instead of only
        // updating when the generation dies.
        float liveDist = (Laps + _lapProgress) - _genStartDist;
        if (liveDist > BestGenDist) BestGenDist = liveDist;

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
                     - (OffTrack ? rc.OffTrack : 0f)
                     // Parking ticket: sitting still on track must score worse
                     // than trying and failing, or the brain learns that
                     // holding the brake forever is the safest policy.
                     - (!OffTrack && speed01 < 0.08f ? rc.Parking : 0f)
                     - (Brake > 0.5f && speed01 < 0.15f ? rc.Parking * 0.5f : 0f)
                     // Rejoin bonus: off track, reward closing the distance
                     // back to the road so the brain learns recovery instead
                     // of sightseeing in the grass.
                     + (OffTrack ? Math.Clamp((_lastLateral - lateral) * 2f, -1f, 1f) : 0f);
        _lastLateral = lateral;
        Reward = reward;
        TotalReward += reward;
        if (OffTrack) OffTrackSteps++; else OnTrackSteps++;
        // roll the window: evict the oldest step, record this one
        if (_windowCount == WindowSteps && _window[_windowIdx]) _windowOnTrack--;
        _window[_windowIdx] = !OffTrack;
        if (!OffTrack) _windowOnTrack++;
        _windowIdx = (_windowIdx + 1) % WindowSteps;
        if (_windowCount < WindowSteps) _windowCount++;
        float pct = OnTrackPct;
        if (pct < WorstOnTrackPct) WorstOnTrackPct = pct;
        if (Steps == 0) { BestReward = WorstReward = reward; }
        else { if (reward > BestReward) BestReward = reward; if (reward < WorstReward) WorstReward = reward; }

        // 5) learn
        float corner = Math.Clamp(Math.Abs(headingError), 0f, 1f);
        float targetSteer = Math.Clamp(headingError * 1.4f, -1f, 1f);
        // Fast on straights, lift into corners; brake hard when facing the
        // wrong way at speed.
        float targetThrottle = 0.95f - 0.55f * corner;
        float targetBrake = (corner > 0.45f && speed01 > 0.5f) ? 1f : 0f;
        if (OffTrack)
        {
            // Recovery lesson: steer back toward the nearest road point at
            // moderate speed instead of charging deeper into the grass.
            var (rcx, rcz, _) = Track.SnapToCenterline(Car.X, Car.Z);
            float homeAngle = (float)Math.Atan2(rcz - Car.Z, rcx - Car.X) - Car.Heading;
            homeAngle = (float)Math.Atan2(Math.Sin(homeAngle), Math.Cos(homeAngle));
            targetSteer = Math.Clamp(homeAngle * 2f, -1f, 1f);
            targetThrottle = 0.35f;
            targetBrake = speed01 > 0.4f ? 0.5f : 0f;
        }
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

        // 6) death after the grace period = next generation at the start line
        if (OffTrack)
        {
            _offTrackStreak++;
            if (_offTrackStreak > MaxOffTrackSteps)
            {
                NewGeneration();
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
