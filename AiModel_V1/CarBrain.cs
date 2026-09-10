using System;

namespace AiModel_V1;

/// <summary>
/// The AI brain — a self-contained, world-agnostic learning model.
///
/// It takes normalized sensor inputs (one value per vision ray, 0..1) and
/// produces a (steer, throttle) action, and learns via policy gradient from a
/// scalar reward. It has no knowledge of the track or the car, which keeps the
/// AI model reusable and decoupled from the game world (which lives in the
/// client project).
///
/// The ray layout is supplied by a <see cref="VisionRays"/> config, so perception
/// is modular: swap in <see cref="VisionRays.Default5"/>, <see cref="VisionRays.Wide7"/>,
/// or a custom set and the network adapts its input size automatically.
/// </summary>
public class CarBrain
{
    public VisionRays Rays { get; }
    public NeuralNetwork Net { get; }
    public TrainingConfig TrainConfig { get; }

    // learning / perception tuning
    private const float LearningRate = 0.008f;
    private const float GuidanceLearningRate = 0.012f;
    private const float BaselineAlpha = 0.03f;   // how fast the reward baseline adapts
    private const float Exploration = 0.12f;      // action noise for exploration
    private const float PpoClip = 0.2f;

    private float _baseline;
    private float _value; // critic: running value estimate for PPO advantage
    private readonly Random _rng;
    private readonly float[] _lastExploration = new float[2];

    public CarBrain(VisionRays? rays = null, int seed = 2024, int hiddenLayerCount = 1, int hiddenNodeCount = 10)
        : this(rays, seed, hiddenLayerCount, hiddenNodeCount, new TrainingConfig()) { }

    public CarBrain(VisionRays? rays, int seed, int hiddenLayerCount, int hiddenNodeCount, TrainingConfig config)
    {
        Rays = rays ?? VisionRays.Default5;
        if (hiddenLayerCount < 1) throw new ArgumentOutOfRangeException(nameof(hiddenLayerCount));
        if (hiddenNodeCount < 1) throw new ArgumentOutOfRangeException(nameof(hiddenNodeCount));
        var sizes = new int[checked(hiddenLayerCount + 2)];
        // Vision alone is symmetric: the same road looks valid in both travel
        // directions. Two guidance inputs encode sin/cos of the desired-track
        // heading error so the policy can learn which way is forward.
        sizes[0] = Rays.Count + 2;
        for (int i = 0; i < hiddenLayerCount; i++) sizes[i + 1] = hiddenNodeCount;
        sizes[^1] = 2;
        TrainConfig = config;
        Net = new NeuralNetwork(sizes, config, seed);
        _rng = new Random(seed + 7);
    }

    public float Baseline => _baseline;

    /// <summary>
    /// Convert raw ray distances (world units) to normalized 0..1 inputs.
    /// 1.0 = wall very close, 0.0 = clear / at max distance.
    /// </summary>
    public float[] Normalize(float[] rawDistances, float headingErrorSin = 0f, float headingErrorCos = 1f)
    {
        var inputs = new float[Rays.Count + 2];
        for (int i = 0; i < Rays.Count; i++)
        {
            float d = rawDistances[i];
            inputs[i] = Math.Clamp(1f - d / Rays.MaxDist, 0f, 1f);
        }
        inputs[Rays.Count] = Math.Clamp(headingErrorSin, -1f, 1f);
        inputs[Rays.Count + 1] = Math.Clamp(headingErrorCos, -1f, 1f);
        return inputs;
    }

    /// <summary>Produce the (steer, throttle) action for the given normalized inputs.</summary>
    public (float steer, float throttle) Act(float[] inputs, bool addNoise = true)
    {
        var outp = Net.Forward(inputs);
        float steerOutput = outp[0];
        float throttleOutput = outp[1];
        if (addNoise)
        {
            _lastExploration[0] = NextGaussian();
            _lastExploration[1] = NextGaussian();
            steerOutput += _lastExploration[0] * Exploration;
            throttleOutput += _lastExploration[1] * Exploration;
        }
        else Array.Clear(_lastExploration);

        float steer = steerOutput;                              // -1..1
        float throttle = (throttleOutput + 1f) * 0.5f;          // 0..1
        steer = Math.Clamp(steer, -1f, 1f);
        throttle = Math.Clamp(throttle, 0f, 1f);
        return (steer, throttle);
    }

    /// <summary>Learn from this step's reward (PPO-clipped advantage + entropy bonus, or plain policy gradient).</summary>
    public void Learn(float reward)
    {
        float advantage;
        if (TrainConfig.UsePpo)
        {
            // Critic update + clipped surrogate-style advantage.
            float raw = reward - _value;
            _value += BaselineAlpha * raw;
            float scale = 1f + Math.Abs(_value);
            advantage = Math.Clamp(raw / scale, -PpoClip, PpoClip) * scale;
        }
        else
        {
            advantage = reward - _baseline;
            _baseline += BaselineAlpha * (reward - _baseline);
            advantage = Math.Clamp(advantage, -5f, 5f);
        }
        // REINFORCE/ES estimate: correlate the advantage with the independent
        // perturbation applied to each action. The old implementation sent the
        // same positive gradient to steer and throttle, inevitably saturating
        // steering at +1 and making the car circle.
        Net.PolicyGradientUpdate(new[]
        {
            advantage * _lastExploration[0] + TrainConfig.EntropyBonus * NextGaussian(),
            advantage * _lastExploration[1] + TrainConfig.EntropyBonus * NextGaussian()
        }, LearningRate);
    }

    public void LearnGuided(float reward, float[] inputs, float targetSteer, float targetThrottle)
    {
        Learn(reward);
        Net.SupervisedUpdate(inputs, new[]
        {
            Math.Clamp(targetSteer, -1f, 1f),
            Math.Clamp(targetThrottle * 2f - 1f, -1f, 1f)
        }, GuidanceLearningRate);
    }

    public void Retrain(int seed)
    {
        Net.Reinitialize(seed);
        _baseline = 0f;
    }

    /// <summary>
    /// Reset the learning state (reward baseline + pending exploration noise)
    /// without wiping the network weights. Used for soft restarts where the
    /// car is put back on track but keeps what it learned.
    /// </summary>
    public void ResetLearningState()
    {
        _baseline = 0f;
        Array.Clear(_lastExploration);
    }

    private float NextGaussian()
    {
        double u1 = Math.Max(double.Epsilon, _rng.NextDouble());
        double u2 = _rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
