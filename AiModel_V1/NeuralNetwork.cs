using System;

namespace AiModel_V1;

/// <summary>
/// Feed-forward network with depth-scaling techniques: residual connections,
/// LayerNorm, ReLU/GELU/tanh activations, He/Xavier/orthogonal init,
/// Adam optimizer, and global gradient clipping. Configured via <see cref="TrainingConfig"/>.
/// </summary>
public class NeuralNetwork
{
    public TrainingConfig Config { get; set; } = new();

    private readonly int[] _sizes;
    private readonly float[][][] _weights;
    private readonly float[][] _biases;
    private readonly float[][] _activations;
    private readonly float[][] _pre;
    private readonly float[][] _normed;
    private readonly float[][] _delta;
    // LayerNorm params per hidden layer (indexed by layer l+1)
    private readonly float[][] _lnGamma;
    private readonly float[][] _lnBeta;
    // Adam state
    private readonly float[][][] _mW;
    private readonly float[][][] _vW;
    private readonly float[][] _mB;
    private readonly float[][] _vB;
    private int _adamStep;
    private Random _rng;

    public NeuralNetwork(int[] sizes, int seed = 1234) : this(sizes, new TrainingConfig(), seed) { }

    public NeuralNetwork(int[] sizes, TrainingConfig config, int seed = 1234)
    {
        if (sizes.Length < 2) throw new ArgumentException("Need at least input and output layers.");
        _sizes = (int[])sizes.Clone();
        Config = config;
        _rng = new Random(seed);
        int L = _sizes.Length - 1;
        _weights = new float[L][][];
        _biases = new float[L][];
        _mW = new float[L][][];
        _vW = new float[L][][];
        _mB = new float[L][];
        _vB = new float[L][];
        for (int l = 0; l < L; l++)
        {
            int fanIn = _sizes[l], fanOut = _sizes[l + 1];
            _weights[l] = new float[fanOut][];
            _mW[l] = new float[fanOut][];
            _vW[l] = new float[fanOut][];
            float scale = InitScale(fanIn, fanOut);
            for (int j = 0; j < fanOut; j++)
            {
                _weights[l][j] = new float[fanIn];
                _mW[l][j] = new float[fanIn];
                _vW[l][j] = new float[fanIn];
                // Near-zero init on the output layer so a fresh brain starts
                // with ~zero steer/throttle instead of a random full-lock
                // bias (which reads as "spinning in a circle").
                float layerScale = l == L - 1 ? scale * 0.1f : scale;
                for (int i = 0; i < fanIn; i++)
                    _weights[l][j][i] = SampleWeight(layerScale, fanIn, fanOut);
            }
            _biases[l] = new float[fanOut];
            _mB[l] = new float[fanOut];
            _vB[l] = new float[fanOut];
        }
        _activations = new float[_sizes.Length][];
        _pre = new float[_sizes.Length][];
        _normed = new float[_sizes.Length][];
        _delta = new float[_sizes.Length][];
        _lnGamma = new float[_sizes.Length][];
        _lnBeta = new float[_sizes.Length][];
        for (int l = 0; l < _sizes.Length; l++)
        {
            _activations[l] = new float[_sizes[l]];
            _pre[l] = new float[_sizes[l]];
            _normed[l] = new float[_sizes[l]];
            _delta[l] = new float[_sizes[l]];
            _lnGamma[l] = new float[_sizes[l]];
            _lnBeta[l] = new float[_sizes[l]];
            for (int j = 0; j < _sizes[l]; j++) _lnGamma[l][j] = 1f;
        }
    }

    public int InputCount => _sizes[0];
    public int OutputCount => _sizes[^1];
    public int LayerCount => _sizes.Length;
    public int[] Sizes => _sizes;

    public float Activation(int layer, int node) => _activations[layer][node];
    public float[] LayerActivations(int layer) => (float[])_activations[layer].Clone();
    public float Weight(int layer, int outNode, int inNode) => _weights[layer][outNode][inNode];
    public float Bias(int layer, int outNode) => _biases[layer][outNode];

    private float InitScale(int fanIn, int fanOut) => Config.Init switch
    {
        1 => (float)Math.Sqrt(2.0 / fanIn),   // He (ReLU)
        2 => (float)Math.Sqrt(1.0 / fanIn),   // orthogonal-ish scaled gaussian
        _ => (float)Math.Sqrt(1.0 / fanIn),   // Xavier
    };

    private float SampleWeight(float scale, int fanIn, int fanOut)
    {
        // Orthogonal approx: gaussian * scale (true QR is overkill here).
        double u1 = Math.Max(double.Epsilon, _rng.NextDouble());
        double u2 = _rng.NextDouble();
        float g = (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        return Config.Init == 0
            ? (float)(_rng.NextDouble() * 2.0 - 1.0) * scale
            : g * scale;
    }

    private float Activate(float z, bool isOutput)
    {
        if (isOutput) return (float)Math.Tanh(z);
        return Config.Activation switch
        {
            1 => Math.Max(0f, z),                                        // ReLU
            2 => 0.5f * z * (1f + (float)Math.Tanh(0.79788458f * (z + 0.044715f * z * z * z))), // GELU approx
            _ => (float)Math.Tanh(z),
        };
    }

    private float ActivateDeriv(float z, float act, bool isOutput)
    {
        if (isOutput) return 1f - act * act;
        return Config.Activation switch
        {
            1 => z > 0f ? 1f : 0f,
            2 => 0.5f * (1f + (float)Math.Tanh(0.79788458f * (z + 0.044715f * z * z * z))), // approx
            _ => 1f - act * act,
        };
    }

    public float[] Forward(float[] input)
    {
        int n = Math.Min(input.Length, _sizes[0]);
        Array.Copy(input, _activations[0], n);
        for (int l = 0; l < _sizes.Length - 1; l++)
        {
            int fanOut = _sizes[l + 1];
            bool isOutput = l == _sizes.Length - 2;
            bool useLn = Config.UseLayerNorm && !isOutput;
            var a = _activations[l];
            // z = W a + b
            for (int j = 0; j < fanOut; j++)
            {
                float z = _biases[l][j];
                var w = _weights[l][j];
                for (int i = 0; i < a.Length; i++) z += w[i] * a[i];
                _pre[l + 1][j] = z;
            }
            // LayerNorm: normalize z across layer, apply gamma/beta
            if (useLn)
            {
                float mean = 0f;
                for (int j = 0; j < fanOut; j++) mean += _pre[l + 1][j];
                mean /= fanOut;
                float var = 0f;
                for (int j = 0; j < fanOut; j++) { float d = _pre[l + 1][j] - mean; var += d * d; }
                var /= fanOut;
                float inv = 1f / (float)Math.Sqrt(var + 1e-5f);
                for (int j = 0; j < fanOut; j++)
                    _normed[l + 1][j] = (_pre[l + 1][j] - mean) * inv * _lnGamma[l + 1][j] + _lnBeta[l + 1][j];
            }
            else Array.Copy(_pre[l + 1], _normed[l + 1], fanOut);

            for (int j = 0; j < fanOut; j++)
                _activations[l + 1][j] = Activate(_normed[l + 1][j], isOutput);

            // Residual: x + F(x) when shapes match (hidden layers only).
            // Scaled by 1/sqrt(2) so variance stays ~constant with depth;
            // unscaled residuals grow activations ~linearly and saturate
            // the outputs (constant full-lock steer = driving in circles).
            if (Config.UseResidual && !isOutput && _sizes[l + 1] == _sizes[l])
                for (int j = 0; j < fanOut; j++) _activations[l + 1][j] = (_activations[l + 1][j] + a[j]) * 0.7071f;

            // Clamp hidden activations so GELU's unbounded positive side
            // can't run away over many generations (pinned-high nodes and
            // all-red saturated weight lines at depth).
            if (!isOutput)
                for (int j = 0; j < fanOut; j++)
                    _activations[l + 1][j] = Math.Clamp(_activations[l + 1][j], -6f, 6f);
        }
        return (float[])_activations[^1].Clone();
    }

    public void PolicyGradientUpdate(float[] outputGradient, float learningRate)
    {
        int last = _sizes.Length - 1;
        if (outputGradient.Length != _sizes[last])
            throw new ArgumentException("Output gradient must match the output layer size.", nameof(outputGradient));
        for (int j = 0; j < _sizes[last]; j++)
            _delta[last][j] = outputGradient[j] * ActivateDeriv(_normed[last][j], _activations[last][j], true);
        BackpropAndUpdate(learningRate);
    }

    public float SupervisedUpdate(float[] input, float[] target, float learningRate)
    {
        Forward(input);
        int last = _sizes.Length - 1;
        float err = 0f;
        for (int j = 0; j < _sizes[last]; j++)
        {
            float d = _activations[last][j] - target[j];
            err += d * d;
            _delta[last][j] = -d * ActivateDeriv(_normed[last][j], _activations[last][j], true);
        }
        BackpropAndUpdate(learningRate);
        return err / _sizes[last];
    }

    private void BackpropAndUpdate(float learningRate)
    {
        _adamStep++; // one optimizer step per update, not per parameter
        int L = _sizes.Length - 1;
        // Depth scaling: a full-strength update through 32 layers compounds
        // into blowup (all weights driven negative = red lines, dead net).
        learningRate /= (float)Math.Sqrt(L);
        // global grad-norm clipping factor
        float normSq = 0f;
        if (Config.GradClip > 0f)
        {
            for (int j = 0; j < _delta[L].Length; j++) normSq += _delta[L][j] * _delta[L][j];
            float norm = (float)Math.Sqrt(normSq);
            if (norm > Config.GradClip)
            {
                float s = Config.GradClip / norm;
                for (int j = 0; j < _delta[L].Length; j++) _delta[L][j] *= s;
            }
        }
        for (int l = L - 1; l >= 0; l--)
        {
            var w = _weights[l];
            var a = _activations[l];
            var dNext = _delta[l + 1];
            bool isOutput = l == L - 1;
            bool residual = Config.UseResidual && !isOutput && _sizes[l + 1] == _sizes[l];

            for (int i = 0; i < _delta[l].Length; i++) _delta[l][i] = 0f;
            for (int j = 0; j < dNext.Length; j++)
            {
                float dj = dNext[j];
                if (dj == 0f) continue;
                var wij = w[j];
                for (int i = 0; i < a.Length; i++)
                {
                    float wOld = wij[i];
                    float grad = dj * a[i];
                    wij[i] += ApplyUpdate(l, j, i, grad, learningRate, true);
                    _delta[l][i] += wOld * dj;
                }
                if (residual && j < _delta[l].Length) _delta[l][j] += dj * 0.7071f; // skip path (matches scaled forward)
            }
            for (int i = 0; i < _delta[l].Length; i++)
                _delta[l][i] *= ActivateDeriv(_normed[l][i], _activations[l][i], false);
            for (int j = 0; j < _biases[l].Length; j++)
                _biases[l][j] += ApplyUpdate(l, j, 0, dNext[j], learningRate, false);
        }
    }

    private float ApplyUpdate(int l, int j, int i, float grad, float lr, bool isWeight)
    {
        // Tiny weight decay pulls weights back toward zero so they can't
        // drift to saturation over many generations at depth.
        const float decay = 1e-4f;
        if (isWeight) grad += decay * _weights[l][j][i];
        if (!Config.UseAdam) return lr * grad;
        const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
        float t = _adamStep;
        float bc1 = 1f - (float)Math.Pow(b1, t), bc2 = 1f - (float)Math.Pow(b2, t);
        if (isWeight)
        {
            _mW[l][j][i] = b1 * _mW[l][j][i] + (1f - b1) * grad;
            _vW[l][j][i] = b2 * _vW[l][j][i] + (1f - b2) * grad * grad;
            float mh = _mW[l][j][i] / bc1, vh = _vW[l][j][i] / bc2;
            return lr * mh / ((float)Math.Sqrt(vh) + eps);
        }
        _mB[l][j] = b1 * _mB[l][j] + (1f - b1) * grad;
        _vB[l][j] = b2 * _vB[l][j] + (1f - b2) * grad * grad;
        float mhb = _mB[l][j] / bc1, vhb = _vB[l][j] / bc2;
        return lr * mhb / ((float)Math.Sqrt(vhb) + eps);
    }

    public void Reinitialize(int seed) => Reinitialize(seed, Config);

    public void Reinitialize(int seed, TrainingConfig config)
    {
        Config = config;
        _rng = new Random(seed);
        _adamStep = 0;
        int L = _sizes.Length - 1;
        for (int l = 0; l < L; l++)
        {
            int fanIn = _sizes[l], fanOut = _sizes[l + 1];
            float scale = InitScale(fanIn, fanOut);
            for (int j = 0; j < fanOut; j++)
            {
                for (int i = 0; i < fanIn; i++)
                {
                    _weights[l][j][i] = SampleWeight(scale, fanIn, fanOut);
                    _mW[l][j][i] = _vW[l][j][i] = 0f;
                }
                _mB[l][j] = _vB[l][j] = 0f;
            }
            for (int j = 0; j < fanOut; j++) _biases[l][j] = 0f;
        }
    }
}
