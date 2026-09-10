using System;
using AiModel_V1.Gpu;

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

    // ---- GPU compute (zero-dep CUDA backend; null = CPU) ----
    // The CPU arrays are always the readable master copy: activations stream
    // down from VRAM after every GPU forward (so the brain HUD keeps working)
    // and weights sync both ways, making CPU -> GPU -> CPU seamless.
    private GpuContext? _gpu;
    private bool _gpuWeightsDirty; // VRAM holds newer weights than _weights
    private string _deviceNote = "CPU";

    /// <summary>Human-readable compute device for the status HUD.</summary>
    public string DeviceLabel => _gpu != null ? "GPU " + _gpu.DeviceName : _deviceNote;
    public bool IsGpu => _gpu != null;
    /// <summary>Our VRAM footprint in MB (0 on CPU).</summary>
    public double GpuVramMb => _gpu != null ? _gpu.VramBytes / 1000000.0 : 0.0;

    /// <summary>
    /// Move all weights to VRAM and run subsequent math on the GPU.
    /// Safe to call anytime; fails gracefully with a reason message.
    /// </summary>
    public bool TryEnableGpu(out string message)
    {
        if (_gpu != null) { message = DeviceLabel; return true; }
        int maxW = 0;
        foreach (var s in _sizes) maxW = Math.Max(maxW, s);
        if (maxW > 1024) { message = "GPU refused: layer wider than 1024"; _deviceNote = "CPU"; return false; }
        // No layer-count cap: the kernels loop over layers dynamically, so any
        // depth the CPU can hold also runs on the GPU.
        GpuContext? ctx = null;
        try
        {
            ctx = GpuContext.Create(_sizes, Config, out _);
            ctx.UploadWeights(_weights, _biases);
            _gpu = ctx;
            _gpuWeightsDirty = false;
            message = DeviceLabel;
            return true;
        }
        catch (Exception ex)
        {
            try { ctx?.Dispose(); } catch { }
            message = "GPU unavailable: " + FriendlyCudaReason(ex);
            _deviceNote = "CPU";
            return false;
        }
    }

    private static string FriendlyCudaReason(Exception ex)
    {
        if (ex is DllNotFoundException)
            return "no CUDA driver found (needs NVIDIA GPU + driver)";
        return ex.GetType().Name + ": " + ex.Message;
    }

    /// <summary>Move weights back to RAM and resume CPU math. Seamless.</summary>
    public void DisableGpu()
    {
        if (_gpu == null) return;
        try { EnsureWeightsSynced(); } catch { /* keep CPU copy on failure */ }
        try { _gpu.Dispose(); } catch { }
        _gpu = null;
        _deviceNote = "CPU";
    }

    /// <summary>Download VRAM weights into the CPU master copy if stale.</summary>
    private void EnsureWeightsSynced()
    {
        if (_gpu != null && _gpuWeightsDirty)
        {
            _gpu.DownloadWeights(_weights, _biases);
            _gpuWeightsDirty = false;
        }
    }

    /// <summary>Push the CPU master copy to VRAM (after mutate/restore/reinit).</summary>
    private void PushWeightsToGpu()
    {
        if (_gpu == null) return;
        _gpu.UploadWeights(_weights, _biases);
        _gpuWeightsDirty = false;
    }

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
    public float Weight(int layer, int outNode, int inNode) { EnsureWeightsSynced(); return _weights[layer][outNode][inNode]; }
    public float Bias(int layer, int outNode) { EnsureWeightsSynced(); return _biases[layer][outNode]; }

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
        if (_gpu != null) return GpuForward(input);
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

            if (isOutput)
            {
                // Temperature scaling: keep the outputs in tanh's linear
                // range so large fan-in can't pin the pedals at an extreme.
                float temp = (float)Math.Sqrt(Math.Max(1, _sizes[l]));
                for (int j = 0; j < fanOut; j++) _normed[l + 1][j] /= temp;
            }

            for (int j = 0; j < fanOut; j++)
                _activations[l + 1][j] = Activate(_normed[l + 1][j], isOutput);

            // Residual: x + F(x) when shapes match (hidden layers only),
            // unscaled (standard ResNet form). LayerNorm above already keeps
            // variance stable, so no 1/sqrt(2) downscaling: that shrinks the
            // signal by 0.7071 per layer (~1e-5 over 32 layers), causing
            // vanishing activations followed by saturating blowup.
            if (Config.UseResidual && !isOutput && _sizes[l + 1] == _sizes[l])
                for (int j = 0; j < fanOut; j++) _activations[l + 1][j] += a[j];

            // Clamp hidden activations so GELU's unbounded positive side
            // can't run away over many generations (pinned-high nodes and
            // all-red saturated weight lines at depth).
            if (!isOutput)
                for (int j = 0; j < fanOut; j++)
                    _activations[l + 1][j] = Math.Clamp(_activations[l + 1][j], -6f, 6f);
        }
        return (float[])_activations[^1].Clone();
    }

    /// <summary>GPU forward: inputs up, kernels run, activations stream down.</summary>
    private float[] GpuForward(float[] input)
    {
        var gpu = _gpu!;
        // NOTE: never push here — VRAM is authoritative; the CPU copy may be
        // stale (flagged dirty) and pushing it would clobber trained weights.
        int n = Math.Min(input.Length, _sizes[0]);
        Array.Copy(input, _activations[0], n);
        gpu.UploadInputs(input, n);
        float temp = (float)Math.Sqrt(Math.Max(1, _sizes[^2]));
        gpu.LaunchForward(Config, temp);
        gpu.DownloadActs(_activations, _normed);
        return (float[])_activations[^1].Clone();
    }

    /// <summary>Shared host-side update prep (adam step, depth scale, clip).</summary>
    private float[] PrepareDelta(float[] outputGradient, float learningRate, out float lr, out float bc1, out float bc2)
    {
        _adamStep++;
        int L = _sizes.Length - 1;
        lr = learningRate / (float)Math.Sqrt(L);
        const float b1 = 0.9f, b2 = 0.999f;
        float t = _adamStep;
        bc1 = 1f - (float)Math.Pow(b1, t);
        bc2 = 1f - (float)Math.Pow(b2, t);
        var d = (float[])outputGradient.Clone();
        if (Config.GradClip > 0f)
        {
            float normSq = 0f;
            for (int j = 0; j < d.Length; j++) normSq += d[j] * d[j];
            float norm = (float)Math.Sqrt(normSq);
            if (norm > Config.GradClip)
            {
                float s = Config.GradClip / norm;
                for (int j = 0; j < d.Length; j++) d[j] *= s;
            }
        }
        return d;
    }

    public void PolicyGradientUpdate(float[] outputGradient, float learningRate)
    {
        int last = _sizes.Length - 1;
        if (outputGradient.Length != _sizes[last])
            throw new ArgumentException("Output gradient must match the output layer size.", nameof(outputGradient));
        for (int j = 0; j < _sizes[last]; j++)
            _delta[last][j] = outputGradient[j] * ActivateDeriv(_normed[last][j], _activations[last][j], true);
        if (_gpu != null)
        {
            var d = PrepareDelta(_delta[last], learningRate, out float lr, out float bc1, out float bc2);
            var gpu = _gpu;
            gpu.UploadDelta(d);
            gpu.LaunchBackward(Config, lr, bc1, bc2);
            // VRAM is now newer than the CPU copy — flag it so CPU-side reads
            // (viz, snapshot, toggle-off) download on demand. Do NOT push here:
            // the CPU copy is stale and would clobber trained weights.
            _gpuWeightsDirty = true;
            return;
        }
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
        if (_gpu != null)
        {
            var d = PrepareDelta(_delta[last], learningRate, out float lr, out float bc1, out float bc2);
            var gpu = _gpu;
            gpu.UploadDelta(d);
            gpu.LaunchBackward(Config, lr, bc1, bc2);
            // NOTE: see above — flag dirty for on-demand download, never push.
            _gpuWeightsDirty = true;
            return err / _sizes[last];
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
                if (residual && j < _delta[l].Length) _delta[l][j] += dj; // skip path (identity)
            }
            for (int i = 0; i < _delta[l].Length; i++)
                _delta[l][i] *= ActivateDeriv(_normed[l][i], _activations[l][i], false);
            for (int j = 0; j < _biases[l].Length; j++)
                _biases[l][j] += ApplyUpdate(l, j, 0, dNext[j], learningRate, false);
            // MaxNorm: hard cap per-neuron weight norm so no layer can ever
            // saturate no matter how many generations train.
            for (int j = 0; j < w.Length; j++)
            {
                float n2 = 0f;
                var wij = w[j];
                for (int i = 0; i < wij.Length; i++) n2 += wij[i] * wij[i];
                float n = (float)Math.Sqrt(n2);
                if (n > 3f)
                {
                    float s = 3f / n;
                    for (int i = 0; i < wij.Length; i++) wij[i] *= s;
                }
            }
        }
    }

    private float ApplyUpdate(int l, int j, int i, float grad, float lr, bool isWeight)
    {
        // Tiny weight decay pulls weights back toward zero so they can't
        // drift to saturation over many generations at depth. Biases get a
        // stronger pull: with no decay they drift until the outputs pin at
        // -1 (car presses neither pedal no matter what the inputs say).
        const float decay = 1e-4f;
        const float biasDecay = 1e-3f;
        if (isWeight) grad += decay * _weights[l][j][i];
        else grad += biasDecay * _biases[l][j];
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
        if (_gpu != null)
        {
            // Fresh optimizer state on VRAM to match the fresh weights.
            PushWeightsToGpu();
            try { _gpu.ZeroAdam(); } catch { DisableGpu(); }
        }
    }

    // ---- generational evolution: snapshot / restore / mutate ----
    /// <summary>Deep copy of all weights + biases (a "genome" for elitism).</summary>
    public (float[][][] W, float[][] B) Snapshot()
    {
        EnsureWeightsSynced(); // champion must capture the newest weights
        int L = _sizes.Length - 1;
        var w = new float[L][][];
        var b = new float[L][];
        for (int l = 0; l < L; l++)
        {
            w[l] = new float[_weights[l].Length][];
            for (int j = 0; j < w[l].Length; j++) w[l][j] = (float[])_weights[l][j].Clone();
            b[l] = (float[])_biases[l].Clone();
        }
        return (w, b);
    }

    /// <summary>Restore a snapshot (must come from an identical architecture).</summary>
    public void Restore((float[][][] W, float[][] B) snap)
    {
        EnsureWeightsSynced(); // never clobber newer VRAM weights
        int L = _sizes.Length - 1;
        for (int l = 0; l < L; l++)
            for (int j = 0; j < _weights[l].Length; j++)
                Array.Copy(snap.W[l][j], _weights[l][j], _weights[l][j].Length);
        for (int l = 0; l < L; l++)
            Array.Copy(snap.B[l], _biases[l], _biases[l].Length);
        PushWeightsToGpu();
    }

    /// <summary>Add gaussian noise to every weight + bias (evolutionary mutation).</summary>
    public void Mutate(float scale, Random rng)
    {
        EnsureWeightsSynced(); // never clobber newer VRAM weights
        int L = _sizes.Length - 1;
        for (int l = 0; l < L; l++)
        {
            for (int j = 0; j < _weights[l].Length; j++)
                for (int i = 0; i < _weights[l][j].Length; i++)
                    _weights[l][j][i] += NextGaussian(rng) * scale;
            for (int j = 0; j < _biases[l].Length; j++)
                _biases[l][j] += NextGaussian(rng) * scale;
        }
        PushWeightsToGpu();
    }

    private static float NextGaussian(Random rng)
    {
        double u1 = Math.Max(double.Epsilon, rng.NextDouble());
        double u2 = rng.NextDouble();
        return (float)(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
    }
}
