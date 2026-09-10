using System;

namespace AiModel_V1;

/// <summary>
/// A small feed-forward neural network with backpropagation.
/// All hidden and output neurons use the tanh activation.
///
/// Two learning modes are supported:
///  - PolicyGradientUpdate(reward): used by the car brain. Treats the reward as a
///    scalar return and pushes the weights in the direction that increases the
///    (reward-weighted) output. This is what lets the car "learn to drive".
///  - SupervisedUpdate(input, target): classic MSE backprop (kept for completeness).
///
/// The network exposes per-node activations and all weights so the client can
/// draw the "brain" and light nodes up as they activate.
/// </summary>
public class NeuralNetwork
{
    private readonly int[] _sizes;
    // _weights[l][outNode][inNode]
    private readonly float[][][] _weights;
    // _biases[l][outNode]
    private readonly float[][] _biases;
    private readonly float[][] _activations;
    private readonly float[][] _pre;
    private readonly float[][] _delta;
    private Random _rng;

    public NeuralNetwork(int[] sizes, int seed = 1234)
    {
        if (sizes.Length < 2) throw new ArgumentException("Need at least input and output layers.");
        _sizes = (int[])sizes.Clone();
        _rng = new Random(seed);
        int L = _sizes.Length - 1;
        _weights = new float[L][][];
        _biases = new float[L][];
        for (int l = 0; l < L; l++)
        {
            int fanIn = _sizes[l], fanOut = _sizes[l + 1];
            _weights[l] = new float[fanOut][];
            float scale = (float)Math.Sqrt(2.0 / fanIn); // Xavier-ish
            for (int j = 0; j < fanOut; j++)
            {
                _weights[l][j] = new float[fanIn];
                for (int i = 0; i < fanIn; i++)
                    _weights[l][j][i] = (float)(_rng.NextDouble() * 2.0 - 1.0) * scale;
            }
            _biases[l] = new float[fanOut];
        }
        _activations = new float[_sizes.Length][];
        _pre = new float[_sizes.Length][];
        _delta = new float[_sizes.Length][];
        for (int l = 0; l < _sizes.Length; l++)
        {
            _activations[l] = new float[_sizes[l]];
            _pre[l] = new float[_sizes[l]];
            _delta[l] = new float[_sizes[l]];
        }
    }

    public int InputCount => _sizes[0];
    public int OutputCount => _sizes[^1];
    public int LayerCount => _sizes.Length;
    public int[] Sizes => _sizes;

    // ---- visualization accessors ----
    public float Activation(int layer, int node) => _activations[layer][node];
    public float[] LayerActivations(int layer) => (float[])_activations[layer].Clone();
    public float Weight(int layer, int outNode, int inNode) => _weights[layer][outNode][inNode];
    public float Bias(int layer, int outNode) => _biases[layer][outNode];

    // ---- inference ----
    public float[] Forward(float[] input)
    {
        int n = Math.Min(input.Length, _sizes[0]);
        Array.Copy(input, _activations[0], n);
        for (int l = 0; l < _sizes.Length - 1; l++)
        {
            int fanOut = _sizes[l + 1];
            var a = _activations[l];
            for (int j = 0; j < fanOut; j++)
            {
                float z = _biases[l][j];
                var w = _weights[l][j];
                for (int i = 0; i < a.Length; i++) z += w[i] * a[i];
                _pre[l + 1][j] = z;
                _activations[l + 1][j] = (float)Math.Tanh(z);
            }
        }
        return (float[])_activations[^1].Clone();
    }

    // ---- learning: policy gradient (reward-weighted) ----
    public void PolicyGradientUpdate(float[] outputGradient, float learningRate)
    {
        int last = _sizes.Length - 1;
        if (outputGradient.Length != _sizes[last])
            throw new ArgumentException("Output gradient must match the output layer size.", nameof(outputGradient));
        for (int j = 0; j < _sizes[last]; j++)
        {
            float act = _activations[last][j];
            float dTanh = 1f - act * act;
            _delta[last][j] = outputGradient[j] * dTanh;
        }
        BackpropAndUpdate(learningRate);
    }

    // ---- learning: supervised (MSE) ----
    public float SupervisedUpdate(float[] input, float[] target, float learningRate)
    {
        Forward(input);
        int last = _sizes.Length - 1;
        float err = 0f;
        for (int j = 0; j < _sizes[last]; j++)
        {
            float d = _activations[last][j] - target[j];
            err += d * d;
            float act = _activations[last][j];
            // BackpropAndUpdate adds the gradient, so use target-output here
            // (the previous output-target sign performed gradient ascent).
            _delta[last][j] = -d * (1f - act * act);
        }
        BackpropAndUpdate(learningRate);
        return err / _sizes[last];
    }

    private void BackpropAndUpdate(float learningRate)
    {
        int L = _sizes.Length - 1;
        for (int l = L - 1; l >= 0; l--)
        {
            var w = _weights[l];
            var a = _activations[l];
            var dNext = _delta[l + 1];

            // zero this layer's delta accumulator
            for (int i = 0; i < _delta[l].Length; i++) _delta[l][i] = 0f;

            // update weights + accumulate delta for this layer
            for (int j = 0; j < dNext.Length; j++)
            {
                float dj = dNext[j];
                if (dj == 0f) continue;
                var wij = w[j];
                for (int i = 0; i < a.Length; i++)
                {
                    float wOld = wij[i];
                    wij[i] += learningRate * dj * a[i];
                    _delta[l][i] += wOld * dj;
                }
            }

            // apply tanh derivative for this layer
            for (int i = 0; i < _delta[l].Length; i++)
            {
                float act = _activations[l][i];
                _delta[l][i] *= (1f - act * act);
            }

            // update biases
            for (int j = 0; j < _biases[l].Length; j++)
                _biases[l][j] += learningRate * dNext[j];
        }
    }

    // Re-randomize all weights (used for "retrain from scratch").
    public void Reinitialize(int seed)
    {
        _rng = new Random(seed);
        int L = _sizes.Length - 1;
        for (int l = 0; l < L; l++)
        {
            int fanIn = _sizes[l], fanOut = _sizes[l + 1];
            float scale = (float)Math.Sqrt(2.0 / fanIn);
            for (int j = 0; j < fanOut; j++)
                for (int i = 0; i < fanIn; i++)
                    _weights[l][j][i] = (float)(_rng.NextDouble() * 2.0 - 1.0) * scale;
            for (int j = 0; j < fanOut; j++) _biases[l][j] = 0f;
        }
    }
}
