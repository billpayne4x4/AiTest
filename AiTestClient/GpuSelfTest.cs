using System;
using System.Diagnostics;
using AiModel_V1;

namespace AiTestClient;

/// <summary>
/// GPU parity self-test (<c>--selftest-gpu</c>): runs identical same-seed
/// training on CPU and GPU and compares every output and weight. Proves the
/// hand-written PTX does the same math as the C# reference, and that
/// CPU→GPU→CPU toggling is lossless. Prints PASS/FAIL with timings.
/// </summary>
public static class GpuSelfTest
{
    public static void Run()
    {
        Console.WriteLine("GPU self-test: CPU vs GPU parity (no window needed)");
        int[] sizes = { 7, 32, 32, 3 };
        var cfg = new TrainingConfig();

        var cpu = new NeuralNetwork(sizes, cfg, 2024);
        var gpu = new NeuralNetwork(sizes, cfg, 2024);
        if (!gpu.TryEnableGpu(out string msg))
        {
            Console.WriteLine("SKIP: " + msg);
            return;
        }
        Console.WriteLine("Device: " + gpu.DeviceLabel + $" ({gpu.GpuVramMb:F2} MB VRAM)");

        // Single-shot: one forward on identical fresh weights isolates pure
        // kernel accuracy from chaotic training-trajectory divergence.
        var probe = new float[7];
        var prng = new Random(99);
        for (int i = 0; i < probe.Length; i++) probe[i] = (float)(prng.NextDouble() * 2 - 1);
        var pc = cpu.Forward(probe);
        var pg = gpu.Forward(probe);
        double singleDiff = 0;
        for (int j = 0; j < pc.Length; j++) singleDiff = Math.Max(singleDiff, Math.Abs(pc[j] - pg[j]));
        Console.WriteLine($"Single forward diff (fresh weights): {singleDiff:E3}");
        for (int l = 0; l < sizes.Length; l++)
        {
            double ld = 0;
            var ca = cpu.LayerActivations(l);
            var ga = gpu.LayerActivations(l);
            for (int j = 0; j < ca.Length; j++) ld = Math.Max(ld, Math.Abs(ca[j] - ga[j]));
            Console.WriteLine($"  layer {l} (n={ca.Length}): max diff {ld:E3}");
        }

        // Single-update isolation: one policy + one supervised step on fresh nets.
        var cpu1 = new NeuralNetwork(sizes, cfg, 2024);
        var gpu1 = new NeuralNetwork(sizes, cfg, 2024);
        gpu1.TryEnableGpu(out _);
        float[] fixedIn = { 0.1f, -0.3f, 0.5f, 0.2f, -0.1f, 0.4f, -0.2f };
        float[] fixedGrad = { 0.05f, -0.04f, 0.03f };
        cpu1.Forward(fixedIn); gpu1.Forward(fixedIn);
        cpu1.PolicyGradientUpdate(fixedGrad, 0.008f);
        gpu1.PolicyGradientUpdate(fixedGrad, 0.008f);
        double pgDiff = MaxWeightDiff(cpu1, gpu1, sizes);
        Console.WriteLine($"Single policy-update weight diff: {pgDiff:E3}");
        float[] fixedTgt = { 0.2f, -0.1f, 0.0f };
        cpu1.SupervisedUpdate(fixedIn, fixedTgt, 0.012f);
        gpu1.SupervisedUpdate(fixedIn, fixedTgt, 0.012f);
        double supDiff = MaxWeightDiff(cpu1, gpu1, sizes);
        Console.WriteLine($"Single supervised-update weight diff: {supDiff:E3}");
        gpu1.DisableGpu();

        // Adam-vs-SGD bisection: two updates on fresh nets, both modes.
        foreach (bool adam in new[] { true, false })
        {
            var cfgg = new TrainingConfig { UseAdam = adam };
            var ca = new NeuralNetwork(sizes, cfgg, 2024);
            var ga = new NeuralNetwork(sizes, cfgg, 2024);
            ga.TryEnableGpu(out _);
            ca.Forward(fixedIn); ga.Forward(fixedIn);
            Console.WriteLine($"  two-update (adam={adam}) fwd diff: {MaxWeightDiff(ca, ga, sizes):E3} (expect ~0)");
            ca.PolicyGradientUpdate(fixedGrad, 0.008f);
            ga.PolicyGradientUpdate(fixedGrad, 0.008f);
            Console.WriteLine($"  two-update (adam={adam}) after policy: {MaxWeightDiff(ca, ga, sizes):E3}");
            ca.SupervisedUpdate(fixedIn, fixedTgt, 0.012f);
            ga.SupervisedUpdate(fixedIn, fixedTgt, 0.012f);
            Console.WriteLine($"  two-update diff (adam={adam}): {MaxWeightDiff(ca, ga, sizes):E3}");
            ga.DisableGpu();
        }

        const int iters = 300;
        var rng = new Random(7);
        var inputs = new float[iters][];
        var grads = new float[iters][];
        var targets = new float[iters][];
        for (int k = 0; k < iters; k++)
        {
            inputs[k] = new float[7];
            grads[k] = new float[3];
            targets[k] = new float[3];
            for (int i = 0; i < 7; i++) inputs[k][i] = (float)(rng.NextDouble() * 2 - 1);
            for (int j = 0; j < 3; j++)
            {
                grads[k][j] = (float)(rng.NextDouble() * 2 - 1);
                targets[k][j] = (float)(rng.NextDouble() * 2 - 1);
            }
        }

        var cpuOut = new float[iters][];
        var preSnaps = new (float[][][] W, float[][] B)[iters];
        var postSnaps = new (float[][][] W, float[][] B)[iters];
        var swCpu = Stopwatch.StartNew();
        for (int k = 0; k < iters; k++)
        {
            preSnaps[k] = cpu.Snapshot();
            cpuOut[k] = cpu.Forward(inputs[k]);
            if (k % 2 == 0) cpu.PolicyGradientUpdate(grads[k], 0.008f);
            else cpu.SupervisedUpdate(inputs[k], targets[k], 0.012f);
            postSnaps[k] = cpu.Snapshot();
        }
        swCpu.Stop();

        // Locked-step GPU run: restore the CPU pre-state every step, so each
        // iteration measures exactly ONE step of GPU error. (Free trajectories
        // diverge chaotically from 1e-7 noise, which proves nothing.)
        double oneStepOut = 0, oneStepW = 0;
        var swGpu = Stopwatch.StartNew();
        for (int k = 0; k < iters; k++)
        {
            gpu.Restore(preSnaps[k]);
            var go = gpu.Forward(inputs[k]);
            for (int j = 0; j < 3; j++)
                oneStepOut = Math.Max(oneStepOut, Math.Abs(cpuOut[k][j] - go[j]));
            if (k % 2 == 0) gpu.PolicyGradientUpdate(grads[k], 0.008f);
            else gpu.SupervisedUpdate(inputs[k], targets[k], 0.012f);
            oneStepW = Math.Max(oneStepW, DiffVsSnapshot(gpu, postSnaps[k], sizes));
        }
        swGpu.Stop();

        Console.WriteLine($"Locked-step: max one-step output diff {oneStepOut:E3}, max one-step weight diff {oneStepW:E3}");

        // seamless toggle: snapshot the GPU-trained brain into a fresh CPU net
        var snap = gpu.Snapshot();
        var fresh = new NeuralNetwork(sizes, cfg, 9999);
        fresh.Restore(snap);
        gpu.DisableGpu();
        double toggleDiff = 0;
        for (int k = 0; k < 20; k++)
        {
            var fa = fresh.Forward(inputs[k]);
            var ga = gpu.Forward(inputs[k]);
            for (int j = 0; j < 3; j++) toggleDiff = Math.Max(toggleDiff, Math.Abs(fa[j] - ga[j]));
        }

        Console.WriteLine($"Iters: {iters}, one-step output diff: {oneStepOut:E3}, one-step weight diff: {oneStepW:E3}");
        Console.WriteLine($"Toggle GPU->CPU diff: {toggleDiff:E3}");
        Console.WriteLine($"CPU: {swCpu.ElapsedMilliseconds} ms, GPU: {swGpu.ElapsedMilliseconds} ms for {iters} iters");
        bool pass = oneStepOut < 1e-5 && oneStepW < 1e-5 && toggleDiff < 1e-6;
        Console.WriteLine(pass ? "RESULT: PASS" : "RESULT: FAIL");
    }

    private static double DiffVsSnapshot(NeuralNetwork net, (float[][][] W, float[][] B) snap, int[] sizes)
    {
        double m = 0;
        for (int l = 0; l < sizes.Length - 1; l++)
            for (int j = 0; j < sizes[l + 1]; j++)
            {
                for (int i = 0; i < sizes[l]; i++)
                    m = Math.Max(m, Math.Abs(net.Weight(l, j, i) - snap.W[l][j][i]));
                m = Math.Max(m, Math.Abs(net.Bias(l, j) - snap.B[l][j]));
            }
        return m;
    }

    private static double MaxWeightDiff(NeuralNetwork a, NeuralNetwork b, int[] sizes)
    {
        double m = 0;
        for (int l = 0; l < sizes.Length - 1; l++)
            for (int j = 0; j < sizes[l + 1]; j++)
            {
                for (int i = 0; i < sizes[l]; i++)
                    m = Math.Max(m, Math.Abs(a.Weight(l, j, i) - b.Weight(l, j, i)));
                m = Math.Max(m, Math.Abs(a.Bias(l, j) - b.Bias(l, j)));
            }
        return m;
    }
}
