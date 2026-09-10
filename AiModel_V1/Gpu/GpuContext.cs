using System;
using System.Runtime.InteropServices;
using System.Text;

namespace AiModel_V1.Gpu;

/// <summary>
/// Owns the CUDA context, compiled PTX module, and all VRAM buffers for one
/// network shape. Weights live in VRAM while attached; the host keeps a master
/// copy that is synced on toggle, snapshot, mutate, and visualization reads.
/// Per-step traffic is tiny (inputs up, activations down); full weight blobs
/// only move on demand. No NuGet dependencies — straight driver-API P/Invoke.
/// </summary>
internal sealed class GpuContext : IDisposable
{
    public readonly int[] Sizes;
    public readonly int L; // weight layers
    public readonly int MaxW; // max layer width
    public string DeviceName { get; private set; } = "";
    public ulong VramBytes { get; private set; }

    private IntPtr _ctx;
    private IntPtr _module;
    private IntPtr _fwdFn;
    private IntPtr _bwdFn;
    private bool _disposed;

    // device buffers (CUdeviceptr as ulong)
    private ulong _w, _b, _a, _n, _mw, _vw, _mb, _vb, _g, _bt, _d, _dn;
    private ulong _tOffW, _tOffB, _tOffA, _tFI, _tFO, _tFL;

    // host tables (u32 element offsets)
    private readonly uint[] _offW, _offB, _offA, _fanIn, _fanOut, _flags;
    private readonly int _totalW, _totalB, _totalA;

    // pinned kernel-param blocks: values + slot pointers (built once)
    private readonly byte[] _fwdParams;
    private readonly IntPtr[] _fwdSlots;
    private readonly GCHandle _fwdPin;
    private readonly byte[] _bwdParams;
    private readonly IntPtr[] _bwdSlots;
    private readonly GCHandle _bwdPin;

    private GpuContext(int[] sizes)
    {
        Sizes = (int[])sizes.Clone();
        L = sizes.Length - 1;
        MaxW = 0;
        foreach (var s in sizes) MaxW = Math.Max(MaxW, s);
        _offW = new uint[L]; _offB = new uint[L]; _offA = new uint[L + 1];
        _fanIn = new uint[L]; _fanOut = new uint[L]; _flags = new uint[L];
        int ow = 0, ob = 0, oa = 0;
        for (int l = 0; l < L; l++)
        {
            _offW[l] = (uint)ow; ow += sizes[l + 1] * sizes[l];
            _offB[l] = (uint)ob; ob += sizes[l + 1];
            _offA[l] = (uint)oa; oa += sizes[l];
            _fanIn[l] = (uint)sizes[l]; _fanOut[l] = (uint)sizes[l + 1];
        }
        _offA[L] = (uint)oa; oa += sizes[L];
        _totalW = ow; _totalB = ob; _totalA = oa;
        _fwdParams = new byte[12 * 8 + 4 + 4 + 4];
        _fwdSlots = new IntPtr[15];
        _fwdPin = GCHandle.Alloc(_fwdParams, GCHandleType.Pinned);
        _bwdParams = new byte[16 * 8 + 4 * 4 + 6 * 4];
        _bwdSlots = new IntPtr[26];
        _bwdPin = GCHandle.Alloc(_bwdParams, GCHandleType.Pinned);
    }

    public static GpuContext Create(int[] sizes, TrainingConfig cfg, out string deviceName)
    {
        var ctx = new GpuContext(sizes);
        try
        {
            CudaDriver.Check(CudaDriver.Init(0), "cuInit");
            CudaDriver.Check(CudaDriver.DeviceGetCount(out int n), "cuDeviceGetCount");
            if (n < 1) throw new CudaException("cuInit ok but no CUDA devices found");
            CudaDriver.Check(CudaDriver.DeviceGet(out int dev, 0), "cuDeviceGet");
            deviceName = CudaDriver.DeviceName(dev);
            ctx.DeviceName = deviceName;
            CudaDriver.Check(CudaDriver.CtxCreate(out ctx._ctx, 0, dev), "cuCtxCreate");
            CudaDriver.Check(CudaDriver.CtxGetCurrent(out IntPtr cur), "cuCtxGetCurrent");
            if (cur != ctx._ctx)
                throw new CudaException($"cuCtxCreate did not bind (created={ctx._ctx:X}, current={cur:X})");
            // load PTX (with JIT error-log capture for diagnostics)
            byte[] ptx = Encoding.ASCII.GetBytes(PtxKernels.Source + "\0");
            byte[] log = new byte[16384];
            var hPtx = GCHandle.Alloc(ptx, GCHandleType.Pinned);
            var hLog = GCHandle.Alloc(log, GCHandleType.Pinned);
            ulong logSize = (ulong)log.Length;
            var hSize = GCHandle.Alloc(logSize, GCHandleType.Pinned);
            try
            {
                int[] opts = { CudaDriver.JIT_ERROR_LOG_BUFFER, CudaDriver.JIT_ERROR_LOG_BUFFER_SIZE };
                IntPtr[] vals = { hLog.AddrOfPinnedObject(), hSize.AddrOfPinnedObject() };
                int rc = CudaDriver.ModuleLoadDataEx(out ctx._module, hPtx.AddrOfPinnedObject(),
                    (uint)opts.Length, opts, vals);
                if (rc != 0)
                {
                    int term = Array.IndexOf(log, (byte)0);
                    string details = Encoding.ASCII.GetString(log, 0, term < 0 ? log.Length : term).Trim();
                    throw new CudaException(
                        $"cuModuleLoadData (PTX JIT) failed (CUDA error {rc}). Log: {details}", rc);
                }
            }
            finally { hPtx.Free(); hLog.Free(); hSize.Free(); }
            CudaDriver.Check(CudaDriver.ModuleGetFunction(out ctx._fwdFn, ctx._module, "fwd"), "cuModuleGetFunction(fwd)");
            CudaDriver.Check(CudaDriver.ModuleGetFunction(out ctx._bwdFn, ctx._module, "bwd"), "cuModuleGetFunction(bwd)");
            // allocate
            ctx._w = ctx.AllocF(ctx._totalW); ctx._b = ctx.AllocF(ctx._totalB);
            ctx._a = ctx.AllocF(ctx._totalA); ctx._n = ctx.AllocF(ctx._totalA);
            ctx._mw = ctx.AllocF(ctx._totalW); ctx._vw = ctx.AllocF(ctx._totalW);
            ctx._mb = ctx.AllocF(ctx._totalB); ctx._vb = ctx.AllocF(ctx._totalB);
            ctx._g = ctx.AllocF(ctx._totalA); ctx._bt = ctx.AllocF(ctx._totalA);
            ctx._d = ctx.AllocF(ctx.L * ctx.MaxW); ctx._dn = ctx.AllocF(ctx.MaxW);
            ctx._tOffW = ctx.AllocU(ctx._offW); ctx._tOffB = ctx.AllocU(ctx._offB);
            ctx._tOffA = ctx.AllocU(ctx._offA); ctx._tFI = ctx.AllocU(ctx._fanIn);
            ctx._tFO = ctx.AllocU(ctx._fanOut); ctx._tFL = ctx.AllocU(ctx._flags);
            ctx.Zero(ctx._mw, ctx._totalW); ctx.Zero(ctx._vw, ctx._totalW);
            ctx.Zero(ctx._mb, ctx._totalB); ctx.Zero(ctx._vb, ctx._totalB);
            // gamma=1, beta=0
            var g = new float[ctx._totalA]; var bt = new float[ctx._totalA];
            for (int i = 0; i < g.Length; i++) g[i] = 1f;
            ctx.Upload(ctx._g, g); ctx.Upload(ctx._bt, bt);
            ctx.WriteParamBlocks();
            return ctx;
        }
        catch
        {
            ctx.Dispose();
            throw;
        }
    }

    // ---- raw buffer helpers ----
    private ulong AllocF(int floats)
    {
        ulong bytes = (ulong)floats * 4;
        CudaDriver.Check(CudaDriver.MemAlloc(out ulong p, Math.Max(bytes, 4)), "cuMemAlloc");
        VramBytes += Math.Max(bytes, 4);
        return p;
    }

    private ulong AllocU(uint[] data)
    {
        CudaDriver.Check(CudaDriver.MemAlloc(out ulong p, (ulong)data.Length * 4), "cuMemAlloc(table)");
        VramBytes += (ulong)data.Length * 4;
        Upload(p, data);
        return p;
    }

    private void Zero(ulong dev, int floats)
    {
        if (floats <= 0) return;
        // NOTE: cuMemsetD8 returns INVALID_CONTEXT on this driver despite a
        // healthy context (allocs/queries work), so zero via HtoD upload.
        var zeros = new byte[(ulong)floats * 4];
        var h = GCHandle.Alloc(zeros, GCHandleType.Pinned);
        try { CudaDriver.Check(CudaDriver.MemcpyHtoD(dev, h.AddrOfPinnedObject(), (ulong)zeros.Length), "cuMemcpyHtoD(zero)"); }
        finally { h.Free(); }
    }

    public void ZeroDelta() => Zero(_d, L * MaxW);
    public void ZeroAdam()
    {
        Zero(_mw, _totalW); Zero(_vw, _totalW); Zero(_mb, _totalB); Zero(_vb, _totalB);
    }

    public void Upload(ulong dev, float[] data)
    {
        var h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { CudaDriver.Check(CudaDriver.MemcpyHtoD(dev, h.AddrOfPinnedObject(), (ulong)data.Length * 4), "cuMemcpyHtoD"); }
        finally { h.Free(); }
    }

    public void Upload(ulong dev, uint[] data)
    {
        var h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { CudaDriver.Check(CudaDriver.MemcpyHtoD(dev, h.AddrOfPinnedObject(), (ulong)data.Length * 4), "cuMemcpyHtoD"); }
        finally { h.Free(); }
    }

    public void Download(ulong dev, float[] data)
    {
        var h = GCHandle.Alloc(data, GCHandleType.Pinned);
        try { CudaDriver.Check(CudaDriver.MemcpyDtoH(h.AddrOfPinnedObject(), dev, (ulong)data.Length * 4), "cuMemcpyDtoH"); }
        finally { h.Free(); }
    }

    // ---- weight/activation sync (flat blobs, same layout as tables) ----
    public void UploadWeights(float[][][] w, float[][] b)
    {
        var flat = new float[_totalW];
        int k = 0;
        for (int l = 0; l < L; l++)
            for (int j = 0; j < w[l].Length; j++)
                for (int i = 0; i < w[l][j].Length; i++) flat[k++] = w[l][j][i];
        Upload(_w, flat);
        var fb = new float[_totalB];
        k = 0;
        for (int l = 0; l < L; l++)
            for (int j = 0; j < b[l].Length; j++) fb[k++] = b[l][j];
        Upload(_b, fb);
    }

    public void DownloadWeights(float[][][] w, float[][] b)
    {
        var flat = new float[_totalW];
        Download(_w, flat);
        int k = 0;
        for (int l = 0; l < L; l++)
            for (int j = 0; j < w[l].Length; j++)
                for (int i = 0; i < w[l][j].Length; i++) w[l][j][i] = flat[k++];
        var fb = new float[_totalB];
        Download(_b, fb);
        k = 0;
        for (int l = 0; l < L; l++)
            for (int j = 0; j < b[l].Length; j++) b[l][j] = fb[k++];
    }

    public void DownloadActs(float[][] acts, float[][] normed)
    {
        var flat = new float[_totalA];
        Download(_a, flat);
        int k = 0;
        for (int l = 0; l < acts.Length; l++)
            for (int j = 0; j < acts[l].Length; j++) acts[l][j] = flat[k++];
        Download(_n, flat);
        k = 0;
        for (int l = 0; l < normed.Length; l++)
            for (int j = 0; j < normed[l].Length; j++) normed[l][j] = flat[k++];
    }

    public void UploadInputs(float[] input, int count)
    {
        var buf = new float[Sizes[0]];
        Array.Copy(input, buf, Math.Min(count, buf.Length));
        var h = GCHandle.Alloc(buf, GCHandleType.Pinned);
        try
        {
            ulong dst = _a + (ulong)_offA[0] * 4;
            CudaDriver.Check(CudaDriver.MemcpyHtoD(dst, h.AddrOfPinnedObject(), (ulong)buf.Length * 4), "cuMemcpyHtoD(inputs)");
        }
        finally { h.Free(); }
    }

    public void UploadDelta(float[] dNext)
    {
        var buf = new float[MaxW];
        Array.Copy(dNext, buf, Math.Min(dNext.Length, buf.Length));
        Upload(_dn, buf);
    }

    public void RefreshTables(TrainingConfig cfg)
    {
        for (int l = 0; l < L; l++)
        {
            bool isOut = l == L - 1;
            uint f = 0;
            if (cfg.UseLayerNorm && !isOut) f |= 1;
            if (cfg.UseResidual && !isOut && Sizes[l + 1] == Sizes[l]) f |= 2;
            if (isOut) f |= 4;
            _flags[l] = f;
        }
        Upload(_tFL, _flags);
    }

    // ---- kernel launches ----
    private void WriteParamBlocks()
    {
        IntPtr baseP = _fwdPin.AddrOfPinnedObject();
        ulong[] devs = { _w, _b, _a, _n, _g, _bt, _tOffW, _tOffB, _tOffA, _tFI, _tFO, _tFL };
        for (int i = 0; i < 12; i++)
        {
            Marshal.WriteInt64(baseP, i * 8, (long)devs[i]);
            _fwdSlots[i] = IntPtr.Add(baseP, i * 8);
        }
        Marshal.WriteInt32(baseP, 96, L);
        _fwdSlots[12] = IntPtr.Add(baseP, 96);
        _fwdSlots[13] = IntPtr.Add(baseP, 100); // aid, written per launch
        _fwdSlots[14] = IntPtr.Add(baseP, 104); // temp, written per launch

        IntPtr bb = _bwdPin.AddrOfPinnedObject();
        // pBB aliases the bias blob (same memory the forward pass reads).
        ulong[] bdevs = { _w, _mw, _vw, _mb, _vb, _b, _a, _n, _d, _dn, _tOffW, _tOffB, _tOffA, _tFI, _tFO, _tFL };
        for (int i = 0; i < 16; i++)
        {
            Marshal.WriteInt64(bb, i * 8, (long)bdevs[i]);
            _bwdSlots[i] = IntPtr.Add(bb, i * 8);
        }
        Marshal.WriteInt32(bb, 128, L);
        _bwdSlots[16] = IntPtr.Add(bb, 128); // nl
        _bwdSlots[17] = IntPtr.Add(bb, 132); // aid
        _bwdSlots[18] = IntPtr.Add(bb, 136); // maxW
        _bwdSlots[19] = IntPtr.Add(bb, 140); // useAdam
        for (int i = 0; i < 6; i++) _bwdSlots[20 + i] = IntPtr.Add(bb, 144 + i * 4); // lr,bc1,bc2,dec,bdec,maxn
        Marshal.WriteInt32(bb, 136, MaxW);
    }

    private static void WriteU32(IntPtr baseP, int off, uint v)
        => Marshal.WriteInt32(baseP, off, (int)v);
    private static void WriteF32(IntPtr baseP, int off, float v)
        => Marshal.WriteInt32(baseP, off, BitConverter.SingleToInt32Bits(v));

    private uint BlockDim => (uint)((MaxW + 31) / 32 * 32);

    public void LaunchForward(TrainingConfig cfg, float tempDiv)
    {
        RefreshTables(cfg);
        IntPtr baseP = _fwdPin.AddrOfPinnedObject();
        WriteU32(baseP, 100, (uint)cfg.Activation);
        WriteF32(baseP, 104, tempDiv);
        CudaDriver.Check(CudaDriver.LaunchKernel(_fwdFn,
            1, 1, 1, BlockDim, 1, 1, (uint)(BlockDim + 2) * 4, IntPtr.Zero, _fwdSlots, IntPtr.Zero),
            "cuLaunchKernel(fwd)");
        CudaDriver.Check(CudaDriver.CtxSynchronize(), "cuCtxSynchronize(fwd)");
    }

    public void LaunchBackward(TrainingConfig cfg, float lr, float bc1, float bc2)
    {
        RefreshTables(cfg);
        IntPtr bb = _bwdPin.AddrOfPinnedObject();
        WriteU32(bb, 132, (uint)cfg.Activation);
        WriteU32(bb, 140, cfg.UseAdam ? 1u : 0u);
        WriteF32(bb, 144, lr);
        WriteF32(bb, 148, bc1);
        WriteF32(bb, 152, bc2);
        WriteF32(bb, 156, 1e-4f);
        WriteF32(bb, 160, 1e-3f);
        WriteF32(bb, 164, 3f);
        ZeroDelta();
        CudaDriver.Check(CudaDriver.LaunchKernel(_bwdFn,
            1, 1, 1, BlockDim, 1, 1, 0, IntPtr.Zero, _bwdSlots, IntPtr.Zero),
            "cuLaunchKernel(bwd)");
        CudaDriver.Check(CudaDriver.CtxSynchronize(), "cuCtxSynchronize(bwd)");
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        try
        {
            foreach (var p in new[] { _w, _b, _a, _n, _mw, _vw, _mb, _vb, _g, _bt, _d, _dn, _tOffW, _tOffB, _tOffA, _tFI, _tFO, _tFL })
                if (p != 0) CudaDriver.MemFree(p);
            if (_module != IntPtr.Zero) CudaDriver.ModuleUnload(_module);
            if (_ctx != IntPtr.Zero) CudaDriver.CtxDestroy(_ctx);
        }
        catch { /* shutdown best-effort */ }
        if (_fwdPin.IsAllocated) _fwdPin.Free();
        if (_bwdPin.IsAllocated) _bwdPin.Free();
    }
}
