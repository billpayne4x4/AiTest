using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AiModel_V1.Gpu;

/// <summary>
/// Thin P/Invoke wrapper over the NVIDIA CUDA *driver* API (libcuda), with no
/// NuGet dependencies. Windows (nvcuda.dll) and Linux (libcuda.so.1) only —
/// macOS has no CUDA driver and is not supported by this project.
///
/// Only the ~20 functions the training backend needs are declared. All calls
/// are synchronous; errors surface as <see cref="CudaException"/>.
/// </summary>
public static class CudaDriver
{
    // Keep in sync with NativeLibs-style resolution: logical name "cuda".
    static CudaDriver()
    {
        NativeLibrary.SetDllImportResolver(typeof(CudaDriver).Assembly, Resolve);
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        if (libraryName != "cuda") return IntPtr.Zero;
        string[] candidates = OperatingSystem.IsWindows()
            ? new[] { "nvcuda.dll" }
            : new[] { "libcuda.so.1", "libcuda.so" };
        foreach (var c in candidates)
            if (NativeLibrary.TryLoad(c, assembly, searchPath, out var ptr))
                return ptr;
        return IntPtr.Zero; // caller turns this into a friendly "GPU unavailable"
    }

    public static bool LibraryPresent
    {
        get
        {
            try { NativeLibrary.TryLoad(OperatingSystem.IsWindows() ? "nvcuda.dll" : "libcuda.so.1", out _); return true; }
            catch { return false; }
        }
    }

    // NOTE: x64 has a single calling convention, so Cdecl is correct on both
    // Windows and Linux here. This would need revisiting for 32-bit builds.
    private const CallingConvention Cc = CallingConvention.Cdecl;

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuInit")]
    public static extern int Init(uint flags);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuDeviceGetCount")]
    public static extern int DeviceGetCount(out int count);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuDeviceGet")]
    public static extern int DeviceGet(out int device, int ordinal);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuDeviceGetName")]
    public static extern int DeviceGetName([Out] byte[] name, int len, int device);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuDeviceGetAttribute")]
    public static extern int DeviceGetAttribute(out int value, int attrib, int device);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuCtxCreate_v2")]
    public static extern int CtxCreate(out IntPtr ctx, uint flags, int device);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuCtxDestroy_v2")]
    public static extern int CtxDestroy(IntPtr ctx);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuCtxGetCurrent")]
    public static extern int CtxGetCurrent(out IntPtr ctx);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuCtxPushCurrent_v2")]
    public static extern int CtxPushCurrent(IntPtr ctx);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuDriverGetVersion")]
    public static extern int DriverGetVersion(out int version);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuCtxSynchronize")]
    public static extern int CtxSynchronize();

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemAlloc_v2")]
    public static extern int MemAlloc(out ulong dptr, ulong bytesize);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemFree_v2")]
    public static extern int MemFree(ulong dptr);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemcpyHtoD_v2")]
    public static extern int MemcpyHtoD(ulong dst, IntPtr src, ulong bytes);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemcpyDtoH_v2")]
    public static extern int MemcpyDtoH(IntPtr dst, ulong src, ulong bytes);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemsetD8")]
    public static extern int MemsetD8(ulong dst, byte value, ulong n);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuModuleLoadData")]
    public static extern int ModuleLoadData(out IntPtr module, IntPtr image);

    // Extended module load with JIT options (used to retrieve the PTX log).
    public const int JIT_ERROR_LOG_BUFFER = 5;
    public const int JIT_ERROR_LOG_BUFFER_SIZE = 6;

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuModuleLoadDataEx")]
    public static extern int ModuleLoadDataEx(out IntPtr module, IntPtr image,
        uint numOptions, [In] int[] options, [In] IntPtr[] optionValues);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuModuleUnload")]
    public static extern int ModuleUnload(IntPtr module);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuModuleGetFunction")]
    public static extern int ModuleGetFunction(out IntPtr func, IntPtr module,
        [MarshalAs(UnmanagedType.LPStr)] string name);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuLaunchKernel")]
    public static extern int LaunchKernel(IntPtr func,
        uint gridDimX, uint gridDimY, uint gridDimZ,
        uint blockDimX, uint blockDimY, uint blockDimZ,
        uint sharedMemBytes, IntPtr stream,
        [In] IntPtr[] kernelParams, IntPtr extra);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuMemGetInfo_v2")]
    public static extern int MemGetInfo(out ulong free, out ulong total);

    [DllImport("cuda", CallingConvention = Cc, EntryPoint = "cuGetErrorString")]
    public static extern int GetErrorString(int error, out IntPtr str);

    public static void Check(int code, string what)
    {
        if (code == 0) return;
        string msg = what + " failed (CUDA error " + code + ")";
        try
        {
            if (GetErrorString(code, out var p) == 0 && p != IntPtr.Zero)
                msg += ": " + Marshal.PtrToStringAnsi(p);
        }
        catch { /* best effort */ }
        throw new CudaException(msg, code);
    }

    public static string DeviceName(int device)
    {
        var buf = new byte[256];
        Check(DeviceGetName(buf, buf.Length, device), "cuDeviceGetName");
        int n = Array.IndexOf(buf, (byte)0);
        return System.Text.Encoding.ASCII.GetString(buf, 0, n < 0 ? buf.Length : n);
    }
}

/// <summary>CUDA driver call failed.</summary>
public sealed class CudaException : Exception
{
    public int Code { get; }
    public CudaException(string message, int code = -1) : base(message) => Code = code;
}
