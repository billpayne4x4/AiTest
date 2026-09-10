using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AiTestClient.Native;

/// <summary>
/// Cross-platform native library resolver (Windows + Linux only).
/// Maps the logical library names used in our P/Invoke declarations
/// ("SDL2", "GL", "cuda") to the real native library names on each OS.
/// macOS is not supported.
/// </summary>
public static class NativeLibs
{
    public static void Install()
    {
        if (OperatingSystem.IsMacOS())
            throw new PlatformNotSupportedException(
                "AiTest supports Windows and Linux only. macOS is not supported.");
        NativeLibrary.SetDllImportResolver(typeof(NativeLibs).Assembly, Resolve);
    }

    private static IntPtr Resolve(string libraryName, Assembly assembly, DllImportSearchPath? searchPath)
    {
        string[] candidates;
        switch (libraryName)
        {
            case "SDL2":
                candidates = OperatingSystem.IsWindows()
                    ? new[] { "SDL2.dll" }
                    : new[] { "libSDL2-2.0.so.0", "libSDL2.so.0", "libSDL2.so" };
                break;
            case "GL":
                candidates = OperatingSystem.IsWindows()
                    ? new[] { "opengl32.dll" }
                    : new[] { "libGL.so.1", "libGL.so" };
                break;
            default:
                candidates = new[] { libraryName };
                break;
        }

        foreach (var c in candidates)
        {
            if (NativeLibrary.TryLoad(c, assembly, searchPath, out var ptr))
                return ptr;
        }
        throw new DllNotFoundException(
            $"Could not load native library '{libraryName}'. Tried: {string.Join(", ", candidates)}. " +
            "Make sure SDL2 and OpenGL are installed.");
    }
}
