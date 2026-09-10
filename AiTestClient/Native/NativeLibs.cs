using System;
using System.Reflection;
using System.Runtime.InteropServices;

namespace AiTestClient.Native;

/// <summary>
/// Cross-platform native library resolver. Maps the logical library names used
/// in our P/Invoke declarations ("SDL2", "GL") to the real native library names
/// on Windows, Linux, and macOS, so the client runs on any of them without
/// hard-coded paths.
/// </summary>
public static class NativeLibs
{
    public static void Install()
    {
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
                    : OperatingSystem.IsMacOS()
                        ? new[] { "libSDL2-2.0.0.dylib", "libSDL2.dylib" }
                        : new[] { "libSDL2-2.0.so.0", "libSDL2.so.0", "libSDL2.so" };
                break;
            case "GL":
                candidates = OperatingSystem.IsWindows()
                    ? new[] { "opengl32.dll" }
                    : OperatingSystem.IsMacOS()
                        ? new[] { "libGL.dylib", "/System/Library/Frameworks/OpenGL.framework/OpenGL" }
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
