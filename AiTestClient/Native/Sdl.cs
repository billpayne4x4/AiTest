using System;
using System.Runtime.InteropServices;

namespace AiTestClient.Native;

/// <summary>
/// Minimal SDL2 P/Invoke bindings: window creation, an OpenGL context, and the
/// event pump (keyboard + mouse + resize + quit).
/// </summary>
public static class Sdl
{
    // init / window flags
    public const uint INIT_VIDEO = 0x00000020;
    public const uint WINDOW_OPENGL = 0x00000002;
    public const uint WINDOW_RESIZABLE = 0x00000020;
    public const uint WINDOW_ALLOW_HIGHDPI = 0x00002000;
    public const uint WINDOWPOS_CENTERED = 0x20000000;

    // GL attributes
    public const int GL_RED_SIZE = 0;
    public const int GL_GREEN_SIZE = 1;
    public const int GL_BLUE_SIZE = 2;
    public const int GL_ALPHA_SIZE = 3;
    public const int GL_DOUBLEBUFFER = 5;
    public const int GL_DEPTH_SIZE = 7;
    public const int GL_MULTISAMPLEBUFFERS = 13;
    public const int GL_MULTISAMPLESAMPLES = 14;
    public const int GL_CONTEXT_MAJOR_VERSION = 0x2091;
    public const int GL_CONTEXT_MINOR_VERSION = 0x2092;
    public const int GL_CONTEXT_PROFILE_MASK = 0x2088;
    public const int GL_CONTEXT_PROFILE_CORE = 0x0001;
    public const int GL_CONTEXT_PROFILE_COMPATIBILITY = 0x0002;

    // event types
    public const uint EV_QUIT = 0x100;
    public const uint EV_WINDOWEVENT = 0x200;
    public const uint EV_KEYDOWN = 0x300;
    public const uint EV_KEYUP = 0x301;
    public const uint EV_MOUSEMOTION = 0x400;
    public const uint EV_MOUSEBUTTONDOWN = 0x401;
    public const uint EV_MOUSEBUTTONUP = 0x402;
    public const uint EV_MOUSEWHEEL = 0x403;

    public const int WINDOWEVENT_RESIZED = 0x05;
    public const int WINDOWEVENT_SIZE_CHANGED = 0x06;

    // key codes (SDLK_*)
    public const int K_ESCAPE = 0x1b;
    public const int K_ENTER = 0x0d;
    public const int K_SPACE = 0x20;
    public const int K_1 = 0x31;
    public const int K_2 = 0x32;
    public const int K_3 = 0x33;
    public const int K_A = 0x61;
    public const int K_B = 0x62;
    public const int K_C = 0x63;
    public const int K_M = 0x6d;
    public const int K_F = 0x66;
    public const int K_H = 0x68;
    public const int K_K = 0x6b;
    public const int K_N = 0x6e;
    public const int K_P = 0x70;
    public const int K_R = 0x72;
    public const int K_S = 0x73;
    public const int K_T = 0x74;
    public const int K_V = 0x76;
    public const int K_DELETE = 0x7f;
    public const int K_LEFTBRACKET = 0x5b;
    public const int K_RIGHTBRACKET = 0x5d;
    public const int K_RIGHT = 1073741903;
    public const int K_LEFT = 1073741904;
    public const int K_DOWN = 1073741905;
    public const int K_UP = 1073741906;

    /// <summary>
    /// Blittable mirror of SDL_Event (56 bytes). Explicit layout with absolute
    /// field offsets, so no managed byte[] marshalling is involved (the old
    /// ByValArray + `out` version left the array null and corrupted the heap:
    /// "free(): invalid pointer").
    /// Layout refs (offset from event start):
    ///   WindowEvent: windowID@8, event@12, data1@16, data2@20
    ///   Keyboard:    windowID@8, state@12, repeat@13, scancode@16, sym@20
    ///   MouseMotion: windowID@8, which@12, state@16, x@20, y@24
    ///   MouseButton: windowID@8, which@12, button@16, state@17, x@20, y@24
    /// </summary>
    [StructLayout(LayoutKind.Explicit, Size = 56)]
    public struct SdlEvent
    {
        [FieldOffset(0)] public uint type;
        [FieldOffset(12)] public byte windowEventId;
        [FieldOffset(16)] public int windowData1;
        [FieldOffset(20)] public int windowData2;
        [FieldOffset(20)] public int keySym;
        [FieldOffset(20)] public int mouseX;
        [FieldOffset(24)] public int mouseY;
    }

    public static int KeySym(this SdlEvent e) => e.keySym;
    public static int MouseX(this SdlEvent e) => e.mouseX;
    public static int MouseY(this SdlEvent e) => e.mouseY;
    public static int WheelY(this SdlEvent e) => e.mouseX;
    public static int WindowEvent(this SdlEvent e) => e.windowEventId;
    public static int WindowData1(this SdlEvent e) => e.windowData1;
    public static int WindowData2(this SdlEvent e) => e.windowData2;

    [DllImport("SDL2", EntryPoint = "SDL_Init")]
    public static extern int SdlInit(uint flags);

    [DllImport("SDL2", EntryPoint = "SDL_Quit")]
    public static extern void SdlQuit();

    [DllImport("SDL2", EntryPoint = "SDL_CreateWindow", CharSet = CharSet.Ansi)]
    public static extern IntPtr SdlCreateWindow(string title, int x, int y, int w, int h, uint flags);

    [DllImport("SDL2", EntryPoint = "SDL_SetWindowTitle", CharSet = CharSet.Ansi)]
    public static extern void SdlSetWindowTitle(IntPtr window, string title);

    [DllImport("SDL2", EntryPoint = "SDL_GL_SetAttribute")]
    public static extern void SdlGlSetAttribute(int attr, int value);

    [DllImport("SDL2", EntryPoint = "SDL_GL_ResetAttributes")]
    public static extern void SdlGlResetAttributes();

    [DllImport("SDL2", EntryPoint = "SDL_GetCurrentVideoDriver")]
    private static extern IntPtr SdlGetCurrentVideoDriverRaw();

    public static string? SdlGetCurrentVideoDriver() =>
        Marshal.PtrToStringAnsi(SdlGetCurrentVideoDriverRaw());

    [DllImport("SDL2", EntryPoint = "SDL_GetNumVideoDrivers")]
    public static extern int SdlGetNumVideoDrivers();

    [DllImport("SDL2", EntryPoint = "SDL_GetVideoDriver")]
    private static extern IntPtr SdlGetVideoDriverRaw(int index);

    public static string? SdlGetVideoDriver(int index) =>
        Marshal.PtrToStringAnsi(SdlGetVideoDriverRaw(index));

    [DllImport("SDL2", EntryPoint = "SDL_GL_CreateContext")]
    public static extern IntPtr SdlGlCreateContext(IntPtr window);

    [DllImport("SDL2", EntryPoint = "SDL_GL_DeleteContext")]
    public static extern void SdlGlDeleteContext(IntPtr context);

    [DllImport("SDL2", EntryPoint = "SDL_GL_SwapWindow")]
    public static extern void SdlGlSwapWindow(IntPtr window);

    [DllImport("SDL2", EntryPoint = "SDL_GL_SetSwapInterval")]
    public static extern int SdlGlSetSwapInterval(int interval);

    [DllImport("SDL2", EntryPoint = "SDL_GL_GetDrawableSize")]
    public static extern void SdlGlGetDrawableSize(IntPtr window, out int w, out int h);

    [DllImport("SDL2", EntryPoint = "SDL_PumpEvents")]
    public static extern void SdlPumpEvents();

    [DllImport("SDL2", EntryPoint = "SDL_PollEvent")]
    public static extern int SdlPollEvent(out SdlEvent e);

    [DllImport("SDL2", EntryPoint = "SDL_GetError")]
    private static extern IntPtr SdlGetErrorRaw();

    // SDL_GetError returns a pointer to internal static storage — never free
    // it, just copy to a managed string.
    public static string SdlGetError() =>
        Marshal.PtrToStringAnsi(SdlGetErrorRaw()) ?? "";
}
