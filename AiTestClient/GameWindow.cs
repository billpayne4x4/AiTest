using System;
using System.Collections.Generic;
using AiTestClient.Native;

namespace AiTestClient;

/// <summary>
/// Owns the SDL window + OpenGL context and pumps input events into simple
/// boolean / coordinate state that the game loop reads.
/// </summary>
public class GameWindow : IDisposable
{
    public IntPtr Handle { get; }
    public IntPtr GlContext { get; }
    public int Width { get; private set; }
    public int Height { get; private set; }

    // keyboard: held state + sticky press latches.
    // Held bools track physical state, but at 2 steps/sec a whole tap can
    // land between frames (down+up = lost press). So every non-repeat keydown
    // also drops its sym into _pressed, which the game loop consumes exactly
    // once via ConsumePress — taps survive arbitrarily slow frames.
    public bool KeyEsc, KeyEnter, KeySpace;
    public bool Key1, Key2, Key3;
    public bool KeyA, KeyB, KeyC, KeyDelete, KeyDown, KeyD, KeyE, KeyF, KeyG, KeyH, KeyK, KeyLeft, KeyLeftBracket, KeyM, KeyN, KeyP, KeyR, KeyRight, KeyRightBracket, KeyS, KeyT, KeyUp, KeyV, KeyW;
    private readonly HashSet<int> _pressed = new();

    /// <summary>True once per physical key press (auto-repeat excluded).</summary>
    public bool ConsumePress(int sym)
    {
        if (_pressed.Contains(sym)) { _pressed.Remove(sym); return true; }
        return false;
    }

    // mouse
    public int MouseX, MouseY;
    public bool MouseDown;
    public bool MousePressed;
    public int MouseWheelY;

    public bool Quit;

    public GameWindow(string title, int w = 1920, int h = 1080)
    {
        NativeLibs.Install();

        // SDL on Wayland uses EGL, and requesting an explicit compatibility
        // profile (e.g. GL 3.0 compat) often yields no matching EGLConfig:
        //   "Couldn't find matching EGL config (call to eglChooseConfig ...)"
        // XWayland (SDL's "x11" driver) + GLX handles compat profiles fine
        // (verified: Mesa Intel exposes compat 4.6 over GLX here).
        // So: try the default driver first, then fall back to x11, then
        // wayland; and within each driver try GL attribute sets from most
        // lenient to most specific.
        string? userDriver = Environment.GetEnvironmentVariable("SDL_VIDEODRIVER");
        var drivers = new List<string?>();
        if (!string.IsNullOrEmpty(userDriver))
        {
            drivers.Add(userDriver); // respect explicit user choice, no fallback games
        }
        else
        {
            drivers.Add(null); // default (usually wayland when WAYLAND_DISPLAY is set)
            drivers.Add("x11"); // XWayland/GLX — best for fixed-function compat
            drivers.Add("wayland");
        }

        string lastError = "";
        string triedDrivers = "";
        foreach (var driver in drivers)
        {
            // Switching video drivers requires re-init.
            Sdl.SdlQuit();
            if (driver is null)
                Environment.SetEnvironmentVariable("SDL_VIDEODRIVER", null);
            else
                Environment.SetEnvironmentVariable("SDL_VIDEODRIVER", driver);

            if (Sdl.SdlInit(Sdl.INIT_VIDEO) != 0)
            {
                lastError = Sdl.SdlGetError();
                triedDrivers += (driver ?? "<default>") + " (init: " + lastError + "); ";
                continue;
            }

            string current = Sdl.SdlGetCurrentVideoDriver() ?? (driver ?? "<default>");
            triedDrivers += current + "; ";

            // Attribute sets, in order. Fixed-function (glBegin/glEnd) needs
            // a compatibility-profile context, but explicitly requesting
            // 3.0-compat breaks EGL config selection on Wayland. The default
            // (no version request) typically yields a 2.x/3.x compat context
            // that supports immediate mode, so try that first.
            var attrSets = new (string name, Action apply)[]
            {
                ("default-compat", () =>
                {
                    Sdl.SdlGlResetAttributes();
                    Sdl.SdlGlSetAttribute(Sdl.GL_DOUBLEBUFFER, 1);
                    Sdl.SdlGlSetAttribute(Sdl.GL_DEPTH_SIZE, 24);
                }),
                ("gl2.1-compat", () =>
                {
                    Sdl.SdlGlResetAttributes();
                    Sdl.SdlGlSetAttribute(Sdl.GL_DOUBLEBUFFER, 1);
                    Sdl.SdlGlSetAttribute(Sdl.GL_DEPTH_SIZE, 24);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_MAJOR_VERSION, 2);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_MINOR_VERSION, 1);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_PROFILE_MASK, Sdl.GL_CONTEXT_PROFILE_COMPATIBILITY);
                }),
                ("gl3.0-compat", () =>
                {
                    Sdl.SdlGlResetAttributes();
                    Sdl.SdlGlSetAttribute(Sdl.GL_DOUBLEBUFFER, 1);
                    Sdl.SdlGlSetAttribute(Sdl.GL_DEPTH_SIZE, 24);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_MAJOR_VERSION, 3);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_MINOR_VERSION, 0);
                    Sdl.SdlGlSetAttribute(Sdl.GL_CONTEXT_PROFILE_MASK, Sdl.GL_CONTEXT_PROFILE_COMPATIBILITY);
                }),
                ("minimal", () =>
                {
                    // Last resort: no depth buffer requirement at all.
                    Sdl.SdlGlResetAttributes();
                    Sdl.SdlGlSetAttribute(Sdl.GL_DOUBLEBUFFER, 1);
                }),
            };

            foreach (var (name, apply) in attrSets)
            {
                apply();
                Handle = Sdl.SdlCreateWindow(title, (int)Sdl.WINDOWPOS_CENTERED, (int)Sdl.WINDOWPOS_CENTERED, w, h,
                    Sdl.WINDOW_OPENGL | Sdl.WINDOW_RESIZABLE | Sdl.WINDOW_ALLOW_HIGHDPI);
                if (Handle != IntPtr.Zero)
                {
                    GlContext = Sdl.SdlGlCreateContext(Handle);
                    if (GlContext != IntPtr.Zero)
                    {
                        Width = w; Height = h;
                        Sdl.SdlGlGetDrawableSize(Handle, out int dw, out int dh);
                        Width = dw; Height = dh;
                        // Best effort vsync: caps Swap() at the display refresh so
                        // Normal speed renders at display rate. Ignored when the
                        // driver doesn't support it — Program.cs also paces the
                        // simulation clock, so nothing depends on this alone.
                        try { Sdl.SdlGlSetSwapInterval(1); } catch { }
                        return;
                    }
                    lastError = "SDL_GL_CreateContext (" + name + "/" + current + ") failed: " + Sdl.SdlGetError();
                    Sdl.SdlGlResetAttributes();
                }
                else
                {
                    lastError = "SDL_CreateWindow (" + name + "/" + current + ") failed: " + Sdl.SdlGetError();
                }
            }
            // This driver didn't work with any attribute set; try the next one.
        }

        throw new Exception(
            lastError + " [tried drivers: " + triedDrivers + "] " +
            "Hint: Wayland+EGL often rejects compatibility-profile GL. " +
            "Run with SDL_VIDEODRIVER=x11 (needs XWayland/:0), or headless with '--headless'. " +
            "DISPLAY=" + Environment.GetEnvironmentVariable("DISPLAY") +
            " WAYLAND_DISPLAY=" + Environment.GetEnvironmentVariable("WAYLAND_DISPLAY"));
    }

    public void ProcessEvents()
    {
        // NOTE: MousePressed/MouseWheelY are edge flags cleared by EndFrame()
        // (after render), not here — ProcessEvents may run several times per
        // frame (slow SuperFast steps), and clearing here would eat clicks.
        Sdl.SdlPumpEvents();
        while (Sdl.SdlPollEvent(out var e) == 1)
        {
            switch (e.type)
            {
                case Sdl.EV_QUIT:
                    Quit = true;
                    break;
                case Sdl.EV_KEYDOWN:
                    HandleKey(e.KeySym(), true);
                    if (!e.KeyRepeat()) _pressed.Add(e.KeySym());
                    break;
                case Sdl.EV_KEYUP:
                    HandleKey(e.KeySym(), false);
                    break;
                case Sdl.EV_MOUSEMOTION:
                    MouseX = e.MouseX();
                    MouseY = e.MouseY();
                    break;
                case Sdl.EV_MOUSEBUTTONDOWN:
                    MouseDown = true;
                    MousePressed = true;
                    MouseX = e.MouseX();
                    MouseY = e.MouseY();
                    break;
                case Sdl.EV_MOUSEBUTTONUP:
                    MouseDown = false;
                    break;
                case Sdl.EV_MOUSEWHEEL:
                    MouseWheelY += e.WheelY();
                    break;
                case Sdl.EV_WINDOWEVENT:
                    if (e.WindowEvent() == Sdl.WINDOWEVENT_RESIZED || e.WindowEvent() == Sdl.WINDOWEVENT_SIZE_CHANGED)
                    {
                        Sdl.SdlGlGetDrawableSize(Handle, out int dw, out int dh);
                        Width = dw; Height = dh;
                    }
                    break;
            }
        }
        // This also catches compositor/HiDPI changes that do not arrive as a
        // logical SDL window-size event.
        Sdl.SdlGlGetDrawableSize(Handle, out int drawableWidth, out int drawableHeight);
        if (drawableWidth > 0 && drawableHeight > 0)
        {
            Width = drawableWidth;
            Height = drawableHeight;
        }
    }

    private void HandleKey(int sym, bool down)
    {
        switch (sym)
        {
            case Sdl.K_ESCAPE: KeyEsc = down; break;
            case Sdl.K_ENTER: KeyEnter = down; break;
            case Sdl.K_SPACE: KeySpace = down; break;
            case Sdl.K_1: Key1 = down; break;
            case Sdl.K_2: Key2 = down; break;
            case Sdl.K_3: Key3 = down; break;
            case Sdl.K_A: KeyA = down; break;
            case Sdl.K_B: KeyB = down; break;
            case Sdl.K_C: KeyC = down; break;
            case Sdl.K_D: KeyD = down; break;
            case Sdl.K_E: KeyE = down; break;
            case Sdl.K_M: KeyM = down; break;
            case Sdl.K_DELETE: KeyDelete = down; break;
            case Sdl.K_F: KeyF = down; break;
            case Sdl.K_G: KeyG = down; break;
            case Sdl.K_H: KeyH = down; break;
            case Sdl.K_K: KeyK = down; break;
            case Sdl.K_N: KeyN = down; break;
            case Sdl.K_P: KeyP = down; break;
            case Sdl.K_R: KeyR = down; break;
            case Sdl.K_S: KeyS = down; break;
            case Sdl.K_T: KeyT = down; break;
            case Sdl.K_V: KeyV = down; break;
            case Sdl.K_W: KeyW = down; break;
            case Sdl.K_LEFTBRACKET: KeyLeftBracket = down; break;
            case Sdl.K_RIGHTBRACKET: KeyRightBracket = down; break;
            case Sdl.K_RIGHT: KeyRight = down; break;
            case Sdl.K_LEFT: KeyLeft = down; break;
            case Sdl.K_DOWN: KeyDown = down; break;
            case Sdl.K_UP: KeyUp = down; break;
        }
    }

    public void Swap() => Sdl.SdlGlSwapWindow(Handle);

    /// <summary>Call once per frame after render/consume: clears edge flags.</summary>
    public void EndFrame()
    {
        MousePressed = false;
        MouseWheelY = 0;
    }

    public void Dispose()
    {
        if (GlContext != IntPtr.Zero) Sdl.SdlGlDeleteContext(GlContext);
        Sdl.SdlQuit();
    }
}
