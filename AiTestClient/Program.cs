﻿using System;
using System.Diagnostics;
using AiTestClient.World;

namespace AiTestClient;

/// <summary>
/// Entry point + main game loop.
///
/// Speed modes:
///   NORMAL    - one simulation step per rendered frame (real time)
///   STEP      - paused; press SPACE to advance one step at a time
///   SUPERFAST - many simulation steps per rendered frame (watch it learn fast)
///
/// Keys: 1/2/3 camera, N/F/S speed, R retrain, T track editor, ESC quit.
/// </summary>
internal static class Program
{
    private enum Speed { Normal, Step, SuperFast }

    private static void Main(string[] args)
    {
        // Headless mode: run the AI + simulation with no window/GPU.
        //   --headless [steps]   (default 6000 steps)
        if (args.Length > 0 && args[0] == "--headless")
        {
            int steps = 6000;
            if (args.Length > 1 && int.TryParse(args[1], out var s)) steps = s;
            RunHeadless(steps, Array.Exists(args, arg => arg == "--kill-offtrack"));
            return;
        }

        try
        {
            using var window = new GameWindow("AI Car - Neural Driver (SDL2 + OpenGL)");
            var renderer = new Renderer();
            var hud = new Hud();
            var sim = new Simulation();

            renderer.Setup(window.Width, window.Height);

            Speed speed = Speed.Normal;
            bool editor = false;
            bool editorDirty = false;
            int draggingPoint = -1;
            int selectedPoint = -1;
            bool aiMenu = false;
            int aiMenuRow = 0;
            int hiddenLayers = 1;
            int hiddenNodes = 10;
            bool showBrain = true;
            bool brainFullscreen = false;
            bool showStatus = true;
            bool orbitPaused = false;

            var last = Stopwatch.GetTimestamp();

            while (!window.Quit)
            {
                window.ProcessEvents();

                if (window.KeyM)
                {
                    aiMenu = !aiMenu;
                    editor = false;
                    window.KeyM = false;
                }
                if (window.KeyB)
                {
                    showBrain = !showBrain;
                    if (!showBrain) brainFullscreen = false;
                    window.KeyB = false;
                }
                if (window.KeyV)
                {
                    brainFullscreen = !brainFullscreen;
                    showBrain = true;
                    window.KeyV = false;
                }
                if (window.KeyH)
                {
                    showStatus = !showStatus;
                    window.KeyH = false;
                }
                if (window.KeyP)
                {
                    orbitPaused = !orbitPaused;
                    window.KeyP = false;
                }
                if (window.KeyK)
                {
                    sim.NewGenerationOnOffTrack = !sim.NewGenerationOnOffTrack;
                    window.KeyK = false;
                }
                if (window.KeyEsc)
                {
                    if (aiMenu) aiMenu = false;
                    else window.Quit = true;
                    window.KeyEsc = false;
                }
                if (aiMenu)
                {
                    if (window.KeyUp) { aiMenuRow = (aiMenuRow + 1) % 2; window.KeyUp = false; }
                    if (window.KeyDown) { aiMenuRow = (aiMenuRow + 1) % 2; window.KeyDown = false; }
                    int delta = 0;
                    if (window.KeyLeft) { delta = -1; window.KeyLeft = false; }
                    if (window.KeyRight) { delta = 1; window.KeyRight = false; }
                    if (aiMenuRow == 0 && delta != 0)
                        hiddenLayers = delta > 0 ? checked(hiddenLayers + 1) : Math.Max(1, hiddenLayers - 1);
                    else if (aiMenuRow == 1 && delta != 0)
                        hiddenNodes = delta > 0 ? checked(hiddenNodes + 1) : Math.Max(1, hiddenNodes - 1);
                    if (window.KeyEnter)
                    {
                        sim.ReconfigureBrain(hiddenLayers, hiddenNodes);
                        window.KeyEnter = false;
                        aiMenu = false;
                    }
                }

                // ---- input: mode switching (edge-triggered) ----
                if (window.Key1) { renderer.Mode = CameraMode.Orbit; window.Key1 = false; }
                if (window.Key2) { renderer.Mode = CameraMode.Chase; window.Key2 = false; }
                if (window.Key3) { renderer.Mode = CameraMode.Free; window.Key3 = false; }
                if (window.KeyN) { speed = Speed.Normal; window.KeyN = false; }
                if (window.KeyF) { speed = Speed.Step; window.KeyF = false; }
                if (window.KeyS) { speed = Speed.SuperFast; window.KeyS = false; }
                if (window.KeyR) { sim.RetrainBrain(); window.KeyR = false; }
                if (window.KeyT)
                {
                    editor = !editor;
                    draggingPoint = -1;
                    selectedPoint = editor && sim.Track.ControlPoints.Count > 0 ? 0 : -1;
                    window.KeyT = false;
                }

                // ---- track editor: drag control points ----
                if (editor)
                {
                    if (window.KeyA && selectedPoint >= 0)
                    {
                        selectedPoint = sim.Track.InsertSectionAfter(selectedPoint);
                        sim.ResetCar();
                        window.KeyA = false;
                    }
                    if (window.KeyDelete)
                    {
                        if (sim.Track.RemoveSection(selectedPoint))
                        {
                            selectedPoint = Math.Min(selectedPoint, sim.Track.ControlPoints.Count - 1);
                            sim.ResetCar();
                        }
                        window.KeyDelete = false;
                    }
                    if (window.KeyLeftBracket)
                    {
                        sim.Track.AdjustWidth(-0.5f);
                        sim.ResetCar();
                        window.KeyLeftBracket = false;
                    }
                    if (window.KeyRightBracket)
                    {
                        sim.Track.AdjustWidth(0.5f);
                        sim.ResetCar();
                        window.KeyRightBracket = false;
                    }

                    if (window.MousePressed && draggingPoint < 0)
                    {
                        if (renderer.UnprojectToGround(window.Width, window.Height, window.MouseX, window.MouseY, out var wx, out var wz))
                        {
                            int best = -1; float bestD = 3f * 3f;
                            for (int i = 0; i < sim.Track.ControlPoints.Count; i++)
                            {
                                var p = sim.Track.ControlPoints[i];
                                float dx = p.x - wx, dz = p.z - wz;
                                float d2 = dx * dx + dz * dz;
                                if (d2 < bestD) { bestD = d2; best = i; }
                            }
                            selectedPoint = best;
                            draggingPoint = best;
                        }
                    }
                    if (draggingPoint >= 0)
                    {
                        if (window.MouseDown && renderer.UnprojectToGround(window.Width, window.Height, window.MouseX, window.MouseY, out var wx, out var wz))
                        {
                            sim.Track.ControlPoints[draggingPoint] = (wx, wz);
                            editorDirty = true;
                        }
                        else
                        {
                            draggingPoint = -1;
                        }
                    }
                    if (editorDirty)
                    {
                        sim.Track.Rebuild();
                        sim.ResetCar();
                        editorDirty = false;
                    }
                }

                // ---- simulation ----
                // Normal speed is paced to real time (60 sim steps/sec): the loop
                // used to run unthrottled (1 step per frame at unlimited fps),
                // which made "Normal" a blur. Vsync (see GameWindow) caps the
                // frame rate when available; this clock caps the step rate even
                // when it isn't, and sleeps instead of burning CPU.
                float dt = (float)(Stopwatch.GetTimestamp() - last) / (float)Stopwatch.Frequency;
                last = Stopwatch.GetTimestamp();
                dt = Math.Min(dt, 0.1f);

                switch (speed)
                {
                    case Speed.Normal:
                        if (!editor && !aiMenu && PaceNormalStep()) sim.Step();
                        break;
                    case Speed.Step:
                        if (!aiMenu && window.KeySpace)
                        {
                            sim.Step();
                            window.KeySpace = false;
                        }
                        // Keep the pacer resynced so switching back to Normal
                        // doesn't burst through catch-up steps.
                        PaceReset();
                        break;
                    case Speed.SuperFast:
                        if (!aiMenu) for (int i = 0; i < 200; i++) sim.Step();
                        PaceReset();
                        break;
                }

                // ---- render ----
                renderer.UpdateCamera(sim, dt, window.MouseX, window.MouseY,
                    window.MouseDown && !editor && !aiMenu, window.MouseWheelY, orbitPaused);
                renderer.BeginFrame(window.Width, window.Height);
                renderer.Render(sim);
                if (editor) renderer.RenderEditorOverlay(sim.Track, selectedPoint);
                hud.Draw(window.Width, window.Height, sim, renderer, editor, selectedPoint,
                    aiMenu, aiMenuRow, hiddenLayers, hiddenNodes, showBrain, brainFullscreen, showStatus,
                    orbitPaused);

                window.Swap();
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine("Fatal: " + ex);
        }
    }

    // ---- Normal-speed pacer: 60 sim steps/sec real time ----
    private static long _paceNextTick;

    /// <summary>
    /// Returns true when a Normal-speed sim step is due (~60Hz). Sleeps
    /// briefly when ahead of schedule so the loop idles instead of spinning.
    /// </summary>
    private static bool PaceNormalStep()
    {
        long freq = Stopwatch.Frequency;
        long interval = freq / 60;
        long now = Stopwatch.GetTimestamp();
        if (_paceNextTick == 0) { _paceNextTick = now + interval; return true; }
        if (now >= _paceNextTick)
        {
            _paceNextTick += interval;
            // Stalled (menu, drag, slow frame): resync rather than bursting.
            if (now > _paceNextTick + 5 * interval) _paceNextTick = now + interval;
            return true;
        }
        long waitMs = (_paceNextTick - now) * 1000 / freq;
        if (waitMs > 0) Thread.Sleep((int)Math.Min(waitMs, 15));
        return false;
    }

    private static void PaceReset() => _paceNextTick = 0;

    /// <summary>
    /// Runs the AI + simulation with no window or GPU. Useful for verifying the
    /// learning loop headlessly (e.g. on a machine whose GPU is busy) and for
    /// benchmarking. Prints periodic progress so you can watch the car improve.
    /// </summary>
    private static void RunHeadless(int steps, bool killOffTrack)
    {
        var sim = new Simulation();
        sim.NewGenerationOnOffTrack = killOffTrack;
        Console.WriteLine($"Headless: {steps} steps, 5-ray vision, net 5-10-2");
        Console.WriteLine("      step  lap   speed offTrack  totalReward  bestProg");

        for (int i = 0; i < steps; i++)
        {
            sim.Step();

            if (i % 500 == 0 || i == steps - 1)
            {
                Console.WriteLine($"  {i,8}  {sim.Laps,3}  {sim.Car.Speed,6:F2}  {(sim.OffTrack ? "yes" : "no"),8}  {sim.TotalReward,12:F1}  {sim.BestLapProgress,8:F3}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"Done. Generation: {sim.Generation}, laps completed: {sim.Laps}, best lap progress: {sim.BestLapProgress:F3}, total reward: {sim.TotalReward:F1}");
    }
}
