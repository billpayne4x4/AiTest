using System;
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
            RunHeadless(steps, Array.Exists(args, arg => arg == "--kill-offtrack"),
                Array.Exists(args, arg => arg == "--gpu"));
            return;
        }
        // GPU parity self-test: same-seed CPU vs GPU math comparison, no window.
        if (Array.Exists(args, arg => arg == "--selftest-gpu"))
        {
            GpuSelfTest.Run();
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
            bool physMenu = false, rewMenu = false;
            int physRow = 0, rewRow = 0;
            int aiMenuRow = 0;
            const int AiMenuRows = 9;
            int hiddenLayers = 2;
            int hiddenNodes = 32;
            var trainCfg = new AiModel_V1.TrainingConfig();
            bool showBrain = true;
            bool brainFullscreen = false;
            bool showStatus = true;
            bool orbitPaused = false;
            // steps/sec rolling meter + compute-device line for the HUD
            long spsSteps = 0;
            long spsTick = Stopwatch.GetTimestamp();
            double stepsPerSec = 0;

            if (args.Length > 0 && Array.Exists(args, arg => arg == "--gpu"))
            {
                if (sim.Brain.Net.TryEnableGpu(out string gpuMsg)) Console.WriteLine("Compute: " + gpuMsg);
                else Console.WriteLine("Compute: " + gpuMsg + " (staying on CPU)");
            }

            var last = Stopwatch.GetTimestamp();

            while (!window.Quit)
            {
                window.ProcessEvents();

                if (window.KeyM || window.KeyC)
                {
                    aiMenu = !aiMenu;
                    physMenu = rewMenu = false;
                    editor = false;
                    window.KeyM = false;
                    window.KeyC = false;
                }
                if (window.KeyE)
                {
                    physMenu = !physMenu;
                    aiMenu = rewMenu = false;
                    editor = false;
                    window.KeyE = false;
                }
                if (window.KeyW)
                {
                    rewMenu = !rewMenu;
                    aiMenu = physMenu = false;
                    editor = false;
                    window.KeyW = false;
                }
                if (window.KeyD)
                {
                    // CPU <-> GPU toggle: weights move seamlessly both ways.
                    if (sim.Brain.Net.IsGpu) { sim.Brain.Net.DisableGpu(); Console.WriteLine("Compute: CPU"); }
                    else if (sim.Brain.Net.TryEnableGpu(out string dmsg)) Console.WriteLine("Compute: " + dmsg);
                    else Console.WriteLine("Compute: " + dmsg);
                    window.KeyD = false;
                }
                // Clickable menu buttons (bottom-left, above controls panel).
                if (window.MousePressed && !aiMenu && !physMenu && !rewMenu)
                {
                    if (Hud.AiButtonHit(window.Width, window.Height, window.MouseX, window.MouseY)) { aiMenu = true; editor = false; }
                    else if (Hud.PhysButtonHit(window.Width, window.Height, window.MouseX, window.MouseY)) { physMenu = true; editor = false; }
                    else if (Hud.RewButtonHit(window.Width, window.Height, window.MouseX, window.MouseY)) { rewMenu = true; editor = false; }
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
                    else if (physMenu) physMenu = false;
                    else if (rewMenu) rewMenu = false;
                    else window.Quit = true;
                    window.KeyEsc = false;
                }
                if (aiMenu)
                {
                    if (window.KeyUp) { aiMenuRow = (aiMenuRow + AiMenuRows - 1) % AiMenuRows; window.KeyUp = false; }
                    if (window.KeyDown) { aiMenuRow = (aiMenuRow + 1) % AiMenuRows; window.KeyDown = false; }
                    int delta = 0;
                    if (window.KeyLeft) { delta = -1; window.KeyLeft = false; }
                    if (window.KeyRight) { delta = 1; window.KeyRight = false; }
                    if (delta != 0)
                    {
                        switch (aiMenuRow)
                        {
                            case 0: hiddenLayers = delta > 0 ? checked(hiddenLayers + 1) : Math.Max(1, hiddenLayers - 1); break;
                            case 1: hiddenNodes = delta > 0 ? checked(hiddenNodes + 1) : Math.Max(1, hiddenNodes - 1); break;
                            case 2: trainCfg.UseResidual = !trainCfg.UseResidual; break;
                            case 3: trainCfg.UseLayerNorm = !trainCfg.UseLayerNorm; break;
                            case 4: trainCfg.Activation = (trainCfg.Activation + delta + 3) % 3; break;
                            case 5: trainCfg.UseAdam = !trainCfg.UseAdam; break;
                            case 6: trainCfg.Init = (trainCfg.Init + delta + 3) % 3; break;
                            case 7: trainCfg.GradClip = trainCfg.GradClip <= 0f ? 1f : trainCfg.GradClip >= 5f ? 0f : trainCfg.GradClip + delta; break;
                            case 8: trainCfg.UsePpo = !trainCfg.UsePpo; break;
                        }
                    }
                    if (window.KeyEnter)
                    {
                        bool wasGpu = sim.Brain.Net.IsGpu;
                        sim.ReconfigureBrain(hiddenLayers, hiddenNodes, trainCfg.Clone());
                        // Seamless: stay on the GPU across brain swaps when active.
                        if (wasGpu && !sim.Brain.Net.TryEnableGpu(out string rmsg))
                            Console.WriteLine("Compute: " + rmsg + " (staying on CPU)");
                        window.KeyEnter = false;
                        aiMenu = false;
                    }
                }
                if (physMenu)
                {
                    const int PhysRows = 7;
                    if (window.KeyUp) { physRow = (physRow + PhysRows - 1) % PhysRows; window.KeyUp = false; }
                    if (window.KeyDown) { physRow = (physRow + 1) % PhysRows; window.KeyDown = false; }
                    float delta = 0;
                    if (window.KeyLeft) { delta = -1; window.KeyLeft = false; }
                    if (window.KeyRight) { delta = 1; window.KeyRight = false; }
                    if (delta != 0)
                    {
                        var p = sim.PhysicsCfg;
                        switch (physRow)
                        {
                            case 0: p.Realistic = !p.Realistic; break;
                            case 1: p.Grip = Math.Clamp(p.Grip + delta * 2f, 2f, 60f); break;
                            case 2: p.Downforce = Math.Clamp(p.Downforce + delta * 0.05f, 0f, 2f); break;
                            case 3: p.Steering = Math.Clamp(p.Steering + delta * 0.05f, 0.1f, 1.2f); break;
                            case 4: p.Oversteer = Math.Clamp(p.Oversteer + delta * 0.25f, 0f, 3f); break;
                            case 5: p.SpinThreshold = Math.Clamp(p.SpinThreshold + delta, 3f, 25f); break;
                            case 6: p.Understeer = !p.Understeer; break;
                        }
                    }
                    if (window.KeyEnter) { physMenu = false; window.KeyEnter = false; }
                }
                if (rewMenu)
                {
                    const int RewRows = 11;
                    if (window.KeyUp) { rewRow = (rewRow + RewRows - 1) % RewRows; window.KeyUp = false; }
                    if (window.KeyDown) { rewRow = (rewRow + 1) % RewRows; window.KeyDown = false; }
                    float delta = 0;
                    if (window.KeyLeft) { delta = -1; window.KeyLeft = false; }
                    if (window.KeyRight) { delta = 1; window.KeyRight = false; }
                    if (delta != 0)
                    {
                        var r = sim.RewardCfg;
                        switch (rewRow)
                        {
                            case 0: r.Speed = Math.Max(0f, r.Speed + delta * 0.5f); break;
                            case 1: r.Centering = Math.Max(0f, r.Centering + delta * 0.25f); break;
                            case 2: r.Alignment = Math.Max(0f, r.Alignment + delta * 0.25f); break;
                            case 3: r.Progress = Math.Max(0f, r.Progress + delta * 250f); break;
                            case 4: r.WrongWay = Math.Max(0f, r.WrongWay + delta * 0.25f); break;
                            case 5: r.Slide = Math.Max(0f, r.Slide + delta); break;
                            case 6: r.Spin = Math.Max(0f, r.Spin + delta); break;
                            case 7: r.Understeer = Math.Max(0f, r.Understeer + delta * 0.5f); break;
                            case 8: r.SteerEffort = Math.Max(0f, r.SteerEffort + delta * 0.02f); break;
                            case 9: r.OffTrack = Math.Max(0f, r.OffTrack + delta * 2f); break;
                            case 10: r.Parking = Math.Max(0f, r.Parking + delta * 0.5f); break;
                        }
                    }
                    if (window.KeyEnter) { rewMenu = false; window.KeyEnter = false; }
                }

                // ---- input: mode switching (edge-triggered) ----
                if (window.Key1) { renderer.Mode = CameraMode.Orbit; window.Key1 = false; }
                if (window.Key2) { renderer.Mode = CameraMode.Chase; window.Key2 = false; }
                if (window.Key3) { renderer.Mode = CameraMode.Free; window.Key3 = false; }
                if (window.KeyN) { speed = Speed.Normal; window.KeyN = false; }
                if (window.KeyF) { speed = Speed.Step; window.KeyF = false; }
                if (window.KeyS) { speed = Speed.SuperFast; window.KeyS = false; }
                if (window.KeyR) { sim.RetrainBrain(); window.KeyR = false; }
                if (window.KeyG)
                {
                    sim.Track.Randomize(Environment.TickCount);
                    sim.ResetCar();
                    window.KeyG = false;
                }
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
                            // Reject drops that would twist the road (curb
                            // crossover): the point simply stays put.
                            if (sim.Track.IsValidControlPoint(draggingPoint, wx, wz))
                            {
                                sim.Track.ControlPoints[draggingPoint] = (wx, wz);
                                editorDirty = true;
                            }
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
                        if (!editor && !aiMenu && !physMenu && !rewMenu && PaceNormalStep()) sim.Step();
                        break;
                    case Speed.Step:
                        if (!aiMenu && !physMenu && !rewMenu && window.KeySpace)
                        {
                            sim.Step();
                            window.KeySpace = false;
                        }
                        // Keep the pacer resynced so switching back to Normal
                        // doesn't burst through catch-up steps.
                        PaceReset();
                        break;
                    case Speed.SuperFast:
                        if (!aiMenu && !physMenu && !rewMenu) for (int i = 0; i < 200; i++) sim.Step();
                        PaceReset();
                        break;
                }

                // ---- render ----
                // steps/sec meter: rolling window over simulation steps
                {
                    long nowSps = Stopwatch.GetTimestamp();
                    double elapsed = (double)(nowSps - spsTick) / Stopwatch.Frequency;
                    if (elapsed >= 0.5)
                    {
                        stepsPerSec = (sim.Steps - spsSteps) / elapsed;
                        if (stepsPerSec < 0) stepsPerSec = 0; // counter reset (retrain)
                        spsSteps = sim.Steps;
                        spsTick = nowSps;
                    }
                }
                renderer.UpdateCamera(sim, dt, window.MouseX, window.MouseY,
                    window.MouseDown && !editor && !aiMenu && !physMenu && !rewMenu, window.MouseWheelY, orbitPaused);
                renderer.BeginFrame(window.Width, window.Height);
                renderer.Render(sim);
                if (editor) renderer.RenderEditorOverlay(sim.Track, selectedPoint);
                hud.Draw(window.Width, window.Height, sim, renderer, editor, selectedPoint,
                    aiMenu, aiMenuRow, hiddenLayers, hiddenNodes, trainCfg,
                    physMenu, physRow, rewMenu, rewRow,
                    showBrain, brainFullscreen, showStatus, orbitPaused,
                    sim.Brain.Net.DeviceLabel, sim.Brain.Net.GpuVramMb, stepsPerSec);

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
    private static void RunHeadless(int steps, bool killOffTrack, bool useGpu)
    {
        var sim = new Simulation();
        sim.NewGenerationOnOffTrack = killOffTrack;
        if (useGpu && !sim.Brain.Net.TryEnableGpu(out string gpuMsg))
            Console.WriteLine("Compute: " + gpuMsg + " (staying on CPU)");
        Console.WriteLine($"Headless: {steps} steps, 5-ray vision, net 7-32-32-3, device {sim.Brain.Net.DeviceLabel}");
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
