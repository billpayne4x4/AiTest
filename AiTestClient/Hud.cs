using System;
using System.Collections.Generic;
using System.Globalization;
using AiTestClient.Native;
using AiTestClient.World;

namespace AiTestClient;

/// <summary>
/// Screen-space HUD: a small 5x7 bitmap font, the stats readout, a live
/// visualization of the AI brain (nodes light up as they activate, weight
/// lines are colored by sign/magnitude), and the track-editor control points.
/// </summary>
public class Hud
{
    // 5x7 font: each glyph is 7 rows of 5 bits (bit4 = leftmost).
    private static readonly Dictionary<char, byte[]> Font = BuildFont();

    private static Dictionary<char, byte[]> BuildFont()
    {
        var f = new Dictionary<char, byte[]>
        {
            { 'A', G(0x0E,0x11,0x11,0x1F,0x11,0x11,0x11) },
            { 'B', G(0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E) },
            { 'C', G(0x0E,0x11,0x10,0x10,0x10,0x11,0x0E) },
            { 'D', G(0x1E,0x11,0x11,0x11,0x11,0x11,0x1E) },
            { 'E', G(0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F) },
            { 'F', G(0x1F,0x10,0x10,0x1E,0x10,0x10,0x10) },
            { 'G', G(0x0E,0x11,0x10,0x17,0x11,0x11,0x0F) },
            { 'H', G(0x11,0x11,0x11,0x1F,0x11,0x11,0x11) },
            { 'I', G(0x0E,0x04,0x04,0x04,0x04,0x04,0x0E) },
            { 'J', G(0x07,0x02,0x02,0x02,0x02,0x12,0x0C) },
            { 'K', G(0x11,0x12,0x14,0x18,0x14,0x12,0x11) },
            { 'L', G(0x10,0x10,0x10,0x10,0x10,0x10,0x1F) },
            { 'M', G(0x11,0x1B,0x15,0x15,0x11,0x11,0x11) },
            { 'N', G(0x11,0x19,0x15,0x13,0x11,0x11,0x11) },
            { 'O', G(0x0E,0x11,0x11,0x11,0x11,0x11,0x0E) },
            { 'P', G(0x1E,0x11,0x11,0x1E,0x10,0x10,0x10) },
            { 'Q', G(0x0E,0x11,0x11,0x11,0x15,0x12,0x0D) },
            { 'R', G(0x1E,0x11,0x11,0x1E,0x14,0x12,0x11) },
            { 'S', G(0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E) },
            { 'T', G(0x1F,0x04,0x04,0x04,0x04,0x04,0x04) },
            { 'U', G(0x11,0x11,0x11,0x11,0x11,0x11,0x0E) },
            { 'V', G(0x11,0x11,0x11,0x11,0x11,0x0A,0x04) },
            { 'W', G(0x11,0x11,0x11,0x15,0x15,0x1B,0x11) },
            { 'X', G(0x11,0x11,0x0A,0x04,0x0A,0x11,0x11) },
            { 'Y', G(0x11,0x11,0x0A,0x04,0x04,0x04,0x04) },
            { 'Z', G(0x1F,0x01,0x02,0x04,0x08,0x10,0x1F) },
            { '0', G(0x0E,0x11,0x13,0x15,0x19,0x11,0x0E) },
            { '1', G(0x04,0x0C,0x04,0x04,0x04,0x04,0x0E) },
            { '2', G(0x0E,0x11,0x01,0x02,0x04,0x08,0x1F) },
            { '3', G(0x1F,0x02,0x04,0x02,0x01,0x11,0x0E) },
            { '4', G(0x02,0x06,0x0A,0x12,0x1F,0x02,0x02) },
            { '5', G(0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E) },
            { '6', G(0x06,0x08,0x10,0x1E,0x11,0x11,0x0E) },
            { '7', G(0x1F,0x01,0x02,0x04,0x08,0x08,0x08) },
            { '8', G(0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E) },
            { '9', G(0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C) },
            { ' ', G(0,0,0,0,0,0,0) },
            { '.', G(0,0,0,0,0,0x0C,0x0C) },
            { ',', G(0,0,0,0,0x0C,0x0C,0x08) },
            { ':', G(0,0x0C,0x0C,0,0x0C,0x0C,0) },
            { '-', G(0,0,0,0x1F,0,0,0) },
            { '/', G(0x01,0x02,0x04,0x04,0x08,0x10,0x10) },
            { '%', G(0x19,0x1A,0x02,0x04,0x08,0x0B,0x13) },
            { '(', G(0x02,0x04,0x08,0x08,0x08,0x04,0x02) },
            { ')', G(0x08,0x04,0x02,0x02,0x02,0x04,0x08) },
            { '=', G(0,0,0x1F,0,0x1F,0,0) },
            { '+', G(0,0x04,0x04,0x1F,0x04,0x04,0) },
            { '>', G(0x08,0x04,0x02,0x01,0x02,0x04,0x08) },
            { '<', G(0x02,0x04,0x08,0x10,0x08,0x04,0x02) },
            { 'x', G(0,0,0x11,0x0A,0x04,0x0A,0x11) },
        };
        return f;
    }

    private static byte[] G(params byte[] rows) => rows;

    private void Begin2D(int w, int h)
    {
        Gl.PushAttrib(Gl.ALL_ATTRIB_BITS);
        Gl.Disable(Gl.DEPTH_TEST);
        Gl.Disable(Gl.LIGHTING);
        Gl.MatrixMode(Gl.PROJECTION);
        Gl.Viewport(0, 0, w, h);
        // Pixel coordinates with (0,0) at the top-left. OpenGL clip space is
        // -1..1, so scale by the drawable size as well as flipping Y.
        Gl.LoadMatrixf(new float[]
        {
            2f / w, 0, 0, 0,
            0, -2f / h, 0, 0,
            0, 0, -1f, 0,
            -1f, 1f, 0, 1f
        });
        Gl.MatrixMode(Gl.MODELVIEW);
        Gl.LoadIdentity();
    }

    private void End2D() => Gl.PopAttrib();

    public void DrawText(int x, int y, string text, float scale, float r, float g, float b)
    {
        Gl.Color3f(r, g, b);
        Gl.Begin(Gl.QUADS);
        int cx = x;
        foreach (var ch in text)
        {
            if (!Font.TryGetValue(char.ToUpperInvariant(ch), out var glyph))
            {
                cx += (int)(6 * scale);
                continue;
            }
            for (int row = 0; row < 7; row++)
            {
                byte bits = glyph[row];
                for (int col = 0; col < 5; col++)
                {
                    if ((bits & (1 << (4 - col))) != 0)
                    {
                        float px = cx + col * scale;
                        float py = y + row * scale;
                        Gl.Vertex3f(px, py, 0);
                        Gl.Vertex3f(px + scale, py, 0);
                        Gl.Vertex3f(px + scale, py + scale, 0);
                        Gl.Vertex3f(px, py + scale, 0);
                    }
                }
            }
            cx += (int)(6 * scale);
        }
        Gl.End();
    }

    private void DrawPanel(int x, int y, int w, int h, float r, float g, float b, float a)
    {
        Gl.Color4f(r, g, b, a);
        Gl.Begin(Gl.QUADS);
        Gl.Vertex3f(x, y, 0);
        Gl.Vertex3f(x + w, y, 0);
        Gl.Vertex3f(x + w, y + h, 0);
        Gl.Vertex3f(x, y + h, 0);
        Gl.End();
    }

    public void Draw(int w, int h, Simulation sim, Renderer renderer, bool editorEnabled, int selectedPoint,
        bool aiMenu, int aiMenuRow, int hiddenLayers, int hiddenNodes, AiModel_V1.TrainingConfig cfg,
        bool physMenu, int physRow, bool rewMenu, int rewRow,
        bool showBrain, bool brainFullscreen, bool showStatus, bool orbitPaused,
        string deviceLabel, double vramMb, double stepsPerSec)
    {
        Begin2D(w, h);

        // ---- stats (top-left) ----
        if (showStatus && !brainFullscreen)
        {
            DrawPanel(10, 10, 300, 324, 0.05f, 0.08f, 0.12f, 0.75f);
            string cam = renderer.Mode == CameraMode.Orbit ? "ORBIT" : renderer.Mode == CameraMode.Chase ? "CHASE" : "FREE";
            string mode = sim.OffTrack ? "OFF TRACK" : "DRIVING";
            DrawText(20, 18, "AI CAR - NEURAL DRIVER", 2, 0.4f, 0.9f, 1f);
            DrawText(20, 40, "CAMERA: " + cam + (renderer.Mode == CameraMode.Orbit && orbitPaused ? " (PAUSED)" : ""), 1, 0.7f, 0.8f, 1f);
            DrawText(20, 56, "STATUS: " + mode, 1, sim.OffTrack ? 1f : 0.4f, sim.OffTrack ? 0.3f : 0.9f, 0.4f);
            DrawText(20, 72, "STEPS: " + sim.Steps.ToString("#,##0", CultureInfo.InvariantCulture), 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 88, "LAPS: " + sim.Laps + "  CURRENT: " + (sim.CurrentLapProgress * 100f).ToString("F0") + "%", 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 104, "SPEED: " + sim.Car.Speed.ToString("F1") + " / " + Car.MaxSpeed, 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 120, "REWARD: " + sim.Reward.ToString("F2"), 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 136, "TOTAL: " + sim.TotalReward.ToString("#,##0", CultureInfo.InvariantCulture), 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 152, "GENERATION: " + sim.Generation, 1, 0.8f, 0.8f, 0.8f);
            DrawText(20, 168, "NEW GEN OFF TRACK: " + (sim.NewGenerationOnOffTrack ? "ON" : "OFF"), 1,
                sim.NewGenerationOnOffTrack ? 0.3f : 0.8f, sim.NewGenerationOnOffTrack ? 1f : 0.8f, 0.5f);
            float avg = sim.Steps > 0 ? sim.TotalReward / sim.Steps : 0f;
            DrawText(20, 184, "AVG/STEP: " + avg.ToString("F3"), 1, 0.8f, 0.8f, 0.8f);
            if (sim.NewGenerationOnOffTrack)
            {
                // K mode: generational distance records (%, multi-lap capable).
                DrawText(20, 200, "BEST DIST: " + (sim.BestGenDist * 100f).ToString("F1") + "%", 1, 0.4f, 1f, 0.5f);
                DrawText(20, 216, "WORST DIST: " + (sim.WorstGenDist * 100f).ToString("F1") + "%", 1, 1f, 0.45f, 0.45f);
            }
            else
            {
                    DrawText(20, 200, "ON TRACK: " + sim.OnTrackPct.ToString("F1") + "% (5K)", 1, 0.4f, 0.9f, 0.4f);
                DrawText(20, 216, "WORST ON: " + sim.WorstOnTrackPct.ToString("F1") + "%", 1, 1f, 0.45f, 0.45f);
            }
            DrawText(20, 232, "BEST STEP: " + sim.BestReward.ToString("F2"), 1, 0.4f, 1f, 0.5f);
            DrawText(20, 248, "WORST STEP: " + sim.WorstReward.ToString("F2"), 1, 1f, 0.45f, 0.45f);
            DrawText(20, 264, "CHAMPS: " + sim.ChampCount, 1, 1f, 0.85f, 0.3f);
            DrawText(20, 280, "CHAMP GEN: " + sim.ChampGen + " / " + sim.Generation, 1, 1f, 0.85f, 0.3f);
            bool onGpu = deviceLabel.StartsWith("GPU");
            string dev = deviceLabel.Length > 32 ? deviceLabel.Substring(0, 32) : deviceLabel;
            DrawText(20, 296, "DEVICE: " + dev, 1, onGpu ? 0.4f : 0.8f, onGpu ? 1f : 0.8f, 0.5f);
            string sps = "STEPS/S: " + ((long)Math.Round(stepsPerSec)).ToString("#,##0", CultureInfo.InvariantCulture);
            if (onGpu) sps += "  VRAM " + vramMb.ToString("F1") + "MB";
            DrawText(20, 312, sps, 1, 0.8f, 0.8f, 0.8f);
        }

        // ---- controls (bottom-left) ----
        if (!brainFullscreen)
        {
        DrawPanel(10, h - 156, 410, 146, 0.05f, 0.08f, 0.12f, 0.75f);
        DrawText(20, h - 142, "CONTROLS", 1, 0.4f, 0.9f, 1f);
        DrawText(20, h - 126, "1/2/3 CAMERA  P PAUSE ORBIT", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 112, "DRAG MOUSE ROTATE  WHEEL ZOOM", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 98, "N/F/S  NORMAL/STEP/SUPERFAST", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 84, "K NEW GENERATION OFF TRACK", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 70, "SPACE  STEP (IN STEP MODE)", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 56, "R RETRAIN  G RANDOM TRACK  T TRACK  M/C AI MENU", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 42, "E PHYSICS  W REWARDS  D CPU/GPU  B BRAIN  V FULL", 1, 0.7f, 0.7f, 0.7f);
        DrawText(20, h - 28, "H STATUS  ESC QUIT", 1, 0.7f, 0.7f, 0.7f);
        // Clickable buttons that open the menus (see *ButtonHit).
        DrawPanel(10, h - 186, 150, 24, 0.1f, 0.3f, 0.45f, 0.9f);
        DrawText(24, h - 180, "[AI SETUP] (M)", 1, 1f, 1f, 1f);
        DrawPanel(166, h - 186, 150, 24, 0.1f, 0.35f, 0.3f, 0.9f);
        DrawText(180, h - 180, "[PHYSICS] (E)", 1, 1f, 1f, 1f);
        DrawPanel(322, h - 186, 150, 24, 0.35f, 0.3f, 0.1f, 0.9f);
        DrawText(336, h - 180, "[REWARDS] (W)", 1, 1f, 1f, 1f);
        }

        // ---- AI brain panel (top-right) ----
        if (showBrain) DrawBrain(w, h, sim, brainFullscreen);

        // ---- track editor markers ----
        if (editorEnabled && !brainFullscreen)
        {
            DrawPanel(w - 330, h - 100, 315, 85, 0.05f, 0.08f, 0.12f, 0.85f);
            DrawText(w - 320, h - 92, "TRACK EDITOR", 1, 1f, 0.8f, 0.3f);
            DrawText(w - 320, h - 76, "DRAG POINT  A ADD SECTION", 1, 0.8f, 0.8f, 0.8f);
            DrawText(w - 320, h - 62, "DEL REMOVE  [/] ROAD WIDTH", 1, 0.8f, 0.8f, 0.8f);
            DrawText(w - 320, h - 48, "SECTION: " + (selectedPoint + 1) + " / " + sim.Track.ControlPoints.Count, 1, 0.8f, 0.8f, 0.8f);
            DrawText(w - 320, h - 34, "WIDTH: " + (sim.Track.HalfWidth * 2f).ToString("F1"), 1, 0.8f, 0.8f, 0.8f);
        }

        if (aiMenu)
            DrawAiMenu(w, h, aiMenuRow, hiddenLayers, hiddenNodes, cfg);
        if (physMenu)
            DrawPhysicsMenu(w, h, physRow, sim.PhysicsCfg);
        if (rewMenu)
            DrawRewardsMenu(w, h, rewRow, sim.RewardCfg);

        End2D();
    }

    /// <summary>Hit-test for the clickable [AI SETUP] button (screen coords, origin top-left).</summary>
    public static bool AiButtonHit(int w, int h, int mx, int my)
        => mx >= 10 && mx <= 160 && my >= h - 186 && my <= h - 162;

    public static bool PhysButtonHit(int w, int h, int mx, int my)
        => mx >= 166 && mx <= 316 && my >= h - 186 && my <= h - 162;

    public static bool RewButtonHit(int w, int h, int mx, int my)
        => mx >= 322 && mx <= 472 && my >= h - 186 && my <= h - 162;

    private void DrawMenuFrame(int w, int h, string title, int rows, out int x, out int y)
    {
        const int mw = 480;
        int mh = 110 + rows * 30;
        x = (w - mw) / 2; y = (h - mh) / 2;
        DrawPanel(x, y, mw, mh, 0.03f, 0.05f, 0.09f, 0.96f);
        DrawText(x + 24, y + 18, title, 2, 0.35f, 0.9f, 1f);
    }

    private void DrawMenuRows(int x, int y, int selectedRow, string[] rows)
    {
        for (int i = 0; i < rows.Length; i++)
        {
            bool sel = selectedRow == i;
            DrawText(x + 35, y + 62 + i * 30, (sel ? "> " : "  ") + rows[i], 2,
                sel ? 1f : 0.75f, sel ? 0.8f : 0.75f, 0.35f);
        }
        DrawText(x + 24, y + 62 + rows.Length * 30 + 8, "UP/DOWN SELECT  LEFT/RIGHT CHANGE", 1, 0.7f, 0.8f, 0.9f);
        DrawText(x + 24, y + 62 + rows.Length * 30 + 26, "ENTER DONE  ESC CANCEL", 1, 0.7f, 0.8f, 0.9f);
    }

    private void DrawPhysicsMenu(int w, int h, int selectedRow, PhysicsConfig p)
    {
        DrawMenuFrame(w, h, "PHYSICS TUNING", 7, out int x, out int y);
        DrawMenuRows(x, y, selectedRow, new[]
        {
            "REALISTIC PHYSICS: " + OnOff(p.Realistic),
            "GRIP: " + p.Grip.ToString("F1"),
            "DOWNFORCE: " + p.Downforce.ToString("F2"),
            "STEERING: " + p.Steering.ToString("F2"),
            "OVERSTEER: " + p.Oversteer.ToString("F2"),
            "SPIN THRESHOLD: " + p.SpinThreshold.ToString("F1"),
            "UNDERSTEER: " + OnOff(p.Understeer),
        });
    }

    private void DrawRewardsMenu(int w, int h, int selectedRow, RewardConfig r)
    {
        DrawMenuFrame(w, h, "REWARD SHAPING", 11, out int x, out int y);
        DrawMenuRows(x, y, selectedRow, new[]
        {
            "SPEED: " + r.Speed.ToString("F2"),
            "CENTERING: " + r.Centering.ToString("F2"),
            "ALIGNMENT: " + r.Alignment.ToString("F2"),
            "PROGRESS: " + r.Progress.ToString("F0"),
            "WRONG WAY: " + r.WrongWay.ToString("F2"),
            "SLIDE: " + r.Slide.ToString("F2"),
            "SPIN: " + r.Spin.ToString("F2"),
            "UNDERSTEER: " + r.Understeer.ToString("F2"),
            "STEER EFFORT: " + r.SteerEffort.ToString("F3"),
            "OFF TRACK: " + r.OffTrack.ToString("F1"),
            "PARKING: " + r.Parking.ToString("F2"),
        });
    }

    private static string OnOff(bool b) => b ? "ON" : "OFF";
    private static string ActName(int a) => a == 1 ? "RELU" : a == 2 ? "GELU" : "TANH";
    private static string InitName(int i) => i == 1 ? "HE" : i == 2 ? "ORTHO" : "XAVIER";

    private void DrawAiMenu(int w, int h, int selectedRow, int hiddenLayers, int hiddenNodes, AiModel_V1.TrainingConfig cfg)
    {
        const int mw = 460, mh = 400;
        int x = (w - mw) / 2, y = (h - mh) / 2;
        DrawPanel(x, y, mw, mh, 0.03f, 0.05f, 0.09f, 0.96f);
        DrawText(x + 24, y + 18, "AI NETWORK CONFIGURATION", 2, 0.35f, 0.9f, 1f);

        string[] rows =
        {
            "HIDDEN LAYERS: " + hiddenLayers,
            "NODES PER LAYER: " + hiddenNodes,
            "RESIDUAL: " + OnOff(cfg.UseResidual),
            "LAYER NORM: " + OnOff(cfg.UseLayerNorm),
            "ACTIVATION: " + ActName(cfg.Activation),
            "ADAM: " + OnOff(cfg.UseAdam),
            "INIT: " + InitName(cfg.Init),
            "GRAD CLIP: " + (cfg.GradClip <= 0f ? "OFF" : cfg.GradClip.ToString("F1")),
            "PPO: " + OnOff(cfg.UsePpo),
        };
        for (int i = 0; i < rows.Length; i++)
        {
            bool sel = selectedRow == i;
            DrawText(x + 35, y + 62 + i * 30, (sel ? "> " : "  ") + rows[i], 2,
                sel ? 1f : 0.75f, sel ? 0.8f : 0.75f, 0.35f);
        }
        DrawText(x + 24, y + mh - 48, "UP/DOWN SELECT  LEFT/RIGHT CHANGE", 1, 0.7f, 0.8f, 0.9f);
        DrawText(x + 24, y + mh - 30, "ENTER APPLY AND RETRAIN  ESC CANCEL", 1, 0.7f, 0.8f, 0.9f);
    }

    private void DrawBrain(int w, int h, Simulation sim, bool fullscreen)
    {
        int pw = fullscreen ? w - 30 : Math.Clamp((int)(w * 0.30f), 250, 440);
        int ph = fullscreen ? h - 30 : Math.Clamp((int)(h * 0.38f), 210, 360);
        int px = fullscreen ? 15 : w - pw - 15;
        int py = 15;
        DrawPanel(px, py, pw, ph, 0.05f, 0.08f, 0.12f, 0.8f);
        DrawText(px + 10, py + 8, fullscreen ? "AI BRAIN (FULL SCREEN)" : "AI BRAIN (LIVE)",
            fullscreen ? 2 : 1, 0.4f, 0.9f, 1f);

        var net = sim.Brain.Net;
        int[] sizes = net.Sizes;
        int L = sizes.Length;
        float graphLeft = px + (fullscreen ? 65f : 40f);
        float graphRight = px + pw - (fullscreen ? 65f : 40f);
        float graphTop = py + (fullscreen ? 62f : 40f);
        float graphBottom = py + ph - (fullscreen ? 72f : 58f);

        // node positions per layer
        var layerX = new float[L];
        var layerYs = new List<float>[L];
        for (int l = 0; l < L; l++)
        {
            layerX[l] = graphLeft + (float)l * ((graphRight - graphLeft) / Math.Max(1, L - 1));
            layerYs[l] = new List<float>();
            int count = sizes[l];
            float top = graphTop, bottom = graphBottom;
            for (int i = 0; i < count; i++)
            {
                float y = count == 1 ? (top + bottom) / 2 : top + (bottom - top) * (i / (float)(count - 1));
                layerYs[l].Add(y);
            }
        }

        // weight lines (2px, brighter base, boosted along active pathways).
        // Decimated for huge nets: drawing all 500k+ lines of a 128x64 brain
        // every frame would choke the renderer regardless of compute device.
        long connectionCount = 0;
        for (int l = 0; l < L - 1; l++) connectionCount += (long)sizes[l] * sizes[l + 1];
        float lineScale = Math.Clamp(1800f / Math.Max(1800f, connectionCount), 0.02f, 1f);
        long stride = Math.Max(1, connectionCount / 4000);
        long drawn = 0;
        bool hasActs = sim.NodeActivations.Length > 0;
        Gl.LineWidth(2f);
        Gl.Begin(Gl.LINES);
        for (int l = 0; l < L - 1; l++)
        {
            for (int j = 0; j < sizes[l + 1]; j++)
            {
                float actTo = hasActs ? Math.Abs(GetActivation(sim, l + 1, j)) : 0f;
                for (int i = 0; i < sizes[l]; i++)
                {
                    if ((drawn++ % stride) != 0) continue;
                    float wgt = net.Weight(l, j, i);
                    float mag = Math.Clamp(Math.Abs(wgt) / 2f, 0f, 1f);
                    float actFrom = hasActs ? Math.Abs(GetActivation(sim, l, i)) : 0f;
                    float energy = Math.Max(actFrom, actTo); // 0..~1 along live paths
                    float alpha = (0.07f + mag * 0.55f) * (0.45f + 0.55f * energy) * lineScale;
                    float boost = 0.55f + 0.45f * energy;
                    if (wgt >= 0) Gl.Color4f(0.2f * boost, 0.9f * boost, 1f, alpha);
                    else Gl.Color4f(1f, 0.3f * boost, 0.4f * boost, alpha);
                    Gl.Vertex3f(layerX[l], layerYs[l][i], 0);
                    Gl.Vertex3f(layerX[l + 1], layerYs[l + 1][j], 0);
                }
            }
        }
        Gl.End();
        Gl.LineWidth(1f);

        // nodes
        for (int l = 0; l < L; l++)
        {
            for (int i = 0; i < sizes[l]; i++)
            {
                float act = sim.NodeActivations.Length > 0 ? GetActivation(sim, l, i) : 0f;
                float norm = (act + 1f) * 0.5f; // -1..1 -> 0..1
                float r, g, b;
                if (l == 0) { r = 0.3f; g = 0.6f + norm * 0.4f; b = 1f; }
                else if (l == L - 1) { r = 1f; g = 0.6f; b = 0.2f; }
                else { r = 0.2f + norm * 0.8f; g = 0.9f; b = 0.3f; }
                float verticalSpacing = sizes[l] <= 1 ? graphBottom - graphTop : (graphBottom - graphTop) / (sizes[l] - 1);
                float horizontalSpacing = L <= 1 ? graphRight - graphLeft : (graphRight - graphLeft) / (L - 1);
                float maxRadius = Math.Max(0.75f, Math.Min(verticalSpacing, horizontalSpacing) * 0.32f);
                float radius = Math.Min(fullscreen ? 10f : 7f, maxRadius) * (0.65f + norm * 0.35f);
                DrawCircle(layerX[l], layerYs[l][i], radius, r, g, b);
            }
        }

        // labels
        DrawText((int)graphLeft - 8, (int)graphBottom + 12, "IN", 1, 0.6f, 0.8f, 1f);
        DrawText((int)graphRight - 12, (int)graphBottom + 12, "OUT", 1, 1f, 0.7f, 0.4f);
        // steer / throttle readout
        DrawText(px + 10, py + ph - 22, "GAS: " + sim.Throttle.ToString("F2") +
            "  BRK: " + sim.Brake.ToString("F2") +
            "  STEER: " + sim.Steer.ToString("F2") + "  LAYERS: " + (L - 2) +
            "  NODES: " + (L > 2 ? sizes[1] : 0), 1, 0.8f, 0.8f, 0.8f);
        if (fullscreen) DrawText(px + pw - 155, py + 12, "V CLOSE", 1, 0.7f, 0.8f, 0.9f);
    }

    private float GetActivation(Simulation sim, int layer, int node)
    {
        int idx = 0;
        for (int l = 0; l < layer; l++) idx += sim.Brain.Net.Sizes[l];
        idx += node;
        return idx < sim.NodeActivations.Length ? sim.NodeActivations[idx] : 0f;
    }

    private void DrawCircle(float cx, float cy, float r, float cr, float cg, float cb)
    {
        Gl.Color3f(cr, cg, cb);
        Gl.Begin(Gl.TRIANGLE_FAN);
        Gl.Vertex3f(cx, cy, 0);
        for (int i = 0; i <= 16; i++)
        {
            double a = i * 2.0 * Math.PI / 16;
            Gl.Vertex3f(cx + (float)Math.Cos(a) * r, cy + (float)Math.Sin(a) * r, 0);
        }
        Gl.End();
    }

}
