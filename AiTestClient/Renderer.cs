using System;
using AiTestClient.Native;
using AiTestClient.World;

namespace AiTestClient;

public enum CameraMode { Orbit, Chase, Free }

/// <summary>
/// Renders the 3D scene: a lit ground, the spline track (road + curbs + dashed
/// centerline), the car, and the car's vision rays. Supports three camera
/// modes (Orbit, Chase, Free) and exposes an unproject helper used by the
/// track editor.
/// </summary>
public class Renderer
{
    public CameraMode Mode { get; set; } = CameraMode.Orbit;

    // orbit / free camera state
    private float _orbitAngle = 0.6f;
    private float _orbitDist = 60f;
    private float _orbitPitch = 0.64f;
    private float _freeYaw = 0.6f;
    private float _freePitch = 0.9f;
    private float _freeDist = 40f;

    private float[] _proj = new float[16];
    private float[] _view = new float[16];
    private float[] _vp = new float[16];

    /// <summary>The current view (camera) matrix, applied to the modelview stack each frame.</summary>
    public float[] ViewMatrix => _view;

    private int _lastMouseX = -1, _lastMouseY = -1;

    public void Setup(int w, int h)
    {
        Gl.Enable(Gl.DEPTH_TEST);
        Gl.DepthFunc(Gl.LEQUAL);
        // The scene is assembled from two-sided strips/quads whose winding is
        // not consistent (the ground, for example, faces down in XZ order).
        // Culling those primitives removes the road, ground, and much of the
        // car from the overhead camera.
        Gl.Disable(Gl.CULL_FACE);
        Gl.ShadeModel(Gl.SMOOTH);
        Gl.Hint(Gl.LINE_SMOOTH_HINT, Gl.NICEST);
        Gl.Enable(Gl.LINE_SMOOTH);
        Gl.Enable(Gl.BLEND);
        Gl.BlendFunc(Gl.SRC_ALPHA, Gl.ONE_MINUS_SRC_ALPHA);

        // lighting
        Gl.Enable(Gl.LIGHTING);
        Gl.Enable(Gl.LIGHT0);
        Gl.Enable(Gl.COLOR_MATERIAL);
        float[] lightPos = { 30f, 60f, 20f, 0f };
        Gl.Lightfv(Gl.LIGHT0, Gl.POSITION, lightPos);
        float[] ambient = { 0.35f, 0.35f, 0.4f, 1f };
        Gl.Lightfv(Gl.LIGHT0, Gl.AMBIENT, ambient);
        float[] diffuse = { 0.85f, 0.85f, 0.9f, 1f };
        Gl.Lightfv(Gl.LIGHT0, Gl.DIFFUSE, diffuse);
        float[] specular = { 0.6f, 0.6f, 0.6f, 1f };
        Gl.Lightfv(Gl.LIGHT0, Gl.SPECULAR, specular);
        float[] globalAmbient = { 0.25f, 0.25f, 0.3f, 1f };
        Gl.LightModelfv(Gl.LIGHT_MODEL_AMBIENT, globalAmbient);
    }

    public void UpdateCamera(Simulation sim, float dt, float mouseX, float mouseY,
        bool mouseDown, int mouseWheelY, bool orbitPaused)
    {
        if (Mode == CameraMode.Orbit && !orbitPaused && !mouseDown)
        {
            _orbitAngle += dt * 0.15f;
        }
        if (mouseDown && _lastMouseX >= 0)
        {
            float dx = mouseX - _lastMouseX;
            float dy = mouseY - _lastMouseY;
            if (Mode == CameraMode.Orbit)
            {
                _orbitAngle -= dx * 0.006f;
                _orbitPitch = Math.Clamp(_orbitPitch + dy * 0.006f, 0.12f, 1.45f);
            }
            else if (Mode == CameraMode.Free)
            {
                _freeYaw -= dx * 0.006f;
                _freePitch = Math.Clamp(_freePitch + dy * 0.006f, 0.12f, 1.45f);
            }
        }
        if (mouseWheelY != 0)
        {
            float zoom = (float)Math.Pow(0.88, mouseWheelY);
            if (Mode == CameraMode.Free) _freeDist = Math.Clamp(_freeDist * zoom, 6f, 220f);
            else _orbitDist = Math.Clamp(_orbitDist * zoom, 12f, 300f);
        }
        _lastMouseX = (int)mouseX;
        _lastMouseY = (int)mouseY;

        var (ex, ey, ez) = ComputeEye(sim);
        var (tx, ty, tz) = ComputeTarget(sim);
        BuildView(ex, ey, ez, tx, ty, tz);
    }

    private (float, float, float) ComputeEye(Simulation sim)
    {
        switch (Mode)
        {
            case CameraMode.Chase:
            {
                var (fx, fz) = sim.Car.ForwardDir();
                float ex = sim.Car.X - fx * 14f;
                float ez = sim.Car.Z - fz * 14f;
                return (ex, 8f, ez);
            }
            case CameraMode.Free:
            {
                float ex = sim.Car.X + (float)Math.Cos(_freeYaw) * (float)Math.Cos(_freePitch) * _freeDist;
                float ez = sim.Car.Z + (float)Math.Sin(_freeYaw) * (float)Math.Cos(_freePitch) * _freeDist;
                float ey = (float)Math.Sin(_freePitch) * _freeDist;
                return (ex, ey, ez);
            }
            default: // Orbit
            {
                float horizontal = (float)Math.Cos(_orbitPitch) * _orbitDist;
                float ex = (float)Math.Cos(_orbitAngle) * horizontal;
                float ez = (float)Math.Sin(_orbitAngle) * horizontal;
                return (ex, (float)Math.Sin(_orbitPitch) * _orbitDist, ez);
            }
        }
    }

    private (float, float, float) ComputeTarget(Simulation sim)
    {
        switch (Mode)
        {
            case CameraMode.Chase:
            case CameraMode.Free:
                return (sim.Car.X, 1f, sim.Car.Z);
            default:
                return (0f, 0f, 0f);
        }
    }

    private void BuildView(float ex, float ey, float ez, float tx, float ty, float tz)
    {
        float fx = tx - ex, fy = ty - ey, fz = tz - ez;
        float fl = (float)Math.Sqrt(fx * fx + fy * fy + fz * fz);
        fx /= fl; fy /= fl; fz /= fl;
        // right = fwd x up(0,1,0)
        float rx = -fz, ry = 0f, rz = fx;
        float rl = (float)Math.Sqrt(rx * rx + rz * rz);
        rx /= rl; rz /= rl;
        // upv = right x fwd
        float ux = ry * fz - rz * fy;
        float uy = rz * fx - rx * fz;
        float uz = rx * fy - ry * fx;

        var m = new float[16];
        // glLoadMatrixf consumes column-major storage. Each camera basis
        // vector is a row of the view transform, so its components are
        // distributed down the corresponding entries below.
        m[0] = rx; m[1] = ux; m[2] = -fx;
        m[4] = ry; m[5] = uy; m[6] = -fy;
        m[8] = rz; m[9] = uz; m[10] = -fz;
        m[3] = 0;  m[7] = 0;  m[11] = 0;
        m[12] = -(rx * ex + ry * ey + rz * ez);
        m[13] = -(ux * ex + uy * ey + uz * ez);
        m[14] = (fx * ex + fy * ey + fz * ez);
        m[15] = 1;
        _view = m;
    }

    public void BeginFrame(int w, int h)
    {
        Gl.Viewport(0, 0, w, h);
        Gl.ClearColor(0.05f, 0.07f, 0.12f, 1f);
        Gl.Clear(Gl.COLOR_BUFFER_BIT | Gl.DEPTH_BUFFER_BIT);

        float aspect = (float)w / (float)h;
        float fov = 50f * (float)Math.PI / 180f;
        float f = 1f / (float)Math.Tan(fov * 0.5f);
        float near = 0.1f, far = 500f;
        _proj = new float[16]
        {
            f / aspect, 0, 0, 0,
            0, f, 0, 0,
            0, 0, (far + near) / (near - far), -1,
            0, 0, (2f * far * near) / (near - far), 0
        };
        _vp = Mat4.Multiply(_proj, _view);

        // OpenGL's fixed-function pipeline keeps projection and model-view in
        // separate stacks. Merely calculating _proj is not enough: it must be
        // loaded before drawing or the default identity projection clips the
        // entire world to the tiny -1..1 cube around the origin.
        Gl.MatrixMode(Gl.PROJECTION);
        Gl.LoadMatrixf(_proj);
        Gl.MatrixMode(Gl.MODELVIEW);
        Gl.LoadMatrixf(_view);
    }

    public void Render(Simulation sim)
    {
        DrawSkybox();
        DrawGround();
        DrawTrack(sim.Track);
        DrawStartArrow(sim.Track);
        DrawRays(sim);
        DrawCar(sim.Car);
    }

    private void DrawSkybox()
    {
        Gl.Disable(Gl.LIGHTING);
        Gl.Disable(Gl.DEPTH_TEST);
        const float s = 240f, top = 180f;
        Gl.Begin(Gl.QUADS);
        // Four horizon walls use per-vertex color for a simple sky gradient.
        DrawSkyWall(-s, -s, s, -s, top);
        DrawSkyWall(s, -s, s, s, top);
        DrawSkyWall(s, s, -s, s, top);
        DrawSkyWall(-s, s, -s, -s, top);
        Gl.Color3f(0.12f, 0.30f, 0.62f);
        Gl.Vertex3f(-s, top, -s); Gl.Vertex3f(-s, top, s);
        Gl.Vertex3f(s, top, s); Gl.Vertex3f(s, top, -s);
        Gl.End();
        Gl.Enable(Gl.DEPTH_TEST);
    }

    private static void DrawSkyWall(float ax, float az, float bx, float bz, float top)
    {
        Gl.Color3f(0.58f, 0.76f, 0.92f);
        Gl.Vertex3f(ax, 0, az); Gl.Vertex3f(bx, 0, bz);
        Gl.Color3f(0.12f, 0.30f, 0.62f);
        Gl.Vertex3f(bx, top, bz); Gl.Vertex3f(ax, top, az);
    }

    private void DrawStartArrow(Track track)
    {
        var (x, z) = track.CenterAt(0f);
        var (tx, tz) = track.TangentAt(0f);
        float rx = -tz, rz = tx;
        float tail = 3.5f, tip = 5.5f, shaftWidth = 0.65f, headWidth = 2.2f;
        Gl.Disable(Gl.LIGHTING);
        Gl.Disable(Gl.DEPTH_TEST);
        Gl.Color4f(0.1f, 1f, 0.55f, 0.95f);
        Gl.Begin(Gl.QUADS);
        Gl.Vertex3f(x - tx * tail + rx * shaftWidth, 0.18f, z - tz * tail + rz * shaftWidth);
        Gl.Vertex3f(x + tx * 1.5f + rx * shaftWidth, 0.18f, z + tz * 1.5f + rz * shaftWidth);
        Gl.Vertex3f(x + tx * 1.5f - rx * shaftWidth, 0.18f, z + tz * 1.5f - rz * shaftWidth);
        Gl.Vertex3f(x - tx * tail - rx * shaftWidth, 0.18f, z - tz * tail - rz * shaftWidth);
        Gl.End();
        Gl.Color4f(0.2f, 0.85f, 1f, 1f);
        Gl.Begin(Gl.TRIANGLES);
        Gl.Vertex3f(x + tx * tip, 0.2f, z + tz * tip);
        Gl.Vertex3f(x + tx * 1.2f + rx * headWidth, 0.2f, z + tz * 1.2f + rz * headWidth);
        Gl.Vertex3f(x + tx * 1.2f - rx * headWidth, 0.2f, z + tz * 1.2f - rz * headWidth);
        Gl.End();
        Gl.Enable(Gl.DEPTH_TEST);
    }

    public void RenderEditorOverlay(Track track, int selectedPoint)
    {
        Gl.Disable(Gl.LIGHTING);
        Gl.Disable(Gl.DEPTH_TEST);
        Gl.LineWidth(2f);
        Gl.Begin(Gl.LINES);
        for (int i = 0; i < track.ControlPoints.Count; i++)
        {
            var a = track.ControlPoints[i];
            var b = track.ControlPoints[(i + 1) % track.ControlPoints.Count];
            Gl.Color4f(i == selectedPoint ? 1f : 0.3f, 0.75f, 0.15f, 0.9f);
            Gl.Vertex3f(a.x, 0.35f, a.z);
            Gl.Vertex3f(b.x, 0.35f, b.z);
        }
        Gl.End();

        Gl.PointSize(12f);
        Gl.Begin(Gl.POINTS);
        for (int i = 0; i < track.ControlPoints.Count; i++)
        {
            var p = track.ControlPoints[i];
            if (i == selectedPoint) Gl.Color3f(1f, 0.25f, 0.1f);
            else Gl.Color3f(1f, 0.85f, 0.25f);
            Gl.Vertex3f(p.x, 0.5f, p.z);
        }
        Gl.End();
        Gl.Enable(Gl.DEPTH_TEST);
    }

    private void DrawGround()
    {
        Gl.Disable(Gl.LIGHTING);
        float s = 200f;
        Gl.Begin(Gl.QUADS);
        const float tile = 10f;
        for (float x = -s; x < s; x += tile)
        for (float z = -s; z < s; z += tile)
        {
            bool light = (((int)((x + s) / tile) + (int)((z + s) / tile)) & 1) == 0;
            Gl.Color3f(light ? 0.10f : 0.075f, light ? 0.25f : 0.20f, light ? 0.08f : 0.065f);
            Gl.Vertex3f(x, 0f, z); Gl.Vertex3f(x + tile, 0f, z);
            Gl.Vertex3f(x + tile, 0f, z + tile); Gl.Vertex3f(x, 0f, z + tile);
        }
        Gl.End();

        // Fine mowing/grid lines give the grass texture some depth.
        Gl.Begin(Gl.LINES);
        Gl.Color4f(0.16f, 0.34f, 0.13f, 0.45f);
        for (float i = -s; i <= s; i += 10f)
        {
            Gl.Vertex3f(i, 0.01f, -s); Gl.Vertex3f(i, 0.01f, s);
            Gl.Vertex3f(-s, 0.01f, i); Gl.Vertex3f(s, 0.01f, i);
        }
        Gl.End();
    }

    private void DrawTrack(Track track)
    {
        int n = track.SampleCount;
        float hw = track.HalfWidth;

        // road surface (triangle strip between the two edges)
        Gl.Disable(Gl.LIGHTING);
        Gl.Begin(Gl.TRIANGLE_STRIP);
        for (int i = 0; i <= n; i++)
        {
            int sample = i % n;
            float nx = -track.TangentZ(sample), nz = track.TangentX(sample);
            float shade = 0.20f + 0.025f * (float)Math.Sin(sample * 0.7f);
            Gl.Color3f(shade, shade, shade + 0.025f);
            Gl.Vertex3f(track.SampleX(sample) + nx * hw, 0.02f, track.SampleZ(sample) + nz * hw);
            Gl.Color3f(shade * 0.94f, shade * 0.94f, shade + 0.015f);
            Gl.Vertex3f(track.SampleX(sample) - nx * hw, 0.02f, track.SampleZ(sample) - nz * hw);
        }
        Gl.End();

        // curbs (red/white) along both edges
        Gl.Begin(Gl.LINES);
        for (int i = 0; i < n; i += 2)
        {
            int a = i, b = (i + 1) % n;
            float nax = -track.TangentZ(a), naz = track.TangentX(a);
            float nbx = -track.TangentZ(b), nbz = track.TangentX(b);
            bool red = (i / 2) % 2 == 0;
            Gl.Color3f(red ? 0.85f : 0.9f, red ? 0.15f : 0.9f, red ? 0.15f : 0.9f);
            Gl.Vertex3f(track.SampleX(a) + nax * hw, 0.05f, track.SampleZ(a) + naz * hw);
            Gl.Vertex3f(track.SampleX(b) + nbx * hw, 0.05f, track.SampleZ(b) + nbz * hw);
            Gl.Vertex3f(track.SampleX(a) - nax * hw, 0.05f, track.SampleZ(a) - naz * hw);
            Gl.Vertex3f(track.SampleX(b) - nbx * hw, 0.05f, track.SampleZ(b) - nbz * hw);
        }
        Gl.End();

        // dashed centerline
        Gl.Begin(Gl.LINES);
        Gl.Color3f(0.9f, 0.85f, 0.3f);
        for (int i = 0; i < n; i += 4)
        {
            int a = i, b = (i + 2) % n;
            Gl.Vertex3f(track.SampleX(a), 0.04f, track.SampleZ(a));
            Gl.Vertex3f(track.SampleX(b), 0.04f, track.SampleZ(b));
        }
        Gl.End();
    }

    private void DrawRays(Simulation sim)
    {
        var car = sim.Car;
        var rays = sim.Brain.Rays;
        Gl.Disable(Gl.LIGHTING);
        Gl.PushAttrib(Gl.ALL_ATTRIB_BITS);
        Gl.Enable(Gl.BLEND);
        Gl.BlendFunc(Gl.SRC_ALPHA, Gl.ONE); // additive glow
        Gl.LineWidth(2.5f);
        Gl.Begin(Gl.LINES);
        for (int i = 0; i < rays.Count; i++)
        {
            double a = car.Heading + Math.PI * rays.AngleDeg(i) / 180.0;
            float dx = (float)Math.Cos(a), dz = (float)Math.Sin(a);
            float dist = sim.RayDistances[i];
            float ex = car.X + dx * dist, ez = car.Z + dz * dist;
            // color: green (far) -> red (near)
            float t = Math.Clamp(1f - dist / rays.MaxDist, 0f, 1f);
            Gl.Color4f(0.2f + t * 0.8f, 0.9f - t * 0.7f, 0.3f, 0.9f);
            Gl.Vertex3f(car.X, 1.2f, car.Z);
            Gl.Vertex3f(ex, 1.2f, ez);
            // endpoint dot
            Gl.Vertex3f(ex, 1.2f, ez);
            Gl.Vertex3f(ex + dx * 0.6f, 1.2f, ez + dz * 0.6f);
        }
        Gl.End();
        Gl.PopAttrib();
    }

    private void DrawCar(Car car)
    {
        Gl.Enable(Gl.LIGHTING);
        Gl.Materialf(Gl.FRONT_AND_BACK, Gl.SHININESS, 30f);

        var (fx, fz) = car.ForwardDir();
        var (rx, rz) = car.RightDir();
        // Four dark wheels, a low sports-car body, glass cabin and trim.
        foreach (float forward in new[] { -1.35f, 1.35f })
        foreach (float side in new[] { -1.08f, 1.08f })
            DrawOrientedBox(car.X + fx * forward + rx * side, car.Z + fz * forward + rz * side,
                fx, fz, rx, rz, 0.55f, 0.24f, 0.14f, 0.58f, 0.035f, 0.04f, 0.05f);

        DrawOrientedBox(car.X, car.Z, fx, fz, rx, rz, 2.35f, 1.02f, 0.35f, 1.02f, 0.82f, 0.06f, 0.08f);
        DrawOrientedBox(car.X - fx * 0.25f, car.Z - fz * 0.25f, fx, fz, rx, rz,
            0.95f, 0.78f, 1.02f, 1.55f, 0.07f, 0.20f, 0.30f);
        DrawOrientedBox(car.X + fx * 0.75f, car.Z + fz * 0.75f, fx, fz, rx, rz,
            1.45f, 0.09f, 1.035f, 1.08f, 1f, 0.75f, 0.08f);

        // Rear spoiler and bright front lamps make direction obvious.
        DrawOrientedBox(car.X - fx * 1.85f, car.Z - fz * 1.85f, fx, fz, rx, rz,
            0.16f, 1.25f, 1.08f, 1.22f, 0.12f, 0.12f, 0.15f);
        foreach (float side in new[] { -0.62f, 0.62f })
            DrawOrientedBox(car.X + fx * 2.37f + rx * side, car.Z + fz * 2.37f + rz * side,
                fx, fz, rx, rz, 0.08f, 0.24f, 0.54f, 0.84f, 1f, 0.95f, 0.55f);
    }

    private static void DrawOrientedBox(float cx, float cz, float fx, float fz, float rx, float rz,
        float halfLength, float halfWidth, float y0, float y1, float red, float green, float blue)
    {
        float x0 = cx - fx * halfLength - rx * halfWidth, z0 = cz - fz * halfLength - rz * halfWidth;
        float x1 = cx + fx * halfLength - rx * halfWidth, z1 = cz + fz * halfLength - rz * halfWidth;
        float x2 = cx + fx * halfLength + rx * halfWidth, z2 = cz + fz * halfLength + rz * halfWidth;
        float x3 = cx - fx * halfLength + rx * halfWidth, z3 = cz - fz * halfLength + rz * halfWidth;
        Gl.Color3f(red, green, blue);
        Gl.Begin(Gl.QUADS);
        Gl.Normal3f(0, 1, 0);
        Gl.Vertex3f(x0, y1, z0); Gl.Vertex3f(x1, y1, z1); Gl.Vertex3f(x2, y1, z2); Gl.Vertex3f(x3, y1, z3);
        Gl.Normal3f(fx, 0, fz);
        Gl.Vertex3f(x1, y0, z1); Gl.Vertex3f(x2, y0, z2); Gl.Vertex3f(x2, y1, z2); Gl.Vertex3f(x1, y1, z1);
        Gl.Normal3f(-fx, 0, -fz);
        Gl.Vertex3f(x3, y0, z3); Gl.Vertex3f(x0, y0, z0); Gl.Vertex3f(x0, y1, z0); Gl.Vertex3f(x3, y1, z3);
        Gl.Normal3f(rx, 0, rz);
        Gl.Vertex3f(x2, y0, z2); Gl.Vertex3f(x3, y0, z3); Gl.Vertex3f(x3, y1, z3); Gl.Vertex3f(x2, y1, z2);
        Gl.Normal3f(-rx, 0, -rz);
        Gl.Vertex3f(x0, y0, z0); Gl.Vertex3f(x1, y0, z1); Gl.Vertex3f(x1, y1, z1); Gl.Vertex3f(x0, y1, z0);
        Gl.End();
    }

    // ---- track editor support ----

    /// <summary>
    /// Unproject a screen pixel to a point on the ground plane (y=0). Returns
    /// false if the ray is parallel to / pointing away from the plane.
    /// </summary>
    public bool UnprojectToGround(int w, int h, int px, int py, out float wx, out float wz)
    {
        var inv = Mat4.Invert(_vp);
        float nx = (2f * px) / w - 1f;
        float ny = 1f - (2f * py) / h;
        var pNear = Mat4.TransformPoint4(inv, nx, ny, -1f);
        var pFar = Mat4.TransformPoint4(inv, nx, ny, 1f);
        float ox = pNear.x / pNear.w, oy = pNear.y / pNear.w, oz = pNear.z / pNear.w;
        float fx = pFar.x / pFar.w - ox, fy = pFar.y / pFar.w - oy, fz = pFar.z / pFar.w - oz;
        if (Math.Abs(fy) < 1e-6f) { wx = 0; wz = 0; return false; }
        float t = -oy / fy;
        if (t < 0) { wx = 0; wz = 0; return false; }
        wx = ox + fx * t;
        wz = oz + fz * t;
        return true;
    }
}
