using System;
using System.Runtime.InteropServices;

namespace AiTestClient.Native;

/// <summary>
/// Minimal OpenGL (fixed-function / compatibility profile) P/Invoke bindings.
/// We use the legacy immediate-mode API, which is available in the default
/// compatibility context that SDL creates on Windows and Linux.
/// </summary>
public static class Gl
{
    // buffers
    public const uint COLOR_BUFFER_BIT = 0x4000;
    public const uint DEPTH_BUFFER_BIT = 0x0100;

    // primitives
    public const uint POINTS = 0;
    public const uint LINES = 1;
    public const uint LINE_LOOP = 2;
    public const uint LINE_STRIP = 3;
    public const uint TRIANGLES = 4;
    public const uint TRIANGLE_STRIP = 5;
    public const uint TRIANGLE_FAN = 6;
    public const uint QUADS = 7;
    public const uint QUAD_STRIP = 8;

    // caps
    public const uint DEPTH_TEST = 0x0B71;
    public const uint LIGHTING = 0x0B50;
    public const uint LIGHT0 = 0x4000;
    public const uint NORMALIZE = 0x1A0A;
    public const uint CULL_FACE = 0x0C40;
    public const uint BLEND = 0x0BE2;
    public const uint LINE_SMOOTH = 0x8640;
    public const uint POINT_SMOOTH = 0x8649;
    public const uint COLOR_MATERIAL = 0x0B57;

    // shading
    public const uint SMOOTH = 0x1D01;
    public const uint FLAT = 0x1D00;

    // material / light
    public const uint FRONT_AND_BACK = 0x0405;
    public const uint AMBIENT = 0x1200;
    public const uint DIFFUSE = 0x1201;
    public const uint SPECULAR = 0x1202;
    public const uint SHININESS = 0x1601;
    public const uint POSITION = 0x1203;
    public const uint LIGHT_MODEL_AMBIENT = 0x0B53;

    // depth
    public const uint LEQUAL = 0x0203;

    // blend (correct GL enum values: SRC_ALPHA=0x0302, ONE=1)
    public const uint SRC_ALPHA = 0x0302;
    public const uint ONE = 1;
    public const uint ONE_MINUS_SRC_ALPHA = 0x0303;

    // attrib
    public const uint ALL_ATTRIB_BITS = 0x000FFFFF;

    // matrices
    public const uint PROJECTION = 0x1701;
    public const uint MODELVIEW = 0x1700;

    // hints
    public const uint LINE_SMOOTH_HINT = 0x8642;
    public const uint NICEST = 0x864C;

    [DllImport("GL", EntryPoint = "glViewport")]
    public static extern void Viewport(int x, int y, int w, int h);

    [DllImport("GL", EntryPoint = "glMatrixMode")]
    public static extern void MatrixMode(uint mode);

    [DllImport("GL", EntryPoint = "glLoadIdentity")]
    public static extern void LoadIdentity();

    [DllImport("GL", EntryPoint = "glLoadMatrixf")]
    public static extern void LoadMatrixf(float[] m);

    [DllImport("GL", EntryPoint = "glClearColor")]
    public static extern void ClearColor(float r, float g, float b, float a);

    [DllImport("GL", EntryPoint = "glClear")]
    public static extern void Clear(uint mask);

    [DllImport("GL", EntryPoint = "glEnable")]
    public static extern void Enable(uint cap);

    [DllImport("GL", EntryPoint = "glDisable")]
    public static extern void Disable(uint cap);

    [DllImport("GL", EntryPoint = "glShadeModel")]
    public static extern void ShadeModel(uint mode);

    [DllImport("GL", EntryPoint = "glColor3f")]
    public static extern void Color3f(float r, float g, float b);

    [DllImport("GL", EntryPoint = "glColor4f")]
    public static extern void Color4f(float r, float g, float b, float a);

    [DllImport("GL", EntryPoint = "glMaterialf")]
    public static extern void Materialf(uint face, uint pname, float param);

    [DllImport("GL", EntryPoint = "glLightfv")]
    public static extern void Lightfv(uint light, uint pname, float[] p);

    [DllImport("GL", EntryPoint = "glLightModelfv")]
    public static extern void LightModelfv(uint pname, float[] p);

    [DllImport("GL", EntryPoint = "glNormal3f")]
    public static extern void Normal3f(float x, float y, float z);

    [DllImport("GL", EntryPoint = "glBegin")]
    public static extern void Begin(uint mode);

    [DllImport("GL", EntryPoint = "glEnd")]
    public static extern void End();

    [DllImport("GL", EntryPoint = "glVertex3f")]
    public static extern void Vertex3f(float x, float y, float z);

    [DllImport("GL", EntryPoint = "glLineWidth")]
    public static extern void LineWidth(float width);

    [DllImport("GL", EntryPoint = "glPointSize")]
    public static extern void PointSize(float size);

    [DllImport("GL", EntryPoint = "glHint")]
    public static extern void Hint(uint target, uint mode);

    [DllImport("GL", EntryPoint = "glDepthFunc")]
    public static extern void DepthFunc(uint func);

    [DllImport("GL", EntryPoint = "glBlendFunc")]
    public static extern void BlendFunc(uint sfactor, uint dfactor);

    [DllImport("GL", EntryPoint = "glPushAttrib")]
    public static extern void PushAttrib(uint mask);

    [DllImport("GL", EntryPoint = "glPopAttrib")]
    public static extern void PopAttrib();

}
