using System;
using System.Collections.Generic;

namespace AiTestClient.World;

/// <summary>
/// A closed racetrack defined by a set of control points connected by a
/// Catmull-Rom spline (the centerline). The drivable road is the band within
/// <see cref="HalfWidth"/> of the centerline.
///
/// This design is what makes the track editor possible: the client can move
/// control points and call <see cref="Rebuild"/> to re-sample the spline.
///
/// Ray casting / on-track tests are backed by a spatial hash grid so they stay
/// fast even when the simulation runs many steps per frame ("super fast").
/// </summary>
public class Track
{
    private const int SamplesPerSection = 24;
    public const int MinimumSections = 3;
    public float HalfWidth { get; private set; }

    /// <summary>Editable control points (XZ plane). Modify then call <see cref="Rebuild"/>.</summary>
    public List<(float x, float z)> ControlPoints { get; } = new();

    // sampled centerline
    private float[] _sx = Array.Empty<float>();
    private float[] _sz = Array.Empty<float>();
    private float[] _tx = Array.Empty<float>();
    private float[] _tz = Array.Empty<float>();
    private int _sampleCount;

    // spatial grid for fast nearest-centerline lookup
    private float _minX, _minZ, _cellSize;
    private int _cols, _rows;
    private List<int>[] _grid = Array.Empty<List<int>>();

    public Track(float halfWidth = 5f)
    {
        HalfWidth = halfWidth;
        AddDefaultLoop();
        Rebuild();
    }

    private void AddDefaultLoop()
    {
        float a = 30f, b = 20f;
        int n = 12;
        for (int i = 0; i < n; i++)
        {
            double ang = i * 2.0 * Math.PI / n;
            ControlPoints.Add(((float)(a * Math.Cos(ang)), (float)(b * Math.Sin(ang))));
        }
    }

    public void SetControlPoints(IEnumerable<(float x, float z)> points, float halfWidth)
    {
        ControlPoints.Clear();
        foreach (var p in points) ControlPoints.Add(p);
        HalfWidth = halfWidth;
        Rebuild();
    }

    /// <summary>Adds a road section by splitting the section after the selected control point.</summary>
    public int InsertSectionAfter(int index)
    {
        int next = (index + 1) % ControlPoints.Count;
        var a = ControlPoints[index];
        var b = ControlPoints[next];
        int inserted = index + 1;
        ControlPoints.Insert(inserted, ((a.x + b.x) * 0.5f, (a.z + b.z) * 0.5f));
        Rebuild();
        return inserted;
    }

    public bool RemoveSection(int index)
    {
        if (ControlPoints.Count <= MinimumSections || index < 0 || index >= ControlPoints.Count)
            return false;
        ControlPoints.RemoveAt(index);
        Rebuild();
        return true;
    }

    public void AdjustWidth(float delta)
    {
        HalfWidth = Math.Clamp(HalfWidth + delta, 2f, 15f);
        Rebuild();
    }

    /// <summary>Re-sample the spline and rebuild the spatial grid.</summary>
    public void Rebuild()
    {
        int n = ControlPoints.Count;
        if (n < 3)
        {
            ControlPoints.Clear();
            ControlPoints.Add((0f, 0f));
            ControlPoints.Add((20f, 0f));
            ControlPoints.Add((10f, 15f));
            n = 3;
        }

        const int perSeg = SamplesPerSection;
        int total = n * perSeg;
        _sx = new float[total];
        _sz = new float[total];
        _tx = new float[total];
        _tz = new float[total];

        float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;

        for (int i = 0; i < n; i++)
        {
            var p0 = ControlPoints[(i - 1 + n) % n];
            var p1 = ControlPoints[i];
            var p2 = ControlPoints[(i + 1) % n];
            var p3 = ControlPoints[(i + 2) % n];
            for (int s = 0; s < perSeg; s++)
            {
                float t = s / (float)perSeg;
                var (x, z) = CatmullRom(p0, p1, p2, p3, t);
                int idx = i * perSeg + s;
                _sx[idx] = x; _sz[idx] = z;
                if (x < minX) minX = x; if (x > maxX) maxX = x;
                if (z < minY) minY = z; if (z > maxY) maxY = z;
            }
        }
        _sampleCount = total;

        for (int i = 0; i < total; i++)
        {
            int a = (i - 1 + total) % total, b = (i + 1) % total;
            float dx = _sx[b] - _sx[a], dz = _sz[b] - _sz[a];
            float len = (float)Math.Sqrt(dx * dx + dz * dz);
            if (len < 1e-6f) len = 1e-6f;
            _tx[i] = dx / len; _tz[i] = dz / len;
        }

        BuildGrid(minX, minY, maxX, maxY);
    }

    private static (float x, float z) CatmullRom((float x, float z) p0, (float x, float z) p1,
        (float x, float z) p2, (float x, float z) p3, float t)
    {
        // Centripetal parameterization avoids the loops/cusps that uniform
        // Catmull-Rom creates when edited control points are unevenly spaced.
        float t0 = 0f;
        float t1 = t0 + ParameterDistance(p0, p1);
        float t2 = t1 + ParameterDistance(p1, p2);
        float t3 = t2 + ParameterDistance(p2, p3);
        float tt = t1 + (t2 - t1) * t;

        var a1 = Blend(p0, p1, t0, t1, tt);
        var a2 = Blend(p1, p2, t1, t2, tt);
        var a3 = Blend(p2, p3, t2, t3, tt);
        var b1 = Blend(a1, a2, t0, t2, tt);
        var b2 = Blend(a2, a3, t1, t3, tt);
        return Blend(b1, b2, t1, t2, tt);
    }

    private static float ParameterDistance((float x, float z) a, (float x, float z) b)
    {
        float dx = b.x - a.x, dz = b.z - a.z;
        return Math.Max(0.001f, (float)Math.Sqrt(Math.Sqrt(dx * dx + dz * dz)));
    }

    private static (float x, float z) Blend((float x, float z) a, (float x, float z) b,
        float ta, float tb, float t)
    {
        float span = Math.Max(0.001f, tb - ta);
        float wa = (tb - t) / span;
        float wb = (t - ta) / span;
        return (a.x * wa + b.x * wb, a.z * wa + b.z * wb);
    }

    private void BuildGrid(float minX, float minY, float maxX, float maxY)
    {
        _minX = minX - HalfWidth; _minZ = minY - HalfWidth;
        float w = (maxX + HalfWidth) - _minX;
        float h = (maxY + HalfWidth) - _minZ;
        _cellSize = Math.Max(1f, HalfWidth);
        _cols = Math.Max(1, (int)Math.Ceiling(w / _cellSize));
        _rows = Math.Max(1, (int)Math.Ceiling(h / _cellSize));
        _grid = new List<int>[_cols * _rows];
        for (int i = 0; i < _grid.Length; i++) _grid[i] = new List<int>();
        for (int i = 0; i < _sampleCount; i++)
        {
            int cx = CellX(_sx[i]), cz = CellZ(_sz[i]);
            _grid[cz * _cols + cx].Add(i);
        }
    }

    private int CellX(float x)
    {
        int c = (int)Math.Floor((x - _minX) / _cellSize);
        return Math.Clamp(c, 0, _cols - 1);
    }
    private int CellZ(float z)
    {
        int c = (int)Math.Floor((z - _minZ) / _cellSize);
        return Math.Clamp(c, 0, _rows - 1);
    }

    private (int index, float dist) NearestSample(float x, float z)
    {
        int cx = CellX(x), cz = CellZ(z);
        float best = float.MaxValue;
        int bestIdx = 0;
        for (int dz = -1; dz <= 1; dz++)
        for (int dx = -1; dx <= 1; dx++)
        {
            int gx = cx + dx, gz = cz + dz;
            if (gx < 0 || gz < 0 || gx >= _cols || gz >= _rows) continue;
            var cell = _grid[gz * _cols + gx];
            for (int k = 0; k < cell.Count; k++)
            {
                int i = cell[k];
                float ddx = _sx[i] - x, ddz = _sz[i] - z;
                float d2 = ddx * ddx + ddz * ddz;
                if (d2 < best) { best = d2; bestIdx = i; }
            }
        }
        return (bestIdx, (float)Math.Sqrt(best));
    }

    public bool IsOnTrack(float x, float z)
    {
        var (_, d) = NearestSample(x, z);
        return d <= HalfWidth;
    }

    /// <summary>Distance from the centerline (0 at center, ~HalfWidth at the edge).</summary>
    public float LateralOffset(float x, float z)
    {
        var (_, d) = NearestSample(x, z);
        return d;
    }

    /// <summary>Normalized 0..1 position around the track.</summary>
    public float ProgressAt(float x, float z)
    {
        var (idx, _) = NearestSample(x, z);
        return idx / (float)_sampleCount;
    }

    public (float x, float z) CenterAt(float t)
    {
        int i = (int)Math.Floor(t * _sampleCount) % _sampleCount;
        if (i < 0) i += _sampleCount;
        return (_sx[i], _sz[i]);
    }

    public (float x, float z) TangentAt(float t)
    {
        int i = (int)Math.Floor(t * _sampleCount) % _sampleCount;
        if (i < 0) i += _sampleCount;
        return (_tx[i], _tz[i]);
    }

    /// <summary>Distance to the nearest wall along a ray (0 if already off track).</summary>
    public float CastRay(float ox, float oz, float dx, float dz, float maxDist = 45f, float step = 0.5f)
    {
        float len = (float)Math.Sqrt(dx * dx + dz * dz);
        if (len < 1e-6f) return 0f;
        dx /= len; dz /= len;
        if (!IsOnTrack(ox, oz)) return 0f;

        for (float d = step; d <= maxDist; d += step)
        {
            float x = ox + dx * d, z = oz + dz * d;
            if (!IsOnTrack(x, z))
            {
                float lo = d - step, hi = d;
                for (int it = 0; it < 10; it++)
                {
                    float mid = (lo + hi) * 0.5f;
                    float mx = ox + dx * mid, mz = oz + dz * mid;
                    if (IsOnTrack(mx, mz)) lo = mid; else hi = mid;
                }
                return (lo + hi) * 0.5f;
            }
        }
        return maxDist;
    }

    /// <summary>Project a point back onto the centerline (position + heading).</summary>
    public (float x, float z, float heading) SnapToCenterline(float x, float z)
    {
        var (idx, _) = NearestSample(x, z);
        float heading = (float)Math.Atan2(_tz[idx], _tx[idx]);
        return (_sx[idx], _sz[idx], heading);
    }

    // ---- accessors for the renderer / editor ----
    public int SampleCount => _sampleCount;
    public float SampleX(int i) => _sx[i];
    public float SampleZ(int i) => _sz[i];
    public float TangentX(int i) => _tx[i];
    public float TangentZ(int i) => _tz[i];
}
