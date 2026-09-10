using System;

namespace AiTestClient;

/// <summary>
/// Small 4x4 column-major matrix helpers (used for the camera and for
/// projecting / unprojecting points between world and screen space, which the
/// track editor relies on).
/// </summary>
public static class Mat4
{
    public static float[] Multiply(float[] a, float[] b)
    {
        var r = new float[16];
        for (int col = 0; col < 4; col++)
        for (int row = 0; row < 4; row++)
        {
            float sum = 0;
            for (int k = 0; k < 4; k++)
                sum += a[k * 4 + row] * b[col * 4 + k];
            r[col * 4 + row] = sum;
        }
        return r;
    }

    public static float[] Invert(float[] m)
    {
        float m00 = m[0], m01 = m[1], m02 = m[2], m03 = m[3];
        float m10 = m[4], m11 = m[5], m12 = m[6], m13 = m[7];
        float m20 = m[8], m21 = m[9], m22 = m[10], m23 = m[11];
        float m30 = m[12], m31 = m[13], m32 = m[14], m33 = m[15];

        float b00 = m00 * m11 - m01 * m10, b01 = m00 * m12 - m02 * m10, b02 = m00 * m13 - m03 * m10;
        float b03 = m01 * m12 - m02 * m11, b04 = m01 * m13 - m03 * m11, b05 = m02 * m13 - m03 * m12;
        float b06 = m20 * m31 - m21 * m30, b07 = m20 * m32 - m22 * m30, b08 = m20 * m33 - m23 * m30;
        float b09 = m21 * m32 - m22 * m31, b10 = m21 * m33 - m23 * m31, b11 = m22 * m33 - m23 * m32;

        float det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
        if (Math.Abs(det) < 1e-9f) return new float[16];
        det = 1f / det;

        var r = new float[16];
        r[0] = (m11 * b11 - m12 * b10 + m13 * b09) * det;
        r[1] = (m02 * b10 - m01 * b11 - m03 * b09) * det;
        r[2] = (m31 * b05 - m32 * b04 + m33 * b03) * det;
        r[3] = (m22 * b04 - m21 * b05 - m23 * b03) * det;
        r[4] = (m12 * b08 - m10 * b11 - m13 * b07) * det;
        r[5] = (m00 * b11 - m02 * b08 + m03 * b07) * det;
        r[6] = (m32 * b02 - m30 * b05 - m33 * b01) * det;
        r[7] = (m20 * b05 - m22 * b02 + m23 * b01) * det;
        r[8] = (m10 * b10 - m11 * b08 + m13 * b06) * det;
        r[9] = (m01 * b08 - m00 * b10 - m03 * b06) * det;
        r[10] = (m30 * b04 - m31 * b02 + m33 * b00) * det;
        r[11] = (m21 * b02 - m20 * b04 - m23 * b00) * det;
        r[12] = (m11 * b07 - m10 * b09 - m12 * b06) * det;
        r[13] = (m00 * b09 - m01 * b07 + m02 * b06) * det;
        r[14] = (m31 * b01 - m30 * b03 - m32 * b00) * det;
        r[15] = (m20 * b03 - m21 * b01 + m22 * b00) * det;
        return r;
    }

    public static (float x, float y, float z, float w) TransformPoint4(float[] m, float x, float y, float z)
    {
        float wx = m[0] * x + m[4] * y + m[8] * z + m[12];
        float wy = m[1] * x + m[5] * y + m[9] * z + m[13];
        float wz = m[2] * x + m[6] * y + m[10] * z + m[14];
        float ww = m[3] * x + m[7] * y + m[11] * z + m[15];
        return (wx, wy, wz, ww);
    }
}
