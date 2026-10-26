package com.geoai.capture.camera

import android.graphics.Bitmap
import android.media.Image
import kotlin.math.max
import kotlin.math.min

/**
 * Cheap per-frame statistics from the small YUV analysis stream (never the 4K stream):
 * road-region luma for exposure control, a sharpness index for the operator, and a small colour
 * preview bitmap. Must return quickly; the caller closes the Image right after.
 */
class AnalysisProcessor(private val previewWidth: Int = 480) {

    data class Result(
        val timestampNs: Long,
        val meanLuma: Double,
        val highlightFraction: Double,
        val darkFraction: Double,
        val sharpness: Double,
        val preview: Bitmap?,
    )

    /**
     * Road region (fractions of the image) for luma and sharpness. Default: lower 60 % of the frame, central 80 %
     * of the width. After the focus plan it is the road rows only ([setRoi]), so exposure and the focus sweep
     * never measure sky, buildings or vehicles. Set from the camera thread, read on the analysis thread.
     */
    @Volatile private var roi = DEFAULT_ROI

    private data class Roi(val top: Double, val bottom: Double, val left: Double, val right: Double)

    fun setRoi(top: Double, bottom: Double, left: Double, right: Double) {
        val t = top.coerceIn(0.0, 1.0)
        val b = bottom.coerceIn(0.0, 1.0)
        val l = left.coerceIn(0.0, 1.0)
        val r = right.coerceIn(0.0, 1.0)
        // A region too thin to measure would make exposure and sharpness meaningless: keep the default then.
        roi = if (b - t >= 0.05 && r - l >= 0.05) Roi(t, b, l, r) else DEFAULT_ROI
    }

    fun resetRoi() {
        roi = DEFAULT_ROI
    }

    fun process(image: Image, makePreview: Boolean): Result {
        val w = image.width
        val h = image.height
        val yPlane = image.planes[0]
        val yBuf = yPlane.buffer
        val yRow = yPlane.rowStride
        val yPix = yPlane.pixelStride

        val step = max(1, w / 160)
        val r = roi
        val x0 = (w * r.left).toInt()
        val x1 = (w * r.right).toInt()
        val y0 = (h * r.top).toInt()
        val y1 = max(y0 + 1, (h * r.bottom).toInt())
        var sum = 0L
        var n = 0L
        var high = 0L
        var dark = 0L
        var lapSq = 0.0
        var lapN = 0L
        var y = y0
        while (y < y1) {
            var x = x0
            while (x < x1) {
                val v = yBuf.get(y * yRow + x * yPix).toInt() and 0xff
                sum += v
                n++
                if (v > 250) high++
                if (v < 5) dark++
                if (x - step >= x0 && x + step < x1 && y - step >= y0 && y + step < y1) {
                    val l = yBuf.get(y * yRow + (x - step) * yPix).toInt() and 0xff
                    val r = yBuf.get(y * yRow + (x + step) * yPix).toInt() and 0xff
                    val u = yBuf.get((y - step) * yRow + x * yPix).toInt() and 0xff
                    val d = yBuf.get((y + step) * yRow + x * yPix).toInt() and 0xff
                    val lap = (4 * v - l - r - u - d).toDouble()
                    lapSq += lap * lap
                    lapN++
                }
                x += step
            }
            y += step
        }
        val mean = if (n > 0) sum.toDouble() / n else 0.0
        val preview = if (makePreview) toPreview(image) else null
        return Result(
            timestampNs = image.timestamp,
            meanLuma = mean,
            highlightFraction = if (n > 0) high.toDouble() / n else 0.0,
            darkFraction = if (n > 0) dark.toDouble() / n else 0.0,
            sharpness = if (lapN > 0) lapSq / lapN else 0.0,
            preview = preview,
        )
    }

    private fun toPreview(image: Image): Bitmap {
        val w = image.width
        val h = image.height
        val pw = min(previewWidth, w)
        val ph = max(1, (h.toLong() * pw / w).toInt())
        val yP = image.planes[0]
        val uP = image.planes[1]
        val vP = image.planes[2]
        val yB = yP.buffer
        val uB = uP.buffer
        val vB = vP.buffer
        val out = IntArray(pw * ph)
        for (py in 0 until ph) {
            val sy = (py.toLong() * h / ph).toInt()
            val cy = sy / 2
            for (px in 0 until pw) {
                val sx = (px.toLong() * w / pw).toInt()
                val cx = sx / 2
                val yy = (yB.get(sy * yP.rowStride + sx * yP.pixelStride).toInt() and 0xff)
                val uu = (uB.get(cy * uP.rowStride + cx * uP.pixelStride).toInt() and 0xff) - 128
                val vv = (vB.get(cy * vP.rowStride + cx * vP.pixelStride).toInt() and 0xff) - 128
                // BT.601 full-range (JFIF) is the usual camera YUV; exact colour is not needed for a preview.
                val r = clamp(yy + ((91881 * vv) shr 16))
                val g = clamp(yy - ((22554 * uu + 46802 * vv) shr 16))
                val b = clamp(yy + ((116130 * uu) shr 16))
                out[py * pw + px] = (0xff shl 24) or (r shl 16) or (g shl 8) or b
            }
        }
        return Bitmap.createBitmap(out, pw, ph, Bitmap.Config.ARGB_8888)
    }

    private fun clamp(v: Int) = if (v < 0) 0 else if (v > 255) 255 else v

    private companion object {
        val DEFAULT_ROI = Roi(0.40, 1.0, 0.10, 0.90)
    }
}
