package io.github.jaelee18.mvtree.core

import kotlin.math.abs
import kotlin.math.sqrt

/**
 * The seven orthographic views of one tree, a line-by-line port of `mvtree/render.py`.
 *
 * Channels are geometric maps, not colours: R = depth of the front-most point,
 * G = log(1 + point count), B = absolute height of the front-most point, each
 * min-max scaled per view. Unit tests check the output against the Python
 * renderer pixel by pixel, because the released weights only work on renders
 * made exactly this way.
 */
object Renderer {
    val VIEW_NAMES = listOf("side_000", "side_090", "side_180", "side_270", "top", "bottom", "detail")
    val SIDE_VIEWS = VIEW_NAMES.subList(0, 4)
    val TOPBOT_VIEWS = VIEW_NAMES.subList(4, 6)
    const val DETAIL_VIEW = "detail"

    private class View(val name: String, val azimuth: Double, val elevation: Double, val detail: Boolean)

    private val VIEWS = listOf(
        View("side_000", 0.0, 0.0, false), View("side_090", 90.0, 0.0, false),
        View("side_180", 180.0, 0.0, false), View("side_270", 270.0, 0.0, false),
        View("top", 0.0, 90.0, false), View("bottom", 0.0, -90.0, false),
        View("detail", 0.0, 0.0, true),
    )

    /** One view: `size * size * 3` bytes, row-major, channels interleaved (like a numpy HWC uint8 image). */
    class Image(val size: Int, val rgb: ByteArray) {
        operator fun get(row: Int, col: Int, channel: Int): Int = rgb[(row * size + col) * 3 + channel].toInt() and 0xFF
    }

    /** Render all seven views. [points] is float32-rounded XYZ, as read by [PointClouds]. */
    fun render(points: PointCloud, size: Int = 256): Map<String, Image> {
        require(points.size >= 3) { "need at least 3 points, got ${points.size}" }
        val n = points.size
        val xs = DoubleArray(n) { points.xyz[3 * it].toDouble() }
        val ys = DoubleArray(n) { points.xyz[3 * it + 1].toDouble() }
        val zs = DoubleArray(n) { points.xyz[3 * it + 2].toDouble() }
        // normalise_cloud: XY centred on the median, lowest point at Z = 0.
        val mx = median(xs)
        val my = median(ys)
        val zmin = zs.min()
        for (i in 0 until n) {
            xs[i] -= mx
            ys[i] -= my
            zs[i] -= zmin
        }
        val treeHeight = zs.max() - zs.min()

        val out = LinkedHashMap<String, Image>()
        for (view in VIEWS) {
            var idx: IntArray? = null  // null = all points
            if (view.detail) {
                // 1.0-1.5 m trunk slice for trees >= 5 m; shorter trees use 20-40 % of their height.
                val (lo, hi) = if (treeHeight >= 5.0) 1.0 to 1.5 else 0.20 * treeHeight to 0.40 * treeHeight
                idx = detailCrop(zs, lo, hi)
            }
            out[view.name] = if (idx != null && idx.isEmpty()) Image(size, ByteArray(size * size * 3))
            else orthoProject(xs, ys, zs, idx, view.azimuth, view.elevation, size)
        }
        return out
    }

    /** `np.median`: middle value, or the mean of the two middle values. */
    private fun median(values: DoubleArray): Double {
        val sorted = values.copyOf().also { it.sort() }
        val m = sorted.size / 2
        return if (sorted.size % 2 == 1) sorted[m] else (sorted[m - 1] + sorted[m]) / 2.0
    }

    /** Indices of the trunk slice, with the same fallbacks as `detail_crop`. */
    private fun detailCrop(z: DoubleArray, lo: Double, hi: Double, minPoints: Int = 20): IntArray? {
        var subset = z.indices.filter { z[it] >= lo && z[it] <= hi }
        if (subset.size < minPoints) subset = z.indices.filter { z[it] >= 0.5 && z[it] <= 2.0 }
        if (subset.size < minPoints) return null
        return subset.toIntArray()
    }

    private fun cross(a: DoubleArray, b: DoubleArray) = doubleArrayOf(
        a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0],
    )

    private fun norm(v: DoubleArray) = sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    /** Rows: image right, image up, viewing direction (`_rotation_matrix`). */
    internal fun rotation(azimuthDeg: Double, elevationDeg: Double): Array<DoubleArray> {
        val az = Math.toRadians(azimuthDeg)
        val el = Math.toRadians(elevationDeg)
        val cosAz = StrictMath.cos(az)
        val sinAz = StrictMath.sin(az)
        val cosEl = StrictMath.cos(el)
        val sinEl = StrictMath.sin(el)
        val forward = doubleArrayOf(cosEl * cosAz, cosEl * sinAz, sinEl)
        val worldUp = if (abs(sinEl) > 0.999) doubleArrayOf(0.0, 1.0, 0.0) else doubleArrayOf(0.0, 0.0, 1.0)
        var right = cross(forward, worldUp)
        val rn = norm(right)
        right = if (rn < 1e-9) doubleArrayOf(1.0, 0.0, 0.0) else doubleArrayOf(right[0] / rn, right[1] / rn, right[2] / rn)
        val up = cross(right, forward)
        val un = norm(up) + 1e-12
        return arrayOf(right, doubleArrayOf(up[0] / un, up[1] / un, up[2] / un), forward)
    }

    private fun orthoProject(
        xs: DoubleArray, ys: DoubleArray, zs: DoubleArray, idx: IntArray?,
        azimuth: Double, elevation: Double, size: Int,
    ): Image {
        val n = idx?.size ?: xs.size
        val r = rotation(azimuth, elevation)
        val u = DoubleArray(n)
        val v = DoubleArray(n)
        val depth = DoubleArray(n)
        val height = DoubleArray(n)
        for (k in 0 until n) {
            val i = idx?.get(k) ?: k
            val x = xs[i]
            val y = ys[i]
            val z = zs[i]
            u[k] = r[0][0] * x + r[0][1] * y + r[0][2] * z
            v[k] = r[1][0] * x + r[1][1] * y + r[1][2] * z
            depth[k] = r[2][0] * x + r[2][1] * y + r[2][2] * z
            height[k] = z
        }
        val margin = 0.02
        val uMin = u.min()
        val uMax = u.max()
        val vMin = v.min()
        val vMax = v.max()
        val span = maxOf(uMax - uMin, vMax - vMin)  // isotropic scale, no distortion
        if (span < 1e-6) return Image(size, ByteArray(size * size * 3))
        val uCenter = 0.5 * (uMin + uMax)
        val vCenter = 0.5 * (vMin + vMax)
        val scale = (1.0 - 2.0 * margin) / span

        // Per pixel: point count, plus depth and height of the front-most point. A sequential
        // strict-< scan keeps the smallest depth and, among ties, the earliest point.
        val pixels = size * size
        val count = IntArray(pixels)
        val frontDepth = DoubleArray(pixels) { Double.POSITIVE_INFINITY }
        val frontHeight = DoubleArray(pixels) { Double.NaN }
        for (k in 0 until n) {
            val px = ((u[k] - uCenter) * scale + 0.5) * size
            val py = ((vCenter - v[k]) * scale + 0.5) * size
            val col = px.toInt().coerceIn(0, size - 1)  // truncation toward zero, as numpy astype(int32)
            val row = py.toInt().coerceIn(0, size - 1)
            val p = row * size + col
            count[p]++
            if (depth[k] < frontDepth[p]) {
                frontDepth[p] = depth[k]
                frontHeight[p] = height[k]
            }
        }

        val occupied = (0 until pixels).filter { count[it] > 0 }.toIntArray()
        val rgb = ByteArray(pixels * 3)
        if (occupied.isNotEmpty()) {
            fill(rgb, occupied, 0) { frontDepth[it] }
            fill(rgb, occupied, 1) { StrictMath.log1p(count[it].toDouble()) }
            fill(rgb, occupied, 2) { frontHeight[it] }
        }
        return Image(size, rgb)
    }

    /** `_minmax_to_uint8` over the occupied pixels, written into one channel. */
    private inline fun fill(rgb: ByteArray, occupied: IntArray, channel: Int, value: (Int) -> Double) {
        val values = DoubleArray(occupied.size) { value(occupied[it]) }
        val lo = values.min()
        val hi = values.max()
        val range = hi - lo
        for (k in occupied.indices) {
            val b = if (range < 1e-9) 128 else ((values[k] - lo) / range * 255.0).coerceIn(0.0, 255.0).toInt()
            rgb[occupied[k] * 3 + channel] = b.toByte()
        }
    }
}
