package io.github.jaelee18.mvtree.ui

import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.withFrameNanos
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.platform.LocalDensity
import androidx.compose.ui.unit.dp
import androidx.compose.foundation.background
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.min
import kotlin.math.sin

private val VIEWER_BG = Color(0xFF10161A)
private val RAMP = listOf(0f to Color(0xFF8A5A2B), 0.4f to Color(0xFF6B8F3A), 0.7f to Color(0xFF3FAE5A), 1f to Color(0xFFC8F08F))
private const val BINS = 24

private fun ramp(t: Float): Color {
    for (k in 1 until RAMP.size) {
        val (t1, c1) = RAMP[k]
        if (t <= t1) {
            val (t0, c0) = RAMP[k - 1]
            val u = (t - t0) / (t1 - t0)
            return Color(c0.red + (c1.red - c0.red) * u, c0.green + (c1.green - c0.green) * u,
                c0.blue + (c1.blue - c0.blue) * u)
        }
    }
    return RAMP.last().second
}

/** Points grouped by height band so each band is one native drawPoints call. */
private class Prepared(val xyz: FloatArray) {
    val n = xyz.size / 3
    val zmax: Float
    val radius: Float
    val bin = IntArray(n)
    val counts = IntArray(BINS)
    val buffers: Array<FloatArray>

    init {
        var zm = 0f
        var r = 0f
        for (i in 0 until n) {
            zm = maxOf(zm, xyz[3 * i + 2])
            r = maxOf(r, hypot(xyz[3 * i], xyz[3 * i + 1]))
        }
        zmax = if (zm > 0f) zm else 1f
        radius = if (r > 0f) r else 1f
        for (i in 0 until n) bin[i] = ((xyz[3 * i + 2] / zmax) * (BINS - 1)).toInt().coerceIn(0, BINS - 1)
        for (b in bin) counts[b]++
        buffers = Array(BINS) { FloatArray(counts[it] * 2) }
    }
}

/** Interactive view of a record's points: drag to rotate, pinch to zoom, double-tap to reset. */
@Composable
fun PointCloudView(points: FloatArray, treeHeight: Double?, modifier: Modifier = Modifier) {
    val prepared = remember(points) { Prepared(points) }
    var yaw by remember { mutableFloatStateOf(0.5f) }
    var pitch by remember { mutableFloatStateOf(0.15f) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var spinning by remember { mutableStateOf(true) }
    val pointPx = with(LocalDensity.current) { 1.6.dp.toPx() }
    val paints = remember(pointPx) {
        Array(BINS) { b ->
            val c = ramp(b / (BINS - 1f))
            Paint().apply {
                color = android.graphics.Color.rgb((c.red * 255).toInt(), (c.green * 255).toInt(), (c.blue * 255).toInt())
                strokeWidth = pointPx
                strokeCap = Paint.Cap.ROUND
                isAntiAlias = true
            }
        }
    }

    LaunchedEffect(spinning) {
        var last = 0L
        while (spinning) {
            withFrameNanos { now ->
                if (last != 0L) yaw += (now - last) / 1e9f * 0.35f
                last = now
            }
        }
    }

    Box(modifier.clip(RoundedCornerShape(16.dp)).background(VIEWER_BG)) {
        Canvas(
            Modifier.fillMaxSize()
                .pointerInput(Unit) {
                    detectTransformGestures { _, pan, zoomChange, _ ->
                        spinning = false
                        yaw += pan.x * 0.01f
                        pitch = (pitch + pan.y * 0.01f).coerceIn(-1.35f, 1.35f)
                        zoom = (zoom * zoomChange).coerceIn(0.5f, 8f)
                    }
                }
                .pointerInput(Unit) {
                    detectTapGestures(onDoubleTap = { yaw = 0.5f; pitch = 0.15f; zoom = 1f; spinning = true },
                        onPress = { spinning = false })
                }
        ) {
            val p = prepared
            val w = size.width
            val h = size.height
            val zc = p.zmax / 2f
            val s = 0.86f * min(h / p.zmax, w / (2f * p.radius)) * zoom
            val cy = cos(yaw)
            val sy = sin(yaw)
            val cp = cos(pitch)
            val sp = sin(pitch)
            val fill = IntArray(BINS)
            val xyz = p.xyz
            for (i in 0 until p.n) {
                val x = xyz[3 * i]
                val y = xyz[3 * i + 1]
                val z = xyz[3 * i + 2] - zc
                val x1 = x * cy - y * sy
                val y1 = x * sy + y * cy
                val b = p.bin[i]
                val k = fill[b]++ * 2
                p.buffers[b][k] = w / 2f + x1 * s
                p.buffers[b][k + 1] = h / 2f - (z * cp - y1 * sp) * s
            }
            drawIntoCanvas { canvas ->
                for (b in 0 until BINS) {
                    if (fill[b] > 0) canvas.nativeCanvas.drawPoints(p.buffers[b], 0, fill[b] * 2, paints[b])
                }
            }
        }
        Row(Modifier.align(Alignment.TopEnd).padding(10.dp), verticalAlignment = Alignment.CenterVertically) {
            Column(horizontalAlignment = Alignment.End) {
                Text(treeHeight?.let { "%.1f m".format(it) } ?: "", style = MaterialTheme.typography.labelSmall,
                    color = Color.White.copy(alpha = 0.75f))
                Spacer(Modifier.height(28.dp))
                Text("0 m", style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.75f))
            }
            Spacer(Modifier.width(6.dp))
            Box(Modifier.size(width = 8.dp, height = 56.dp).clip(RoundedCornerShape(3.dp))
                .background(Brush.verticalGradient(RAMP.reversed().map { it.second })))
        }
        Text(
            "Drag to rotate · pinch to zoom · double-tap to reset",
            Modifier.align(Alignment.BottomStart).padding(10.dp),
            style = MaterialTheme.typography.labelSmall, color = Color.White.copy(alpha = 0.6f),
        )
    }
}
