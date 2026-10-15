package io.github.jaelee18.mvtree.core

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import org.json.JSONObject
import java.nio.FloatBuffer
import kotlin.random.Random

/** `model_info.json`, written by `scripts/export/export_mobile.py` next to `model.onnx`. */
class ModelInfo(
    val imgSize: Int,
    val fold: Int,
    val species: List<String>,
    val mean: FloatArray,
    val std: FloatArray,
    val sha256: String,
    /** Accuracy of the shipped int8 model on its fold's held-out trees, if measured. */
    val heldOutOa: Double?,
    val heldOutMacroF1: Double?,
    val heldOutTrees: Int?,
) {
    companion object {
        fun parse(json: String): ModelInfo {
            val o = JSONObject(json)
            val stats = o.getJSONObject("render_stats")
            val species = o.getJSONArray("species").let { a -> List(a.length()) { a.getString(it) } }
            fun floats(key: String) = stats.getJSONArray(key).let { a -> FloatArray(a.length()) { a.getDouble(it).toFloat() } }
            val eval = o.optJSONObject("evaluation")?.optJSONObject("onnx_int8")
            return ModelInfo(
                imgSize = o.getInt("img_size"), fold = o.optInt("fold", 0), species = species,
                mean = floats("mean"), std = floats("std"), sha256 = o.optString("sha256"),
                heldOutOa = eval?.optDouble("oa"), heldOutMacroF1 = eval?.optDouble("macro_f1"),
                heldOutTrees = eval?.optInt("n"),
            )
        }
    }
}

/** Model input tensors, normalised exactly like torchvision's ToTensor + Normalize. */
class ModelInput(val side: FloatArray, val topbot: FloatArray, val detail: FloatArray, val height: Float, val size: Int) {
    companion object {
        fun from(views: Map<String, Renderer.Image>, height: Double, info: ModelInfo): ModelInput {
            val s = info.imgSize
            fun pack(names: List<String>): FloatArray {
                val out = FloatArray(names.size * 3 * s * s)
                names.forEachIndexed { v, name ->
                    val img = views.getValue(name)
                    require(img.size == s) { "render is ${img.size}px, model expects ${s}px" }
                    for (c in 0 until 3) {
                        val base = (v * 3 + c) * s * s
                        val mean = info.mean[c]
                        val std = info.std[c]
                        for (p in 0 until s * s) {
                            out[base + p] = ((img.rgb[p * 3 + c].toInt() and 0xFF).toFloat() / 255f - mean) / std
                        }
                    }
                }
                return out
            }
            return ModelInput(pack(Renderer.SIDE_VIEWS), pack(Renderer.TOPBOT_VIEWS), pack(listOf(Renderer.DETAIL_VIEW)),
                height.toFloat(), s)
        }
    }
}

/** One-fold ConvNeXtV2-Base x 3 classifier running in ONNX Runtime (int8 weights, CPU). */
class SpeciesClassifier(modelPath: String, val info: ModelInfo, threads: Int = 0) : AutoCloseable {
    private val env = OrtEnvironment.getEnvironment().also {
        runCatching { it.setTelemetry(false) }  // never phone home; the app has no network permission either
    }
    private val session: OrtSession = env.createSession(modelPath, OrtSession.SessionOptions().apply {
        setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        if (threads > 0) setIntraOpNumThreads(threads)
    })

    /** Class probabilities in the order of [ModelInfo.species]. */
    fun predict(input: ModelInput): FloatArray {
        val s = input.size.toLong()
        val tensors = mapOf(
            "side_views" to OnnxTensor.createTensor(env, FloatBuffer.wrap(input.side), longArrayOf(1, 4, 3, s, s)),
            "topbot_views" to OnnxTensor.createTensor(env, FloatBuffer.wrap(input.topbot), longArrayOf(1, 2, 3, s, s)),
            "detail_view" to OnnxTensor.createTensor(env, FloatBuffer.wrap(input.detail), longArrayOf(1, 3, s, s)),
            "tree_height" to OnnxTensor.createTensor(env, FloatBuffer.wrap(floatArrayOf(input.height)), longArrayOf(1, 1)),
        )
        try {
            session.run(tensors).use { result ->
                @Suppress("UNCHECKED_CAST")
                return (result[0].value as Array<FloatArray>)[0].copyOf()
            }
        } finally {
            tensors.values.forEach { it.close() }
        }
    }

    override fun close() = session.close()
}

/** Everything the app stores about one prediction. */
class Analysis(
    val probs: FloatArray,
    val views: Map<String, Renderer.Image>,
    val nPoints: Int,
    val treeHeight: Double,
    val millis: Map<String, Long>,
) {
    val best: Int get() = probs.indices.maxBy { probs[it] }
}

object Pipeline {
    /** Render, normalise, and classify one tree (the steps of `mvtree.predict.Predictor`, TTA 1). */
    fun analyze(cloud: PointCloud, classifier: SpeciesClassifier, progress: (String) -> Unit = {}): Analysis {
        val t0 = System.currentTimeMillis()
        progress("render")
        val views = Renderer.render(cloud, classifier.info.imgSize)
        val t1 = System.currentTimeMillis()
        val height = cloud.height
        progress("predict")
        val probs = classifier.predict(ModelInput.from(views, height, classifier.info))
        val t2 = System.currentTimeMillis()
        return Analysis(probs, views, cloud.size, height, mapOf("render" to t1 - t0, "predict" to t2 - t1))
    }
}

/** Display helpers (not model input). */
object Viz {
    /**
     * `mvtree.viz.on_white`: empty pixels become white; occupied pixels keep their channel values after a
     * 1-99 percentile contrast stretch. Returns ARGB pixels for an Android Bitmap.
     */
    fun onWhite(img: Renderer.Image): IntArray {
        val px = img.size * img.size
        val occupied = BooleanArray(px) { p -> img.rgb[p * 3] != 0.toByte() || img.rgb[p * 3 + 1] != 0.toByte() || img.rgb[p * 3 + 2] != 0.toByte() }
        val vals = ArrayList<Int>()
        for (p in 0 until px) if (occupied[p]) for (c in 0 until 3) vals.add(img.rgb[p * 3 + c].toInt() and 0xFF)
        val out = IntArray(px) { 0xFFFFFFFF.toInt() }
        if (vals.isEmpty()) return out
        val sorted = vals.sorted()
        fun percentile(q: Double): Double {
            val pos = q / 100.0 * (sorted.size - 1)
            val lo = pos.toInt()
            val hi = minOf(lo + 1, sorted.size - 1)
            return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo)
        }
        val lo = percentile(1.0)
        val hi = percentile(99.0)
        val range = maxOf(hi - lo, 1.0)
        fun stretch(v: Int) = (40 + 200 * ((v - lo) / range).coerceIn(0.0, 1.0)).toInt()
        for (p in 0 until px) if (occupied[p]) {
            val r = stretch(img.rgb[p * 3].toInt() and 0xFF)
            val g = stretch(img.rgb[p * 3 + 1].toInt() and 0xFF)
            val b = stretch(img.rgb[p * 3 + 2].toInt() and 0xFF)
            out[p] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return out
    }

    /** Up to [max] points, XY centred on the mean and Z from 0, for the 3D viewer. */
    fun viewerPoints(cloud: PointCloud, max: Int = 20_000): FloatArray {
        val n = cloud.size
        val pick = if (n <= max) IntArray(n) { it } else {
            val all = IntArray(n) { it }
            val rnd = Random(0)
            for (i in 0 until max) {
                val j = i + rnd.nextInt(n - i)
                val t = all[i]; all[i] = all[j]; all[j] = t
            }
            all.copyOf(max)
        }
        var mx = 0.0
        var my = 0.0
        var zmin = Double.POSITIVE_INFINITY
        for (i in pick) {
            mx += cloud.xyz[3 * i]
            my += cloud.xyz[3 * i + 1]
            zmin = minOf(zmin, cloud.xyz[3 * i + 2].toDouble())
        }
        mx /= pick.size
        my /= pick.size
        val out = FloatArray(pick.size * 3)
        pick.forEachIndexed { k, i ->
            out[3 * k] = (cloud.xyz[3 * i] - mx).toFloat()
            out[3 * k + 1] = (cloud.xyz[3 * i + 1] - my).toFloat()
            out[3 * k + 2] = (cloud.xyz[3 * i + 2] - zmin).toFloat()
        }
        return out
    }
}
