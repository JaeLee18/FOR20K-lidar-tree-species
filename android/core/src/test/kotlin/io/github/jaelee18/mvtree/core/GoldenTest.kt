package io.github.jaelee18.mvtree.core

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.security.MessageDigest
import javax.imageio.ImageIO
import kotlin.math.abs

/**
 * Checks the Kotlin pipeline against the Python package, using data written by
 * `android/tools/make_golden.py`: readers byte-for-byte, renders pixel-for-pixel,
 * and (when the model file is present) the int8 model's probabilities.
 */
class GoldenTest {
    private val resources = File("src/test/resources")
    private val golden = JSONObject(File(resources, "golden/golden.json").readText())
    private val examples = File("../../examples")

    private fun sha256(cloud: PointCloud): String {
        val bytes = ByteBuffer.allocate(cloud.xyz.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        cloud.xyz.forEach { bytes.putFloat(it) }
        return MessageDigest.getInstance("SHA-256").digest(bytes.array()).joinToString("") { "%02x".format(it) }
    }

    @Test
    fun readersMatchPython() {
        val formats = golden.getJSONObject("formats")
        for (name in formats.keys()) {
            val expected = formats.getJSONObject(name)
            val cloud = PointClouds.read(File(resources, "formats/$name"))
            assertEquals("$name: point count", expected.getInt("n"), cloud.size)
            assertEquals("$name: float32 XYZ", expected.getString("sha256"), sha256(cloud))
        }
        println("readers: ${formats.length()} files identical to mvtree.io")
    }

    @Test
    fun sniffingRecognisesEveryFormat() {
        for (file in File(resources, "formats").listFiles()!!.sortedBy { it.name }) {
            val head = file.inputStream().use { it.readNBytes(4096) }
            val expected = when (val ext = PointClouds.extension(file.name)) { "laz" -> "las"; else -> ext }
            assertEquals(file.name, expected, PointClouds.sniff(head))
            // A file named only by its sniffed format reads exactly like the original.
            val renamed = File.createTempFile("shared", ".${PointClouds.sniff(head)}").apply { deleteOnExit() }
            file.copyTo(renamed, overwrite = true)
            assertEquals(file.name, sha256(PointClouds.read(file)), sha256(PointClouds.read(renamed)))
        }
        assertEquals(null, PointClouds.sniff("hello, this is not a point cloud\n".toByteArray()))
        assertEquals(null, PointClouds.sniff(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(), 0, 1, 2)))
    }

    @Test
    fun examplesReadAndRenderLikePython() {
        val ex = golden.getJSONObject("examples")
        val size = golden.getInt("size")
        for (name in ex.keys().asSequence().sorted()) {
            val expected = ex.getJSONObject(name)
            val cloud = PointClouds.read(File(examples, name))
            assertEquals("$name: point count", expected.getInt("n"), cloud.size)
            assertEquals("$name: float32 XYZ", expected.getString("sha256"), sha256(cloud))
            assertEquals("$name: height", expected.getDouble("height"), cloud.height, 0.0)

            val t0 = System.nanoTime()
            val views = Renderer.render(cloud, size)
            val ms = (System.nanoTime() - t0) / 1e6
            var differing = 0
            var maxDelta = 0
            for (view in Renderer.VIEW_NAMES) {
                val png = ImageIO.read(File(resources, "golden/${name.removeSuffix(".laz")}/$view.png"))
                val img = views.getValue(view)
                for (row in 0 until size) for (col in 0 until size) {
                    val argb = png.getRGB(col, row)
                    val ref = intArrayOf((argb shr 16) and 0xFF, (argb shr 8) and 0xFF, argb and 0xFF)
                    var pixelDiffers = false
                    for (c in 0 until 3) {
                        val d = abs(img[row, col, c] - ref[c])
                        if (d > 0) pixelDiffers = true
                        maxDelta = maxOf(maxDelta, d)
                    }
                    if (pixelDiffers) differing++
                }
            }
            println("$name: ${cloud.size} points, 7 views in ${"%.0f".format(ms)} ms, " +
                "$differing of ${7 * size * size} pixels differ from Python (max delta $maxDelta)")
            assertEquals("$name: pixels differing from the Python renders", 0, differing)
        }
    }

    @Test
    fun modelMatchesPythonOnnxRuntime() {
        val model = File(System.getenv("MVTREE_MODEL") ?: "../app/src/main/assets/model.onnx")
        assumeTrue("model not exported (see scripts/export/export_mobile.py); skipping", model.isFile)
        val info = ModelInfo.parse(File(model.parentFile, "model_info.json").readText())
        val ex = golden.getJSONObject("examples")
        SpeciesClassifier(model.path, info).use { classifier ->
            for (name in ex.keys().asSequence().sorted()) {
                val expected = ex.getJSONObject(name)
                if (!expected.has("probs")) continue
                val analysis = Pipeline.analyze(PointClouds.read(File(examples, name)), classifier)
                val ref = expected.getJSONArray("probs")
                var maxDiff = 0.0
                for (i in 0 until ref.length()) maxDiff = maxOf(maxDiff, abs(analysis.probs[i] - ref.getDouble(i)))
                val species = info.species[analysis.best]
                println("$name -> $species ${"%.4f".format(analysis.probs[analysis.best])} " +
                    "(max |dp| vs Python ${"%.1e".format(maxDiff)}, ${analysis.millis})")
                assertEquals(expected.getString("species"), species)
                assert(maxDiff < 1e-4) { "$name: probabilities differ from Python by $maxDiff" }
            }
        }
    }
}
