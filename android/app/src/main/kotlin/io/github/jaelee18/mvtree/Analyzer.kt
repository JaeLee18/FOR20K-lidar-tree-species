package io.github.jaelee18.mvtree

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.net.Uri
import android.provider.OpenableColumns
import android.util.Log
import io.github.jaelee18.mvtree.core.ModelInfo
import io.github.jaelee18.mvtree.core.Pipeline
import io.github.jaelee18.mvtree.core.PointClouds
import io.github.jaelee18.mvtree.core.Renderer
import io.github.jaelee18.mvtree.core.SpeciesClassifier
import io.github.jaelee18.mvtree.core.Viz
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

sealed interface ModelState {
    /** [copying]: first launch, the model is being copied out of the APK; otherwise it is loading into memory. */
    data class Preparing(val progress: Float, val copying: Boolean = false) : ModelState
    data class Ready(val info: ModelInfo) : ModelState
    data class Failed(val message: String) : ModelState
}

/**
 * Runs the on-device model. One worker takes records in arrival order, so the
 * phone never holds two inputs at once; everything stays on the device.
 */
class Analyzer(private val context: Context, private val store: RecordStore) {
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.Default)
    private val jobs = Channel<Long>(Channel.UNLIMITED)
    private var classifier: SpeciesClassifier? = null

    private val _model = MutableStateFlow<ModelState>(ModelState.Preparing(0f))
    val model: StateFlow<ModelState> = _model

    /** Record id -> what the worker is doing right now. */
    private val _stage = MutableStateFlow<Map<Long, String>>(emptyMap())
    val stage: StateFlow<Map<Long, String>> = _stage

    val info: ModelInfo? get() = (model.value as? ModelState.Ready)?.info

    fun start() {
        scope.launch(Dispatchers.IO) {
            store.reload()
            for (id in store.unfinished()) {  // left over from a previous run
                store.markQueued(id)
                jobs.send(id)
            }
            if (prepareModel()) for (id in jobs) process(id)
        }
    }

    /** Copy the model out of the APK once (ONNX Runtime needs a file), then load it. */
    private fun prepareModel(): Boolean = try {
        val info = ModelInfo.parse(context.assets.open("model_info.json").bufferedReader().readText())
        val target = File(context.filesDir, "model.onnx")
        val marker = File(context.filesDir, "model.sha256")
        if (!target.isFile || !marker.isFile || marker.readText() != info.sha256) {
            val total = context.assets.openFd("model.onnx").use { it.length }.coerceAtLeast(1)
            val tmp = File(context.filesDir, "model.onnx.part")
            context.assets.open("model.onnx").use { input ->
                tmp.outputStream().use { out ->
                    val buf = ByteArray(1 shl 20)
                    var done = 0L
                    while (true) {
                        val n = input.read(buf)
                        if (n < 0) break
                        out.write(buf, 0, n)
                        done += n
                        _model.value = ModelState.Preparing(done.toFloat() / total, copying = true)
                    }
                }
            }
            tmp.renameTo(target)
            marker.writeText(info.sha256)
        }
        _model.value = ModelState.Preparing(1f, copying = false)
        classifier = SpeciesClassifier(target.path, info)
        _model.value = ModelState.Ready(info)
        true
    } catch (e: Throwable) {
        Log.e(TAG, "model failed to load", e)
        _model.value = ModelState.Failed("${e.javaClass.simpleName}: ${e.message}")
        false
    }

    /** Copy a picked or shared file into the app and queue it. Returns an error message, or null. */
    suspend fun add(uri: Uri, site: String): String? = withContext(Dispatchers.IO) {
        var name = displayName(uri) ?: uri.lastPathSegment?.substringAfterLast('/') ?: "scan"
        val id = store.create(name, 0, site)
        val dir = store.dir(id).apply { mkdirs() }
        val tmp = File(dir, "upload.tmp")
        try {
            val size = context.contentResolver.openInputStream(uri)!!.use { input ->
                tmp.outputStream().use { input.copyTo(it, 1 shl 20) }
            }
            // Trust the file name's extension; otherwise (e.g. "content://…/1234") recognise the format from its bytes.
            val ext = PointClouds.extension(name).takeIf { PointClouds.isSupported(name) }
                ?: PointClouds.sniff(tmp.inputStream().use { it.readNBytes(4096) })?.also { name = "$name.$it" }
                ?: throw IllegalArgumentException(
                    "not a point cloud (${PointClouds.EXTENSIONS.joinToString(" ") { ".$it" }})")
            tmp.renameTo(File(dir, "scan.$ext"))
            store.setFileInfo(id, name, size)
        } catch (e: Exception) {
            Log.w(TAG, "could not add $uri", e)
            store.delete(id)
            return@withContext "$name: ${if (e is IllegalArgumentException) e.message else "could not be read (${e.message})"}"
        }
        jobs.send(id)
        null
    }

    fun rerun(id: Long) {
        val rec = store.get(id) ?: return
        if (rec.pending || store.scanFile(id) == null) return
        store.markQueued(id)
        jobs.trySend(id)
    }

    private fun setStage(id: Long, stage: String?) =
        _stage.update { if (stage == null) it - id else it + (id to stage) }

    private fun process(id: Long) {
        val rec = store.get(id) ?: return
        val classifier = classifier ?: return
        val scan = store.scanFile(id) ?: run { store.fail(id, "scan file is missing"); return }
        val t0 = System.currentTimeMillis()
        store.markRunning(id, describe(classifier.info))
        try {
            setStage(id, "Reading points")
            val cloud = PointClouds.read(scan)  // scan.<ext>: the format chosen when the file was added
            val analysis = Pipeline.analyze(cloud, classifier) { step ->
                setStage(id, if (step == "render") "Rendering views" else "Running the model")
            }
            setStage(id, "Saving")
            saveAssets(id, analysis.views, Viz.viewerPoints(cloud))
            val best = analysis.best
            store.finish(id, analysis.nPoints, analysis.treeHeight, classifier.info.species[best],
                analysis.probs[best].toDouble(), analysis.probs, System.currentTimeMillis() - t0)
            Log.i(TAG, "record $id ${rec.filename}: ${classifier.info.species[best]} in ${System.currentTimeMillis() - t0} ms")
        } catch (e: Throwable) {
            Log.e(TAG, "record $id failed", e)
            store.fail(id, if (e is OutOfMemoryError) "Out of memory: this scan has too many points for the phone"
                           else "${e.javaClass.simpleName}: ${e.message}")
        } finally {
            setStage(id, null)
        }
    }

    private fun saveAssets(id: Long, views: Map<String, Renderer.Image>, points: FloatArray) {
        val dir = store.dir(id)
        if (!dir.isDirectory) return  // deleted while running
        val bitmaps = Renderer.VIEW_NAMES.map { name ->
            val img = views.getValue(name)
            Bitmap.createBitmap(Viz.onWhite(img), img.size, img.size, Bitmap.Config.ARGB_8888)
        }
        // Montage: the seven model-input views side by side, labelled.
        val tile = 160
        val pad = 6
        val label = 18
        val montage = Bitmap.createBitmap(7 * (tile + pad) + pad, tile + 2 * pad + label, Bitmap.Config.ARGB_8888)
        Canvas(montage).apply {
            drawColor(Color.WHITE)
            val nearest = Paint().apply { isFilterBitmap = false }
            val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { color = Color.rgb(70, 70, 70); textSize = 12f }
            bitmaps.forEachIndexed { i, bmp ->
                val x = pad + i * (tile + pad)
                drawBitmap(bmp, null, Rect(x, pad, x + tile, pad + tile), nearest)
                drawText(Renderer.VIEW_NAMES[i], x + 2f, pad + tile + 14f, text)
            }
        }
        File(dir, "montage.png").outputStream().use { montage.compress(Bitmap.CompressFormat.PNG, 100, it) }
        Bitmap.createScaledBitmap(bitmaps[0], 112, 112, false).let { thumb ->
            File(dir, "thumb.png").outputStream().use { thumb.compress(Bitmap.CompressFormat.PNG, 100, it) }
        }
        val bytes = ByteBuffer.allocate(points.size * 4).order(ByteOrder.LITTLE_ENDIAN)
        points.forEach { bytes.putFloat(it) }
        File(dir, "points.f32").writeBytes(bytes.array())
    }

    private fun displayName(uri: Uri): String? = runCatching {
        context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
            if (c.moveToFirst()) c.getString(0) else null
        }
    }.getOrNull()

    companion object {
        private const val TAG = "mvtree"
        fun describe(info: ModelInfo) = "ConvNeXtV2-Base x3, ${info.imgSize}px, fold ${info.fold}, int8, on device"
    }
}
