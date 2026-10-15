package io.github.jaelee18.mvtree

import android.content.ContentValues
import android.content.Context
import android.database.Cursor
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.File

/** One scan and what the model (and the person) said about it. */
data class Record(
    val id: Long,
    val createdAt: Long,
    val filename: String,
    val fileSize: Long,
    val site: String,
    val notes: String,
    val verifiedSpecies: String,
    val status: String,
    val error: String?,
    val startedAt: Long?,
    val finishedAt: Long?,
    val durationMs: Long?,
    val model: String?,
    val nPoints: Int?,
    val treeHeight: Double?,
    val species: String?,
    val confidence: Double?,
    val probs: FloatArray?,
    val latitude: Double?,
    val longitude: Double?,
    val locationAccuracy: Float?,
) {
    val pending: Boolean get() = status == STATUS_QUEUED || status == STATUS_RUNNING

    /** True/false once a verified species is entered and the model has an answer. */
    val match: Boolean? get() = if (verifiedSpecies.isBlank() || species == null) null else verifiedSpecies == species

    fun top(k: Int, species: List<String>): List<Pair<String, Float>> {
        val p = probs ?: return emptyList()
        return p.indices.sortedByDescending { p[it] }.take(k).map { species[it] to p[it] }
    }

    companion object {
        const val STATUS_QUEUED = "queued"
        const val STATUS_RUNNING = "running"
        const val STATUS_DONE = "done"
        const val STATUS_ERROR = "error"
    }
}

/** SQLite store (`records.db`) plus one folder of files per record under `files/records/<id>/`. */
class RecordStore(context: Context) : SQLiteOpenHelper(context, "records.db", null, 1) {
    private val root = File(context.filesDir, "records")
    private val _records = MutableStateFlow<List<Record>>(emptyList())
    val records: StateFlow<List<Record>> = _records

    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            """CREATE TABLE records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at INTEGER NOT NULL,
                filename TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                site TEXT NOT NULL DEFAULT '',
                notes TEXT NOT NULL DEFAULT '',
                verified_species TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL DEFAULT 'queued',
                error TEXT,
                started_at INTEGER,
                finished_at INTEGER,
                duration_ms INTEGER,
                model TEXT,
                n_points INTEGER,
                tree_height REAL,
                species TEXT,
                confidence REAL,
                probs TEXT,
                latitude REAL,
                longitude REAL,
                location_accuracy REAL
            )"""
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) = Unit

    fun dir(id: Long): File = File(root, id.toString())

    fun scanFile(id: Long): File? = dir(id).listFiles()?.firstOrNull { it.name.startsWith("scan.") }

    @Synchronized
    fun reload() {
        _records.value = readableDatabase.rawQuery("SELECT * FROM records ORDER BY id DESC", null).use { c ->
            buildList { while (c.moveToNext()) add(c.toRecord()) }
        }
    }

    fun get(id: Long): Record? =
        readableDatabase.rawQuery("SELECT * FROM records WHERE id = ?", arrayOf(id.toString())).use { c ->
            if (c.moveToFirst()) c.toRecord() else null
        }

    fun create(filename: String, fileSize: Long, site: String): Long {
        val id = writableDatabase.insertOrThrow("records", null, ContentValues().apply {
            put("created_at", System.currentTimeMillis())
            put("filename", filename)
            put("file_size", fileSize)
            put("site", site.trim().take(200))
        })
        reload()
        return id
    }

    private fun update(id: Long, values: ContentValues) {
        writableDatabase.update("records", values, "id = ?", arrayOf(id.toString()))
        reload()
    }

    fun setFileInfo(id: Long, filename: String, size: Long) = update(id, ContentValues().apply {
        put("filename", filename); put("file_size", size)
    })

    fun markQueued(id: Long) = update(id, ContentValues().apply {
        put("status", Record.STATUS_QUEUED); putNull("error"); putNull("started_at"); putNull("finished_at")
        putNull("duration_ms")
    })

    fun markRunning(id: Long, model: String) = update(id, ContentValues().apply {
        put("status", Record.STATUS_RUNNING); put("started_at", System.currentTimeMillis()); put("model", model)
    })

    fun finish(id: Long, nPoints: Int, treeHeight: Double, species: String, confidence: Double, probs: FloatArray,
               durationMs: Long) = update(id, ContentValues().apply {
        put("status", Record.STATUS_DONE); putNull("error")
        put("finished_at", System.currentTimeMillis()); put("duration_ms", durationMs)
        put("n_points", nPoints); put("tree_height", treeHeight)
        put("species", species); put("confidence", confidence)
        put("probs", probs.joinToString(",") { "%.6g".format(java.util.Locale.ROOT, it) })
    })

    fun fail(id: Long, message: String) = update(id, ContentValues().apply {
        put("status", Record.STATUS_ERROR); put("error", message.take(2000)); put("finished_at", System.currentTimeMillis())
    })

    fun edit(id: Long, verifiedSpecies: String, site: String, notes: String) = update(id, ContentValues().apply {
        put("verified_species", verifiedSpecies.trim().take(100)); put("site", site.trim().take(200))
        put("notes", notes.trim().take(5000))
    })

    fun setLocation(id: Long, latitude: Double?, longitude: Double?, accuracy: Float?) = update(id, ContentValues().apply {
        if (latitude == null) {
            putNull("latitude"); putNull("longitude"); putNull("location_accuracy")
        } else {
            put("latitude", latitude); put("longitude", longitude); put("location_accuracy", accuracy)
        }
    })

    fun delete(id: Long) {
        writableDatabase.delete("records", "id = ?", arrayOf(id.toString()))
        dir(id).deleteRecursively()
        reload()
    }

    fun unfinished(): List<Long> =
        readableDatabase.rawQuery("SELECT id FROM records WHERE status IN ('queued', 'running') ORDER BY id", null)
            .use { c -> buildList { while (c.moveToNext()) add(c.getLong(0)) } }

    private fun Cursor.toRecord(): Record {
        fun str(name: String): String? = getColumnIndexOrThrow(name).let { if (isNull(it)) null else getString(it) }
        fun long(name: String): Long? = getColumnIndexOrThrow(name).let { if (isNull(it)) null else getLong(it) }
        fun dbl(name: String): Double? = getColumnIndexOrThrow(name).let { if (isNull(it)) null else getDouble(it) }
        return Record(
            id = long("id")!!, createdAt = long("created_at")!!, filename = str("filename")!!,
            fileSize = long("file_size") ?: 0, site = str("site") ?: "", notes = str("notes") ?: "",
            verifiedSpecies = str("verified_species") ?: "", status = str("status")!!, error = str("error"),
            startedAt = long("started_at"), finishedAt = long("finished_at"), durationMs = long("duration_ms"),
            model = str("model"), nPoints = long("n_points")?.toInt(), treeHeight = dbl("tree_height"),
            species = str("species"), confidence = dbl("confidence"),
            probs = str("probs")?.split(',')?.map { it.toFloat() }?.toFloatArray(),
            latitude = dbl("latitude"), longitude = dbl("longitude"), locationAccuracy = dbl("location_accuracy")?.toFloat(),
        )
    }
}
