package io.github.jaelee18.mvtree

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import io.github.jaelee18.mvtree.core.ModelInfo
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.TimeZone

/** CSV with the same columns as the web app's export, plus the phone's location. */
object Export {
    private fun iso(ms: Long?): String = ms?.let {
        SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'", Locale.ROOT).apply { timeZone = TimeZone.getTimeZone("UTC") }.format(Date(it))
    } ?: ""

    private fun cell(v: Any?): String {
        val s = v?.toString() ?: ""
        return if (s.any { it == ',' || it == '"' || it == '\n' || it == '\r' }) "\"" + s.replace("\"", "\"\"") + "\"" else s
    }

    fun csv(records: List<Record>, info: ModelInfo?): String = buildString {
        val header = mutableListOf("id", "created_at_utc", "filename", "site", "status", "n_points", "tree_height_m",
            "species", "genus", "confidence")
        for (k in 2..5) header += listOf("top${k}_species", "top${k}_prob")
        header += listOf("verified_species", "match", "notes", "latitude", "longitude", "location_accuracy_m",
            "model", "duration_s", "error")
        appendLine(header.joinToString(","))
        for (r in records.sortedBy { it.id }) {
            val top = if (info != null) r.top(5, info.species).drop(1) else emptyList()
            val row = mutableListOf<Any?>(r.id, iso(r.createdAt), r.filename, r.site, r.status, r.nPoints,
                r.treeHeight?.let { "%.3f".format(Locale.ROOT, it) }, r.species, r.species?.substringBefore('_'),
                r.confidence?.let { "%.4f".format(Locale.ROOT, it) })
            for (k in 0 until 4) row.addAll(top.getOrNull(k)?.let { listOf(it.first, "%.4f".format(Locale.ROOT, it.second)) }
                ?: listOf("", ""))
            row.addAll(listOf(r.verifiedSpecies, r.match?.let { if (it) 1 else 0 }, r.notes, r.latitude, r.longitude,
                r.locationAccuracy, r.model, r.durationMs?.let { "%.1f".format(Locale.ROOT, it / 1000.0) }, r.error))
            appendLine(row.joinToString(",") { cell(it) })
        }
    }

    /** Write the CSV to the cache and open the share sheet. */
    fun share(context: Context, records: List<Record>, info: ModelInfo?) {
        val dir = File(context.cacheDir, "exports").apply { mkdirs() }
        val stamp = SimpleDateFormat("yyyyMMdd_HHmm", Locale.ROOT).format(Date())
        val file = File(dir, "tree_species_$stamp.csv").apply { writeText(csv(records, info)) }
        val uri = FileProvider.getUriForFile(context, "${context.packageName}.files", file)
        val send = Intent(Intent.ACTION_SEND).apply {
            type = "text/csv"
            putExtra(Intent.EXTRA_STREAM, uri)
            putExtra(Intent.EXTRA_SUBJECT, file.name)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
        context.startActivity(Intent.createChooser(send, "Export ${records.size} records").addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
    }
}
