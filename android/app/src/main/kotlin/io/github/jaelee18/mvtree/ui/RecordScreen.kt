package io.github.jaelee18.mvtree.ui

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.LocationOn
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.MenuAnchorType
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.produceState
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.FilterQuality
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.jaelee18.mvtree.ModelState
import io.github.jaelee18.mvtree.PhoneLocation
import io.github.jaelee18.mvtree.Record
import io.github.jaelee18.mvtree.mvtree
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordScreen(id: Long, onBack: () -> Unit, snackbar: SnackbarHostState) {
    val context = LocalContext.current
    val app = context.mvtree
    val records by app.store.records.collectAsStateWithLifecycle()
    val stage by app.analyzer.stage.collectAsStateWithLifecycle()
    val model by app.analyzer.model.collectAsStateWithLifecycle()
    val record = records.firstOrNull { it.id == id }
    if (record == null) {
        LaunchedEffect(Unit) { onBack() }
        return
    }
    val species = (model as? ModelState.Ready)?.info?.species.orEmpty()
    var confirmDelete by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    Scaffold(
        topBar = {
            TopAppBar(
                navigationIcon = { IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Filled.ArrowBack, "Back") } },
                title = { Text(record.filename, maxLines = 1, overflow = TextOverflow.Ellipsis) },
                actions = {
                    IconButton(onClick = { app.analyzer.rerun(id) }, enabled = !record.pending && app.store.scanFile(id) != null) {
                        Icon(Icons.Default.Refresh, "Run again")
                    }
                    IconButton(onClick = { confirmDelete = true }, enabled = record.status != Record.STATUS_RUNNING) {
                        Icon(Icons.Default.Delete, "Delete")
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        Column(
            Modifier.fillMaxSize().padding(pad).imePadding().verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp),
        ) {
            StatusBanner(record, stage[id])
            if (record.status == Record.STATUS_DONE) {
                val dir = app.store.dir(id)
                val points by produceState<FloatArray?>(null, record.finishedAt) {
                    value = withContext(Dispatchers.IO) { readPoints(File(dir, "points.f32")) }
                }
                points?.let { PointCloudView(it, record.treeHeight, Modifier.fillMaxWidth().aspectRatio(4f / 3f)) }
                Prediction(record, species)
                ModelInput(File(dir, "montage.png"), record.finishedAt)
            }
            RecordForm(record, species, snackbar)
            Details(record)
            Spacer(Modifier.height(24.dp))
        }
    }

    if (confirmDelete) {
        AlertDialog(
            onDismissRequest = { confirmDelete = false },
            title = { Text("Delete this record?") },
            text = { Text("${record.filename}, its prediction, notes, and the stored scan are removed from this phone.") },
            confirmButton = {
                TextButton(onClick = {
                    confirmDelete = false
                    scope.launch(Dispatchers.IO) { app.store.delete(id) }
                    onBack()
                }) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = { TextButton(onClick = { confirmDelete = false }) { Text("Cancel") } },
        )
    }
}

private fun readPoints(file: File): FloatArray? {
    if (!file.isFile) return null
    val buf = ByteBuffer.wrap(file.readBytes()).order(ByteOrder.LITTLE_ENDIAN)
    return FloatArray(buf.remaining() / 4) { buf.getFloat() }
}

@Composable
private fun StatusBanner(r: Record, stage: String?) {
    when (r.status) {
        Record.STATUS_QUEUED, Record.STATUS_RUNNING -> Card(
            colors = CardDefaults.cardColors(containerColor = WaitColor.copy(alpha = 0.12f))
        ) {
            Row(Modifier.fillMaxWidth().padding(14.dp), verticalAlignment = Alignment.CenterVertically) {
                CircularProgressIndicator(Modifier.size(20.dp), color = WaitColor, strokeWidth = 2.5.dp)
                Spacer(Modifier.width(12.dp))
                Column {
                    Text(stage ?: "Waiting in queue", color = WaitColor, fontWeight = FontWeight.SemiBold)
                    Text("Usually a few seconds on a recent phone.", style = MaterialTheme.typography.bodySmall,
                        color = WaitColor)
                }
            }
        }
        Record.STATUS_ERROR -> Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
            Column(Modifier.fillMaxWidth().padding(14.dp)) {
                Text("Prediction failed", fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onErrorContainer)
                Text(r.error ?: "", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }
        else -> Unit
    }
}

@Composable
private fun Prediction(r: Record, species: List<String>) {
    Column {
        Row(verticalAlignment = Alignment.Bottom) {
            SpeciesText(r.species, Modifier.weight(1f), style = MaterialTheme.typography.headlineSmall, weight = FontWeight.SemiBold)
            Text(pct(r.confidence), style = MaterialTheme.typography.headlineSmall, fontWeight = FontWeight.SemiBold)
        }
        Row {
            Text("Genus ", style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(r.species?.substringBefore('_') ?: "", style = MaterialTheme.typography.bodySmall,
                fontStyle = FontStyle.Italic, color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(" · top of 33 species", style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Spacer(Modifier.height(10.dp))
        r.top(5, species).forEachIndexed { i, (s, p) -> ProbabilityRow(s, p, strong = i == 0) }
        if ((r.confidence ?: 1.0) < 0.6) {
            Spacer(Modifier.height(8.dp))
            Text("Low confidence: check the alternatives above, or verify in the field.",
                style = MaterialTheme.typography.bodySmall, color = WaitColor)
        }
    }
}

@Composable
private fun ModelInput(montage: File, version: Long?) {
    val bmp = rememberBitmap(montage, version) ?: return
    Column {
        Text("MODEL INPUT", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(6.dp))
        Image(
            bmp, "The seven views the model saw",
            Modifier.fillMaxWidth().background(Color.White, RoundedCornerShape(8.dp)).padding(2.dp),
            contentScale = ContentScale.FillWidth, filterQuality = FilterQuality.None,
        )
        Text("Four side views, top, bottom, and a trunk close-up, encoding depth, point density, and height.",
            style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(top = 4.dp))
    }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
private fun RecordForm(r: Record, species: List<String>, snackbar: SnackbarHostState) {
    val context = LocalContext.current
    val app = context.mvtree
    val scope = rememberCoroutineScope()
    // Shown as "Fagus sylvatica"; stored as the model's key "Fagus_sylvatica" when it names one of the 33 species.
    fun canonical(text: String): String =
        text.trim().let { t -> species.firstOrNull { it.equals(t.replace(' ', '_'), ignoreCase = true) } ?: t }
    var verified by remember(r.id) { mutableStateOf(r.verifiedSpecies.replace('_', ' ')) }
    var site by remember(r.id) { mutableStateOf(r.site) }
    var notes by remember(r.id) { mutableStateOf(r.notes) }
    var expanded by remember { mutableStateOf(false) }
    var locating by remember { mutableStateOf(false) }
    val dirty = canonical(verified) != r.verifiedSpecies || site.trim() != r.site || notes.trim() != r.notes

    fun locate() {
        locating = true
        scope.launch {
            val fix = PhoneLocation.current(context)
            locating = false
            if (fix == null) snackbar.showSnackbar("No location fix. Is location turned on?")
            else withContext(Dispatchers.IO) { app.store.setLocation(r.id, fix.latitude, fix.longitude, fix.accuracy) }
        }
    }
    val permission = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { granted ->
        if (granted.values.any { it }) locate()
        else scope.launch { snackbar.showSnackbar("Location permission denied") }
    }

    Column {
        HorizontalDivider()
        Spacer(Modifier.height(12.dp))
        Text("YOUR RECORD", style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
        Spacer(Modifier.height(8.dp))
        ExposedDropdownMenuBox(expanded, onExpandedChange = { expanded = it }) {
            val q = verified.trim().lowercase().replace('_', ' ')
            val options = species.filter { q.isEmpty() || it.lowercase().replace('_', ' ').contains(q) }
            OutlinedTextField(
                verified, { verified = it.take(100); expanded = true },
                label = { Text("Verified species") }, placeholder = { Text("What the tree really is, if known") },
                singleLine = true, trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(expanded) },
                modifier = Modifier.fillMaxWidth().menuAnchor(MenuAnchorType.PrimaryEditable),
            )
            if (options.isNotEmpty()) {
                ExposedDropdownMenu(expanded, onDismissRequest = { expanded = false }) {
                    options.forEach { s ->
                        DropdownMenuItem(text = { SpeciesText(s) }, onClick = { verified = s.replace('_', ' '); expanded = false })
                    }
                }
            }
        }
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(site, { site = it.take(200) }, label = { Text("Site or plot") }, singleLine = true,
            modifier = Modifier.fillMaxWidth())
        Spacer(Modifier.height(8.dp))
        OutlinedTextField(notes, { notes = it.take(5000) }, label = { Text("Notes") },
            placeholder = { Text("DBH, health, photo IDs, anything worth keeping") }, minLines = 3,
            modifier = Modifier.fillMaxWidth())
        Spacer(Modifier.height(10.dp))
        Row(verticalAlignment = Alignment.CenterVertically) {
            Button(enabled = dirty, onClick = {
                scope.launch {
                    withContext(Dispatchers.IO) { app.store.edit(r.id, canonical(verified), site, notes) }
                    snackbar.showSnackbar("Saved")
                }
            }) { Text("Save") }
            Spacer(Modifier.width(10.dp))
            OutlinedButton(enabled = !locating, onClick = {
                val granted = listOf(Manifest.permission.ACCESS_FINE_LOCATION, Manifest.permission.ACCESS_COARSE_LOCATION)
                    .any { ContextCompat.checkSelfPermission(context, it) == PackageManager.PERMISSION_GRANTED }
                if (granted) locate() else permission.launch(arrayOf(Manifest.permission.ACCESS_FINE_LOCATION,
                    Manifest.permission.ACCESS_COARSE_LOCATION))
            }) {
                if (locating) CircularProgressIndicator(Modifier.size(16.dp), strokeWidth = 2.dp)
                else Icon(Icons.Default.LocationOn, null, Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text(if (r.latitude == null) "Add location" else "Update location")
            }
        }
        if (r.latitude != null && r.longitude != null) {
            Row(Modifier.padding(top = 6.dp), verticalAlignment = Alignment.CenterVertically) {
                Text("%.6f, %.6f".format(Locale.ROOT, r.latitude, r.longitude) +
                    (r.locationAccuracy?.let { "  (±%.0f m)".format(Locale.ROOT, it) } ?: ""),
                    style = MaterialTheme.typography.bodySmall, modifier = Modifier.weight(1f))
                TextButton(onClick = { scope.launch(Dispatchers.IO) { app.store.setLocation(r.id, null, null, null) } }) {
                    Text("Remove")
                }
            }
        }
    }
}

@Composable
private fun Details(r: Record) {
    val rows = listOf(
        "Points" to (r.nPoints?.let { "%,d".format(it) } ?: "–"),
        "Tree height" to (r.treeHeight?.let { "%.2f m".format(it) } ?: "–"),
        "File size" to formatBytes(r.fileSize),
        "Prediction time" to (r.durationMs?.let { "%.1f s".format(it / 1000.0) } ?: "–"),
        "Model" to (r.model ?: "–"),
        "Added" to formatDate(r.createdAt),
        "Finished" to formatDate(r.finishedAt),
        "Record" to "#${r.id}",
    )
    Column {
        HorizontalDivider()
        Spacer(Modifier.height(12.dp))
        rows.forEach { (k, v) ->
            Row(Modifier.padding(vertical = 3.dp)) {
                Text(k, Modifier.width(120.dp), style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant)
                Text(v, style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}
