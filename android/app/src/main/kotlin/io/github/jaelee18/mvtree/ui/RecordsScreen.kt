package io.github.jaelee18.mvtree.ui

import android.content.Context
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Add
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Edit
import androidx.compose.material.icons.filled.Info
import androidx.compose.material.icons.filled.MoreVert
import androidx.compose.material.icons.filled.Place
import androidx.compose.material.icons.filled.Search
import androidx.compose.material.icons.filled.Share
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import io.github.jaelee18.mvtree.Export
import io.github.jaelee18.mvtree.ModelState
import io.github.jaelee18.mvtree.Record
import io.github.jaelee18.mvtree.mvtree
import kotlinx.coroutines.launch
import java.io.File

private const val PREFS = "mvtree"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun RecordsScreen(onOpen: (Long) -> Unit, snackbar: SnackbarHostState) {
    val context = LocalContext.current
    val app = context.mvtree
    val records by app.store.records.collectAsStateWithLifecycle()
    val model by app.analyzer.model.collectAsStateWithLifecycle()
    val stage by app.analyzer.stage.collectAsStateWithLifecycle()
    val scope = rememberCoroutineScope()
    val prefs = remember { context.getSharedPreferences(PREFS, Context.MODE_PRIVATE) }

    var site by rememberSaveable { mutableStateOf(prefs.getString("site", "") ?: "") }
    var query by rememberSaveable { mutableStateOf("") }
    var searching by rememberSaveable { mutableStateOf(false) }
    var editingSite by remember { mutableStateOf(false) }
    var showAbout by remember { mutableStateOf(false) }
    var menu by remember { mutableStateOf(false) }

    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenMultipleDocuments()) { uris ->
        scope.launch {
            for (uri in uris) app.analyzer.add(uri, site)?.let { snackbar.showSnackbar(it) }
        }
    }

    val shown = remember(records, query) {
        val q = query.trim().lowercase().replace('_', ' ')
        if (q.isEmpty()) records else records.filter { r ->
            listOf(r.filename, r.site, r.notes, r.species ?: "", r.verifiedSpecies)
                .any { it.lowercase().replace('_', ' ').contains(q) }
        }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    if (searching) {
                        TextField(
                            query, { query = it }, placeholder = { Text("Search file, site, species, notes") },
                            singleLine = true, modifier = Modifier.fillMaxWidth(),
                            colors = TextFieldDefaults.colors(
                                focusedContainerColor = Color.Transparent, unfocusedContainerColor = Color.Transparent,
                            ),
                        )
                    } else {
                        Column {
                            Text("Tree Species", fontWeight = FontWeight.SemiBold)
                            Text(modelLine(model), style = MaterialTheme.typography.labelMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant)
                        }
                    }
                },
                actions = {
                    IconButton(onClick = { searching = !searching; if (!searching) query = "" }) {
                        Icon(if (searching) Icons.Default.Close else Icons.Default.Search,
                            if (searching) "Close search" else "Search")
                    }
                    IconButton(onClick = { menu = true }) { Icon(Icons.Default.MoreVert, "More") }
                    DropdownMenu(menu, onDismissRequest = { menu = false }) {
                        DropdownMenuItem(
                            text = { Text("Export CSV (${records.size})") },
                            leadingIcon = { Icon(Icons.Default.Share, null) },
                            enabled = records.isNotEmpty(),
                            onClick = { menu = false; Export.share(context, records, app.analyzer.info) },
                        )
                        DropdownMenuItem(
                            text = { Text("About the model") },
                            leadingIcon = { Icon(Icons.Default.Info, null) },
                            onClick = { menu = false; showAbout = true },
                        )
                    }
                },
                colors = TopAppBarDefaults.topAppBarColors(containerColor = MaterialTheme.colorScheme.background),
            )
        },
        floatingActionButton = {
            ExtendedFloatingActionButton(
                text = { Text("Add scans") }, icon = { Icon(Icons.Default.Add, null) },
                onClick = { picker.launch(arrayOf("*/*")) },
            )
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { pad ->
        LazyColumn(
            Modifier.fillMaxSize(),
            contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = pad.calculateTopPadding() + 4.dp,
                bottom = pad.calculateBottomPadding() + 96.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp),
        ) {
            item { ModelBanner(model) }
            item { SummaryCard(records, site) { editingSite = true } }
            if (shown.isEmpty()) {
                item { EmptyState(searching = query.isNotBlank()) }
            }
            items(shown, key = { it.id }) { record ->
                RecordRow(record, stage[record.id], File(app.store.dir(record.id), "thumb.png")) { onOpen(record.id) }
            }
        }
    }

    if (editingSite) {
        var draft by remember { mutableStateOf(site) }
        AlertDialog(
            onDismissRequest = { editingSite = false },
            title = { Text("Site or plot") },
            text = {
                Column {
                    Text("Saved with every scan you add from now on.", style = MaterialTheme.typography.bodyMedium)
                    Spacer(Modifier.height(12.dp))
                    OutlinedTextField(draft, { draft = it.take(200) }, singleLine = true,
                        placeholder = { Text("e.g. Plot 12, north transect") })
                }
            },
            confirmButton = {
                TextButton(onClick = {
                    site = draft.trim()
                    prefs.edit().putString("site", site).apply()
                    editingSite = false
                }) { Text("Save") }
            },
            dismissButton = { TextButton(onClick = { editingSite = false }) { Text("Cancel") } },
        )
    }
    if (showAbout) AboutDialog(model) { showAbout = false }
}

private fun modelLine(model: ModelState) = when (model) {
    is ModelState.Preparing -> if (model.copying) "Setting up the model…" else "Loading the model…"
    is ModelState.Failed -> "Model unavailable"
    is ModelState.Ready -> "On-device model ready · works offline"
}

@Composable
private fun ModelBanner(model: ModelState) {
    when (model) {
        is ModelState.Preparing -> Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer)) {
            Column(Modifier.padding(14.dp)) {
                Text(if (model.copying) "Setting up the model on this phone" else "Loading the model",
                    fontWeight = FontWeight.SemiBold, color = MaterialTheme.colorScheme.onPrimaryContainer)
                Text(if (model.copying) "Only on first launch; scans you add now are queued."
                     else "A few seconds; scans you add now are queued.",
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onPrimaryContainer)
                Spacer(Modifier.height(10.dp))
                if (model.copying) LinearProgressIndicator(progress = { model.progress }, modifier = Modifier.fillMaxWidth())
                else LinearProgressIndicator(Modifier.fillMaxWidth())
            }
        }
        is ModelState.Failed -> Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.errorContainer)) {
            Column(Modifier.padding(14.dp)) {
                Text("The model could not be loaded", fontWeight = FontWeight.SemiBold,
                    color = MaterialTheme.colorScheme.onErrorContainer)
                Text(model.message, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onErrorContainer)
            }
        }
        is ModelState.Ready -> Unit
    }
}

@Composable
private fun SummaryCard(records: List<Record>, site: String, onEditSite: () -> Unit) {
    val verified = records.filter { it.match != null }
    val agree = verified.count { it.match == true }
    Card(colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Column(Modifier.padding(14.dp)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(Icons.Default.Place, null, Modifier.size(18.dp), tint = MaterialTheme.colorScheme.primary)
                Spacer(Modifier.width(6.dp))
                Text(if (site.isBlank()) "No site set" else site, Modifier.weight(1f),
                    style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                TextButton(onClick = onEditSite) {
                    Icon(Icons.Default.Edit, null, Modifier.size(16.dp))
                    Spacer(Modifier.width(4.dp))
                    Text(if (site.isBlank()) "Set site" else "Change")
                }
            }
            Row(Modifier.fillMaxWidth().padding(top = 4.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                Stat("${records.size}", "trees", Modifier.weight(1f))
                Stat("${verified.size}", "verified", Modifier.weight(1f))
                Stat(if (verified.isEmpty()) "–" else pct(agree.toDouble() / verified.size, 0), "agree", Modifier.weight(1f))
            }
        }
    }
}

@Composable
private fun Stat(value: String, label: String, modifier: Modifier) {
    Column(modifier.clip(RoundedCornerShape(10.dp)).background(MaterialTheme.colorScheme.surfaceContainerHigh).padding(10.dp)) {
        Text(value, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
        Text(label, style = MaterialTheme.typography.labelMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun EmptyState(searching: Boolean) {
    Column(Modifier.fillMaxWidth().padding(vertical = 48.dp, horizontal = 24.dp), horizontalAlignment = Alignment.CenterHorizontally) {
        Text(if (searching) "No records match" else "No scans yet", style = MaterialTheme.typography.titleMedium)
        Spacer(Modifier.height(8.dp))
        Text(
            if (searching) "Try a file name, site, species, or a word from your notes."
            else "Tap Add scans and pick single-tree point clouds (.las, .laz, .ply, .xyz…), or share them to this " +
                "app from a file manager. Everything runs on this phone, with no internet connection.",
            style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = androidx.compose.ui.text.style.TextAlign.Center,
        )
    }
}

@Composable
private fun RecordRow(r: Record, stage: String?, thumb: File, onClick: () -> Unit) {
    Card(onClick = onClick, colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainer)) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            val bmp = if (r.status == Record.STATUS_DONE) rememberBitmap(thumb, r.finishedAt) else null
            Box(
                Modifier.size(56.dp).clip(RoundedCornerShape(10.dp)).background(Color.White)
                    .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(10.dp)),
                contentAlignment = Alignment.Center,
            ) {
                if (bmp != null) Image(bmp, null, Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                else Text(if (r.status == Record.STATUS_ERROR) "!" else "…", color = Color.Gray)
            }
            Spacer(Modifier.width(12.dp))
            Column(Modifier.weight(1f)) {
                Text(r.filename, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis)
                Text(listOf(formatDate(r.createdAt), r.site).filter { it.isNotBlank() }.joinToString(" · "),
                    style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant,
                    maxLines = 1, overflow = TextOverflow.Ellipsis)
                Spacer(Modifier.height(4.dp))
                when (r.status) {
                    Record.STATUS_DONE -> Row(verticalAlignment = Alignment.CenterVertically) {
                        SpeciesText(r.species, Modifier.weight(1f, fill = false), weight = FontWeight.SemiBold)
                        Spacer(Modifier.width(8.dp))
                        Meter((r.confidence ?: 0.0).toFloat(), Modifier.width(48.dp),
                            color = if ((r.confidence ?: 0.0) < 0.6) WaitColor else MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(6.dp))
                        Text(pct(r.confidence, 0), style = MaterialTheme.typography.labelMedium,
                            color = MaterialTheme.colorScheme.onSurfaceVariant)
                    }
                    Record.STATUS_ERROR -> Text("Failed: ${r.error}", style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.error, maxLines = 2, overflow = TextOverflow.Ellipsis)
                    else -> Column {
                        Text(stage ?: "Waiting in queue", style = MaterialTheme.typography.bodySmall, color = WaitColor)
                        Spacer(Modifier.height(4.dp))
                        LinearProgressIndicator(Modifier.fillMaxWidth().height(3.dp), color = WaitColor)
                    }
                }
            }
            r.match?.let { ok ->
                Spacer(Modifier.width(10.dp))
                Box(
                    Modifier.size(26.dp).clip(CircleShape)
                        .background(if (ok) MaterialTheme.colorScheme.primaryContainer else MaterialTheme.colorScheme.errorContainer),
                    contentAlignment = Alignment.Center,
                ) {
                    Text(if (ok) "✓" else "✗", color = if (ok) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
                        fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
fun AboutDialog(model: ModelState, onDismiss: () -> Unit) {
    val info = (model as? ModelState.Ready)?.info
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("About the model") },
        text = {
            Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Identifies 33 European and North American tree species from a single-tree LiDAR scan " +
                    "(FOR-species20K). The scan is rendered into seven views (four sides, top, bottom, trunk) " +
                    "and classified by three ConvNeXtV2-Base networks.")
                Text("This phone version runs one of the five cross-validation models at 256 px, compressed to " +
                    "8-bit weights. It works offline; scans never leave the device.")
                if (info?.heldOutOa != null) {
                    Text("Measured on ${info.heldOutTrees} trees it never saw in training: " +
                        "${pct(info.heldOutOa)} correct, macro-F1 ${"%.3f".format(info.heldOutMacroF1)}.",
                        fontWeight = FontWeight.SemiBold)
                }
                Text("The full desktop model (10 networks) reaches 85.0% on the official benchmark; " +
                    "use it for final numbers. Predictions outside the 33 species, or for partial scans, are unreliable.",
                    style = MaterialTheme.typography.bodySmall)
                Text("github.com/JaeLee18/FOR20K-lidar-tree-species · GPL-3.0", style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.primary)
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Close") } },
    )
}
