package io.github.jaelee18.mvtree.ui

import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.BackHandler
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.SnackbarHostState
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.lifecycle.lifecycleScope
import io.github.jaelee18.mvtree.mvtree
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.receiveAsFlow
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    // Buffered until the UI collects, so messages from a share that cold-starts the app are not lost.
    private val messages = Channel<String>(Channel.BUFFERED)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        if (savedInstanceState == null) handleIntent(intent)
        setContent {
            MvtreeTheme {
                val snackbar = remember { SnackbarHostState() }
                LaunchedEffect(Unit) { messages.receiveAsFlow().collect { snackbar.showSnackbar(it) } }
                var openId by rememberSaveable { mutableStateOf<Long?>(null) }
                val id = openId
                if (id == null) {
                    RecordsScreen(onOpen = { openId = it }, snackbar = snackbar)
                } else {
                    BackHandler { openId = null }
                    RecordScreen(id, onBack = { openId = null }, snackbar = snackbar)
                }
            }
        }
    }

    override fun onNewIntent(intent: Intent) {
        super.onNewIntent(intent)
        handleIntent(intent)
    }

    /** Scans shared to the app ("Share" or "Open with") are queued like picked files. */
    private fun handleIntent(intent: Intent?) {
        val uris: List<Uri> = when (intent?.action) {
            Intent.ACTION_SEND -> listOfNotNull(intent.streamExtra())
            Intent.ACTION_SEND_MULTIPLE -> intent.streamExtras()
            Intent.ACTION_VIEW -> listOfNotNull(intent.data)
            else -> emptyList()
        }
        if (uris.isEmpty()) return
        val site = getSharedPreferences("mvtree", MODE_PRIVATE).getString("site", "") ?: ""
        lifecycleScope.launch {
            var added = 0
            for (uri in uris) {
                val error = mvtree.analyzer.add(uri, site)
                if (error == null) added++ else messages.send(error)
            }
            if (added > 0) messages.send("Added $added scan${if (added == 1) "" else "s"}")
        }
    }

    @Suppress("DEPRECATION")
    private fun Intent.streamExtra(): Uri? =
        if (Build.VERSION.SDK_INT >= 33) getParcelableExtra(Intent.EXTRA_STREAM, Uri::class.java)
        else getParcelableExtra(Intent.EXTRA_STREAM)

    @Suppress("DEPRECATION")
    private fun Intent.streamExtras(): List<Uri> =
        (if (Build.VERSION.SDK_INT >= 33) getParcelableArrayListExtra(Intent.EXTRA_STREAM, Uri::class.java)
         else getParcelableArrayListExtra(Intent.EXTRA_STREAM)) ?: emptyList()
}
