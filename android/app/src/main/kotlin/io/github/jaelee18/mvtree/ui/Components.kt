package io.github.jaelee18.mvtree.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.background
import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.produceState
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.text.DateFormat
import java.util.Date
import java.util.Locale

private val Light = lightColorScheme(
    primary = Color(0xFF2F6B4F), onPrimary = Color.White,
    primaryContainer = Color(0xFFD5EBDD), onPrimaryContainer = Color(0xFF0B2A1B),
    secondary = Color(0xFF52634F), onSecondary = Color.White,
    secondaryContainer = Color(0xFFE3EBE2), onSecondaryContainer = Color(0xFF1D2A1F),
    tertiary = Color(0xFF8A5A2B), onTertiary = Color.White,
    background = Color(0xFFF4F5F1), onBackground = Color(0xFF1D231F),
    surface = Color(0xFFF4F5F1), onSurface = Color(0xFF1D231F),
    surfaceVariant = Color(0xFFE1E4DC), onSurfaceVariant = Color(0xFF5F6861),
    surfaceContainerLowest = Color.White, surfaceContainerLow = Color(0xFFFAFBF8),
    surfaceContainer = Color(0xFFFFFFFF), surfaceContainerHigh = Color(0xFFEFF1EC),
    surfaceContainerHighest = Color(0xFFE8EBE5),
    outline = Color(0xFFB8BDB5), outlineVariant = Color(0xFFE1E4DC),
    error = Color(0xFFB3261E), onError = Color.White, errorContainer = Color(0xFFFBE5E3),
    onErrorContainer = Color(0xFF5C1410),
)

private val Dark = darkColorScheme(
    primary = Color(0xFF6FBF8F), onPrimary = Color(0xFF0D1A12),
    primaryContainer = Color(0xFF1E3227), onPrimaryContainer = Color(0xFFCDEBD9),
    secondary = Color(0xFFB6CCB5), onSecondary = Color(0xFF1D2A1F),
    secondaryContainer = Color(0xFF2A352B), onSecondaryContainer = Color(0xFFD5E3D3),
    tertiary = Color(0xFFE0B98A), onTertiary = Color(0xFF2E1B07),
    background = Color(0xFF0F1311), onBackground = Color(0xFFE6EBE7),
    surface = Color(0xFF0F1311), onSurface = Color(0xFFE6EBE7),
    surfaceVariant = Color(0xFF2A322D), onSurfaceVariant = Color(0xFF9AA59E),
    surfaceContainerLowest = Color(0xFF0B0E0C), surfaceContainerLow = Color(0xFF141916),
    surfaceContainer = Color(0xFF171C19), surfaceContainerHigh = Color(0xFF1E2420),
    surfaceContainerHighest = Color(0xFF252C27),
    outline = Color(0xFF59625C), outlineVariant = Color(0xFF2A322D),
    error = Color(0xFFEF8A80), onError = Color(0xFF3A0A06), errorContainer = Color(0xFF3A1C1A),
    onErrorContainer = Color(0xFFFBD9D5),
)

/** Amber for "working on it" states, which Material's scheme has no role for. */
val WaitColor @Composable get() = if (isSystemInDarkTheme()) Color(0xFFE0A64A) else Color(0xFFA8670A)

@Composable
fun MvtreeTheme(content: @Composable () -> Unit) {
    MaterialTheme(colorScheme = if (isSystemInDarkTheme()) Dark else Light, content = content)
}

/** "Picea_abies" -> italic "Picea abies", as scientific names are written. */
@Composable
fun SpeciesText(
    species: String?,
    modifier: Modifier = Modifier,
    style: TextStyle = MaterialTheme.typography.bodyMedium,
    weight: FontWeight? = null,
    color: Color = Color.Unspecified,
) {
    Text(
        species?.replace('_', ' ') ?: "–", modifier = modifier, style = style, fontStyle = FontStyle.Italic,
        fontWeight = weight, color = color, maxLines = 1, overflow = TextOverflow.Ellipsis,
    )
}

fun pct(p: Number?, digits: Int = 1): String {
    val v = p?.toDouble() ?: return "–"
    val floor = Math.pow(10.0, -digits.toDouble())
    return if (v > 0 && v * 100 < floor / 2) "<${"%.${digits}f".format(Locale.ROOT, floor)}%"
    else "%.${digits}f%%".format(Locale.ROOT, v * 100)
}

fun formatDate(ms: Long?): String =
    ms?.let { DateFormat.getDateTimeInstance(DateFormat.MEDIUM, DateFormat.SHORT).format(Date(it)) } ?: "–"

fun formatBytes(n: Long): String = when {
    n >= 1 shl 30 -> "%.1f GB".format(n / (1 shl 30).toDouble())
    n >= 1 shl 20 -> "%.1f MB".format(n / (1 shl 20).toDouble())
    n >= 1 shl 10 -> "%.0f KB".format(n / (1 shl 10).toDouble())
    else -> "$n B"
}

/** A thin horizontal probability bar. */
@Composable
fun Meter(fraction: Float, modifier: Modifier = Modifier, color: Color = MaterialTheme.colorScheme.primary) {
    Box(
        modifier.height(6.dp).clip(RoundedCornerShape(3.dp)).background(MaterialTheme.colorScheme.surfaceVariant)
    ) {
        Box(Modifier.fillMaxHeight().fillMaxWidth(fraction.coerceIn(0f, 1f)).background(color))
    }
}

/** Species, bar, and percentage on one line. */
@Composable
fun ProbabilityRow(species: String, p: Float, strong: Boolean) {
    Row(Modifier.fillMaxWidth().padding(vertical = 3.dp), verticalAlignment = Alignment.CenterVertically) {
        SpeciesText(species, Modifier.width(150.dp), weight = if (strong) FontWeight.SemiBold else null)
        Meter(
            maxOf(p, 0.005f), Modifier.weight(1f),
            color = if (strong) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.primary.copy(alpha = 0.45f),
        )
        Spacer(Modifier.width(10.dp))
        Text(pct(p), Modifier.width(52.dp), style = MaterialTheme.typography.bodySmall,
            color = MaterialTheme.colorScheme.onSurfaceVariant, textAlign = androidx.compose.ui.text.style.TextAlign.End)
    }
}

/** Decode an image file off the main thread; [key] changes when the file is rewritten. */
@Composable
fun rememberBitmap(file: File, key: Any?): ImageBitmap? {
    val bitmap by produceState<ImageBitmap?>(null, file.path, key) {
        value = withContext(Dispatchers.IO) {
            if (file.isFile) BitmapFactory.decodeFile(file.path)?.asImageBitmap() else null
        }
    }
    return bitmap
}
