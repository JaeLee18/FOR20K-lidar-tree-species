package io.github.jaelee18.mvtree

import android.annotation.SuppressLint
import android.content.Context
import android.location.Location
import android.location.LocationListener
import android.location.LocationManager
import android.os.Build
import android.os.CancellationSignal
import android.os.Looper
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume

/** One position fix from the platform LocationManager (no Google Play services needed). */
object PhoneLocation {
    @SuppressLint("MissingPermission")  // callers request the permission first
    suspend fun current(context: Context, timeoutMs: Long = 30_000): Location? {
        val lm = context.getSystemService(Context.LOCATION_SERVICE) as LocationManager
        val provider = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
            .firstOrNull { lm.isProviderEnabled(it) } ?: return null
        val fix = withTimeoutOrNull(timeoutMs) {
            suspendCancellableCoroutine { cont ->
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
                    val signal = CancellationSignal()
                    cont.invokeOnCancellation { signal.cancel() }
                    lm.getCurrentLocation(provider, signal, context.mainExecutor) { cont.resume(it) }
                } else {
                    val listener = object : LocationListener {
                        override fun onLocationChanged(location: Location) {
                            lm.removeUpdates(this)
                            if (cont.isActive) cont.resume(location)
                        }
                        @Deprecated("Deprecated in Java")
                        override fun onStatusChanged(p: String?, s: Int, e: android.os.Bundle?) = Unit
                        override fun onProviderEnabled(p: String) = Unit
                        override fun onProviderDisabled(p: String) = Unit
                    }
                    cont.invokeOnCancellation { lm.removeUpdates(listener) }
                    lm.requestLocationUpdates(provider, 0L, 0f, listener, Looper.getMainLooper())
                }
            }
        }
        return fix ?: lm.getLastKnownLocation(provider)
    }
}
