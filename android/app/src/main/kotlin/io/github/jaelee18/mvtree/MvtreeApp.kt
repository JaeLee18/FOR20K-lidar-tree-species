package io.github.jaelee18.mvtree

import android.app.Application
import android.content.Context

class MvtreeApp : Application() {
    lateinit var store: RecordStore
        private set
    lateinit var analyzer: Analyzer
        private set

    override fun onCreate() {
        super.onCreate()
        store = RecordStore(this)
        analyzer = Analyzer(this, store).also { it.start() }
    }
}

val Context.mvtree: MvtreeApp get() = applicationContext as MvtreeApp
