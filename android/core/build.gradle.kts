// Pure Kotlin/JVM: point-cloud readers, the 7-view renderer, model input, and the
// ONNX classifier. No Android dependencies, so it is unit-tested on the host
// against golden outputs from the Python package (see tools/make_golden.py).
plugins {
    id("org.jetbrains.kotlin.jvm")
}

java {
    sourceCompatibility = JavaVersion.VERSION_17
    targetCompatibility = JavaVersion.VERSION_17
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    implementation("com.github.mreutegg:laszip4j:0.21")
    // Provided at runtime by onnxruntime-android (app) or onnxruntime (host tests): same Java API.
    compileOnly("com.microsoft.onnxruntime:onnxruntime:1.31.0")
    // org.json ships with Android; tests need a copy on the host.
    compileOnly("org.json:json:20240303")

    testImplementation("junit:junit:4.13.2")
    testImplementation("com.microsoft.onnxruntime:onnxruntime:1.31.0")
    testImplementation("org.json:json:20240303")
}

tasks.test {
    maxHeapSize = "6g"
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
        exceptionFormat = org.gradle.api.tasks.testing.logging.TestExceptionFormat.FULL
    }
}
