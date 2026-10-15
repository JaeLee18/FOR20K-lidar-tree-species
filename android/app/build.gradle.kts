plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "io.github.jaelee18.mvtree"
    compileSdk = 35

    defaultConfig {
        applicationId = "io.github.jaelee18.mvtree"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0"
        // Phones (arm64) and the x86_64 emulator; drops 32-bit ONNX Runtime libraries.
        ndk { abiFilters += listOf("arm64-v8a", "x86_64") }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Signed with the debug key so `assembleRelease` gives an installable APK;
            // use your own keystore for distribution.
            signingConfig = signingConfigs.getByName("debug")
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
        buildConfig = true
    }
    // The 290 MB model is copied out of the APK once; keep it stored, not deflated.
    androidResources {
        noCompress += "onnx"
    }
    // Default: model.onnx + model_info.json in src/main/assets (scripts/export/export_mobile.py).
    // Or build from another folder: ./gradlew assembleRelease -PmvtreeModelDir=/path/to/export
    providers.gradleProperty("mvtreeModelDir").orNull?.let { dir ->
        sourceSets.getByName("main").assets.setSrcDirs(listOf(dir))
    }
    packaging {
        resources {
            excludes += setOf("/META-INF/{AL2.0,LGPL2.1}", "META-INF/maven/**")
        }
    }
}

dependencies {
    implementation(project(":core"))
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.31.0")

    implementation(platform("androidx.compose:compose-bom:2024.12.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-core")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")
    implementation("androidx.core:core-ktx:1.15.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.9.0")
}
