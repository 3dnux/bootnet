import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val rustDir = rootProject.file("rust")
val generatedJniLibs = layout.buildDirectory.dir("generated/jniLibs")
val generatedAssets = layout.buildDirectory.dir("generated/shieldAssets")
// Scriptlets de uBO + recursos de Brave, generados por `npm install` en youtube-shield/.
val resourcesJson = rootProject.file("../data/resources.json")

android {
    namespace = "dev.bootnet.shieldtube"
    compileSdk = 36

    defaultConfig {
        applicationId = "dev.bootnet.shieldtube"
        minSdk = 26
        targetSdk = 36
        versionCode = 10
        versionName = "0.7.0"
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            // Firmado con la clave de depuración para poder instalar el APK directamente.
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets["main"].jniLibs.srcDir(generatedJniLibs)
    sourceSets["main"].assets.srcDir(generatedAssets)
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.webkit:webkit:1.14.0")
    // Seguimiento de cabeza con la cámara frontal (detección facial en el móvil).
    // ML Kit vía Google Play Services: el modelo lo descarga Play Services y el
    // APK no crece ~30 MB como con el modelo incluido.
    implementation("androidx.camera:camera-camera2:1.5.1")
    implementation("androidx.camera:camera-lifecycle:1.5.1")
    implementation("androidx.lifecycle:lifecycle-runtime:2.8.7") // CameraX arrastra la 2.0.0
    implementation("com.google.android.gms:play-services-mlkit-face-detection:17.1.0")
    testImplementation("junit:junit:4.13.2")
}

// Compila el motor de Brave (Rust) para cada ABI.
val buildRust by tasks.registering(Exec::class) {
    inputs.dir(rustDir.resolve("src"))
    inputs.files(rustDir.resolve("Cargo.toml"), rustDir.resolve("Cargo.lock"), rustDir.resolve("build-android.sh"))
    outputs.dir(generatedJniLibs)
    workingDir = rustDir
    environment("ANDROID_HOME", android.sdkDirectory.absolutePath)
    commandLine("bash", "build-android.sh", generatedJniLibs.get().asFile.absolutePath)
}

val copyShieldResources by tasks.registering(Copy::class) {
    doFirst {
        check(resourcesJson.exists()) {
            "Falta ${resourcesJson.path}: ejecuta `npm install` en youtube-shield/ antes de compilar el APK"
        }
    }
    from(resourcesJson)
    into(generatedAssets)
}

tasks.named("preBuild") {
    dependsOn(buildRust, copyShieldResources)
}
