plugins {
  alias(libs.plugins.android.application)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.serialization)
  alias(libs.plugins.compose.compiler)
  alias(libs.plugins.ktfmt)
}

android {
  namespace = "ai.flower.intelligence.examples.helloworld"
  compileSdk = 35

  defaultConfig {
    applicationId = "ai.flower.intelligence.examples.helloworld"
    minSdk = 28
    targetSdk = 35
    versionCode = 1
    versionName = "1.0"

    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"

    val apiKey: String = project.findProperty("API_KEY") as? String ?: ""
    buildConfigField("String", "API_KEY", "\"$apiKey\"")
  }

  buildTypes {
    release {
      isMinifyEnabled = false
      proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
    }
  }

  compileOptions {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
  }

  kotlinOptions { jvmTarget = "11" }

  buildFeatures {
    compose = true
    buildConfig = true
  }
}

dependencies {
  implementation(project(":flwr"))

  // AndroidX
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.appcompat)

  // Compose
  implementation(platform(libs.compose.bom))
  implementation(libs.compose.ui)
  implementation(libs.compose.ui.graphics)
  implementation(libs.compose.ui.tooling.preview)
  implementation(libs.compose.material3)
  implementation(libs.activity.compose)
  implementation(libs.navigation.compose)
  implementation(libs.lifecycle.runtime.ktx)
}

ktfmt { googleStyle() }
