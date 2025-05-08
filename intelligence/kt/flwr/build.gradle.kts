import org.jetbrains.dokka.gradle.DokkaTask

plugins {
  alias(libs.plugins.android.library)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.serialization)
  alias(libs.plugins.ktfmt)
  alias(libs.plugins.dokka)
}

android {
  namespace = "ai.flower.intelligence"
  compileSdk = 35

  defaultConfig {
    minSdk = 28
    testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
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

  testOptions { unitTests.isIncludeAndroidResources = true }

  packaging {
    resources {
      excludes += "/META-INF/LICENSE.md"
      excludes += "/META-INF/LICENSE-notice.md"
    }
  }
}

dependencies {
  // Main
  implementation(libs.androidx.core.ktx)
  implementation(libs.androidx.appcompat)
  implementation(libs.kotlinx.serialization.json)
  implementation(libs.ktor.client.core)
  implementation(libs.ktor.client.cio)
  implementation(libs.ktor.client.content.negotiation)
  implementation(libs.ktor.serialization.kotlinx.json)

  // Unit Testing (JUnit 5 + MockK)
  testImplementation(libs.junit.jupiter.api)
  testRuntimeOnly(libs.junit.jupiter.engine)
  testImplementation(libs.mockk)

  // Android Instrumentation Tests
  androidTestImplementation(libs.androidx.junit)
  androidTestImplementation(libs.androidx.core)
  androidTestImplementation(libs.mockk.android)
  androidTestImplementation(libs.kotlinx.coroutines.test)
  androidTestImplementation(libs.androidx.espresso.core)
}

ktfmt { googleStyle() }

tasks.withType<Test>().configureEach { useJUnitPlatform() }

tasks.withType<DokkaTask>().configureEach {
  moduleName.set(project.name)
  moduleVersion.set(project.version.toString())
  outputDirectory.set(layout.buildDirectory.dir("dokka/$name"))
  failOnWarning.set(false)
  suppressObviousFunctions.set(true)
  suppressInheritedMembers.set(false)
  offlineMode.set(false)

  dokkaSourceSets {
    configureEach {
      if (name == "test" || name == "androidTest") {
        suppress.set(true)
      }
      perPackageOption {
        matchingRegex.set(".*FlowerIntelligence(Android)?Test.*")
        suppress.set(true)
      }
      noStdlibLink.set(true)
      noJdkLink.set(true)
      noAndroidSdkLink.set(true)
    }
  }
}
