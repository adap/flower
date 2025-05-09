plugins {
  alias(libs.plugins.android.library)
  alias(libs.plugins.kotlin.android)
  alias(libs.plugins.kotlin.serialization)
  alias(libs.plugins.ktfmt)
  alias(libs.plugins.maven.publish)
  alias(libs.plugins.signing)
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

  publishing {
    singleVariant("release") {
      withSourcesJar()
      withJavadocJar()
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

afterEvaluate {
  publishing {
    publications {
      create<MavenPublication>("release") {
        from(components["release"])

        groupId = project.property("GROUP_ID") as String
        artifactId = project.property("ARTIFACT_ID") as String
        version = project.property("VERSION_NAME") as String

        pom {
          name.set("Flower Intelligence")
          description.set("Open-Source On-Device AI with optional Confidential Remote Compute")
          url.set("https://github.com/adap/flower/")

          licenses {
            license {
              name.set("The Apache License, Version 2.0")
              url.set("http://www.apache.org/licenses/LICENSE-2.0.txt")
            }
          }
          developers {
            developer {
              name.set("The Flower Authors")
              url.set("flower.ai")
            }
          }
          scm {
            connection.set("scm:git:git://github.com/adap/flower.git")
            developerConnection.set("scm:git:ssh://git@github.com/adap/flower.git")
            url.set("https://github.com/adap/flower/")
          }
        }
      }
    }
  }

  signing {
    val signingKey: String? by project
    val signingPassword: String? by project
    useInMemoryPgpKeys(signingKey, signingPassword)
    sign(publishing.publications["release"])
  }
}
