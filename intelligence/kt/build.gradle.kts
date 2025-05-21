// Top-level build file where you can add configuration options common to all sub-projects/modules.
plugins {
  alias(libs.plugins.android.library) apply false
  alias(libs.plugins.kotlin.android) apply false
  alias(libs.plugins.kotlin.serialization) apply false
  alias(libs.plugins.ktfmt) apply false
  alias(libs.plugins.dokka) apply false
  alias(libs.plugins.android.application) apply false
}
