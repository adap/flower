buildscript {
    repositories {
        google()
        mavenCentral()
    }
    dependencies {
        // Not 8.0.2 because on macOS `Could not find protoc-3.11.0-osx-aarch_64.exe (com.google.protobuf:protoc:3.11.0)`
        classpath("com.android.tools.build:gradle:7.4.2")
        // Not 0.9.2 because of https://github.com/grpc/grpc-kotlin/issues/380
        classpath("com.google.protobuf:protobuf-gradle-plugin:0.9.1")
        classpath("org.jetbrains.kotlin:kotlin-gradle-plugin:1.9.0")

        classpath("com.vanniktech:gradle-maven-publish-plugin:0.25.3") // NEW

        // NOTE: Do not place your application dependencies here; they belong
        // in the individual module build.gradle files
    }
}
