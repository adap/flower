# AGENTS.md

Guidance for working in this repository.

## Scope

This is a monorepo spanning the Python Flower framework, research baselines, examples, datasets, and multi-language SDKs. Always confirm which subproject you are touching before installing dependencies or running tests.

## Repo Map

- `framework/`: Python Flower framework, docs, e2e, proto, and SDKs.
- `framework/py/`: Python packages `flwr` and `flwr_tool`.
- `framework/docs/`: Sphinx docs for the framework.
- `framework/e2e/`: End-to-end test apps (often Docker or external deps).
- `framework/proto/`: protobuf definitions and generated stubs.
- `framework/devtool/`: Formatting, docs, and CI helper scripts.
- `framework/kotlin/`, `framework/swift/`: framework mobile SDKs.
- `intelligence/`: Flower Intelligence SDKs (Swift, Kotlin, TypeScript) and docs.
- `baselines/`: Research baselines (each subdir is its own Python project).
- `examples/`: Runnable examples (each subdir has its own Python deps).
- `benchmarks/`: Benchmark harnesses (often separate deps).
- `datasets/`: Dataset tooling and e2e dataset projects.

## Python Framework (framework/)

- Tooling: Poetry via `framework/pyproject.toml` is the source of truth.
- Install: `cd framework` then `poetry install --all-extras`, or run `./devtool/bootstrap.sh` for a clean setup.
- Tests: `cd framework` then `poetry run pytest` (defaults to `py/flwr` and `py/flwr_tool`).
- Formatting/checks for examples and benchmarks: `framework/devtool/format.sh` and `framework/devtool/test.sh`.

## Examples/Baselines/Datasets

- Each subproject usually has its own `pyproject.toml` or `requirements.txt`. Install and run commands from that subdirectory.
- Many examples download data or require GPUs. Confirm before running large workloads.

## Swift (Flower Intelligence + Framework SDK)

- SwiftPM entry point: `Package.swift` (library target `FlowerIntelligence`).
- Sources: `intelligence/swift/src`. Tests: `intelligence/swift/tests`. Examples: `intelligence/swift/examples`.
- Build/test: `swift build` and `swift test` from repo root or `intelligence/swift`.
- Xcode: `project.yml` is for XcodeGen. `FlowerIntelligenceExamples.xcodeproj` is pre-generated.
- Framework Swift SDK lives in `framework/swift/`.

## TypeScript (Flower Intelligence)

- Location: `intelligence/ts`.
- Tooling: `pnpm`.
- Common commands: `pnpm install`, `pnpm test`, `pnpm build`, `pnpm lint`, `pnpm format`.

## Kotlin

- Flower Intelligence Kotlin SDK: `intelligence/kt` (Gradle wrapper available).
- Framework Kotlin SDK: `framework/kotlin`.
- Common commands: `./gradlew build`, `./gradlew test` from the respective directory.

## Architecture Notes

- Platform/public/private endpoints and long-running processes are listed in `ARCHITECTURE.md`.

## Contrib Expectations

- Follow the formatting and lint tooling defined in each subproject.
- If you introduce new dependencies or scripts, update the relevant `pyproject.toml` or `package.json` in that subproject.





## TODO

The `flower` repo is organized into several independent subprojects, each with its own `README.md` file.

When writing documentation, please follow these guidelines:
- Read related documentation pages and ensure consistency in style and terminology.
- Use clear and concise language.
- Don't use emojis (exception: documentation and user-facing messages).

Python:
- Use NumPy style for docstrings.
- Follow PEP 8 guidelines for code style.