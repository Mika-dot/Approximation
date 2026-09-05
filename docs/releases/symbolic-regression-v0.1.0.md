# Symbolic Regression v0.1.0

This release packages the `feature/universal-symbolic-regression` implementation for direct Windows use and .NET integration.

## Assets

- `Approximation.exe` — self-contained Windows x64 demo and CSV CLI. The .NET runtime is bundled.
- `ApproximationLib.dll` — .NET 8 class library for embedding symbolic regression into another application.
- `ApproximationLib.xml` — XML API documentation.
- `SHA256SUMS.txt` — SHA-256 checksums for the executable and DLL.

## Release validation

The release pipeline:

1. restores the solution;
2. builds in `Release` with warnings treated as errors;
3. runs the repository test executable;
4. publishes a single-file self-contained Windows x64 executable;
5. smoke-tests the published executable with `--help`;
6. calculates SHA-256 checksums;
7. publishes the assets to the GitHub Release `symbolic-regression-v0.1.0`.
