using System.Globalization;

namespace ApproximationLib;

/// <summary>
/// Configuration for the hybrid symbolic-basis + Moore-Penrose least-squares regressor.
/// </summary>
public sealed class HybridConfig
{
    /// <summary>Number of independently discovered symbolic basis functions.</summary>
    public int BasisModels { get; set; } = 6;

    /// <summary>
    /// Fraction of observations reserved for fitting only the final linear combination.
    /// Base symbolic models never see these target values during structure discovery.
    /// </summary>
    public double BlendFraction { get; set; } = 0.25;

    /// <summary>Seed controlling the structure/blend split and derived basis-model seeds.</summary>
    public int Seed { get; set; } = 2026;

    /// <summary>Configuration copied for every symbolic basis search.</summary>
    public Config Symbolic { get; set; } = new();

    internal HybridConfig Copy() => new()
    {
        BasisModels = BasisModels,
        BlendFraction = BlendFraction,
        Seed = Seed,
        Symbolic = Symbolic?.Copy() ?? throw new ArgumentNullException(nameof(Symbolic))
    };

    internal void Validate()
    {
        if (BasisModels < 1 || BasisModels > 64)
            throw new ArgumentOutOfRangeException(nameof(BasisModels), "BasisModels must be in [1, 64].");
        if (BlendFraction <= 0 || BlendFraction >= 0.5)
            throw new ArgumentOutOfRangeException(nameof(BlendFraction), "BlendFraction must be in (0, 0.5).");
        ArgumentNullException.ThrowIfNull(Symbolic);
        Symbolic.Validate();
    }
}

/// <summary>Fitted hybrid model and diagnostics.</summary>
public sealed class HybridResult
{
    public required string Formula { get; init; }
    public required IReadOnlyList<string> BasisFormulas { get; init; }
    public required IReadOnlyList<double> Coefficients { get; init; }
    public double Bias { get; init; }
    public int NumericalRank { get; init; }
    public int StructureSamples { get; init; }
    public int BlendSamples { get; init; }
    public required ModelMetrics Metrics { get; init; }
    public required ModelMetrics BlendMetrics { get; init; }
    public required Func<double[], double> MultiFunction { get; init; }

    public double Predict(params double[] features) => MultiFunction(features);
    public double[] Predict(IEnumerable<double[]> rows) => rows.Select(MultiFunction).ToArray();
}

/// <summary>
/// Discovers several interpretable nonlinear symbolic basis functions and combines them with
/// one rank-aware Moore-Penrose least-squares solve. The final predictor remains an explicit
/// weighted sum of symbolic formulas.
/// </summary>
public sealed class HybridSymbolicLsqRegressor
{
    private const int SeedStride = 104729;
    private readonly HybridConfig _config;

    public HybridSymbolicLsqRegressor(HybridConfig? config = null)
    {
        _config = (config ?? new HybridConfig()).Copy();
        _config.Validate();
    }

    public HybridResult Fit(double[] xData, double[] yData)
    {
        ArgumentNullException.ThrowIfNull(xData);
        return Fit(xData.Select(x => new[] { x }).ToArray(), yData, new[] { "x" });
    }

    public HybridResult Fit(double[][] features, double[] targets, string[]? featureNames = null,
        CancellationToken cancellationToken = default)
    {
        ValidateData(features, targets, featureNames);
        if (features.Length < 8)
            throw new ArgumentException("Hybrid fitting requires at least eight observations.", nameof(features));

        string[] names = featureNames is null
            ? Enumerable.Range(0, features[0].Length).Select(i => i == 0 ? "x" : $"x{i}").ToArray()
            : (string[])featureNames.Clone();

        (int[] structure, int[] blend) = SplitIndices(features.Length);
        double[][] structureX = structure.Select(i => (double[])features[i].Clone()).ToArray();
        double[] structureY = structure.Select(i => targets[i]).ToArray();

        var basis = new Result[_config.BasisModels];
        for (int modelIndex = 0; modelIndex < basis.Length; modelIndex++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Config symbolicConfig = _config.Symbolic.Copy();
            symbolicConfig.Seed = unchecked(_config.Seed + SeedStride * (modelIndex + 1));
            symbolicConfig.GenerateReport = false;
            symbolicConfig.EnableLogging = false;
            basis[modelIndex] = new SymbolicRegressor(symbolicConfig)
                .Fit(structureX, structureY, names, cancellationToken);
        }

        double[,] phi = new double[blend.Length, basis.Length];
        double[] blendY = new double[blend.Length];
        for (int row = 0; row < blend.Length; row++)
        {
            cancellationToken.ThrowIfCancellationRequested();
            double[] input = features[blend[row]];
            blendY[row] = targets[blend[row]];
            for (int column = 0; column < basis.Length; column++)
            {
                double value = basis[column].MultiFunction(input);
                if (!double.IsFinite(value))
                    throw new InvalidOperationException($"Symbolic basis {column} produced a non-finite value.");
                phi[row, column] = value;
            }
        }

        LinearBlend blendModel = MoorePenroseBlend.Fit(phi, blendY);
        double[] coefficients = blendModel.Coefficients;
        double bias = blendModel.Bias;
        Result[] basisSnapshot = basis.ToArray();
        double[] coefficientSnapshot = (double[])coefficients.Clone();

        double PredictRow(double[] row)
        {
            if (row is null || row.Length != features[0].Length)
                throw new ArgumentException($"Expected {features[0].Length} features.", nameof(row));
            if (row.Any(v => !double.IsFinite(v)))
                throw new ArgumentException("Prediction input contains NaN or infinity.", nameof(row));

            double prediction = bias;
            for (int i = 0; i < basisSnapshot.Length; i++)
                prediction += coefficientSnapshot[i] * basisSnapshot[i].MultiFunction(row);
            return prediction;
        }

        double[] allPredictions = features.Select(row => PredictRow(row)).ToArray();
        double[] blendPredictions = blend.Select(i => PredictRow(features[i])).ToArray();
        string[] formulas = basis.Select(item => item.BestFormula).ToArray();

        return new HybridResult
        {
            Formula = BuildFormula(bias, coefficients, formulas),
            BasisFormulas = Array.AsReadOnly(formulas),
            Coefficients = Array.AsReadOnly((double[])coefficients.Clone()),
            Bias = bias,
            NumericalRank = blendModel.NumericalRank,
            StructureSamples = structure.Length,
            BlendSamples = blend.Length,
            Metrics = ModelMetrics.Calculate(targets, allPredictions),
            BlendMetrics = ModelMetrics.Calculate(blendY, blendPredictions),
            MultiFunction = PredictRow
        };
    }

    private (int[] Structure, int[] Blend) SplitIndices(int count)
    {
        int[] indices = Enumerable.Range(0, count).ToArray();
        var rng = new Random(_config.Seed);
        for (int i = indices.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        int blendCount = (int)Math.Round(count * _config.BlendFraction);
        blendCount = Math.Clamp(blendCount, 3, count - 3);
        return (indices.Skip(blendCount).ToArray(), indices.Take(blendCount).ToArray());
    }

    private static string BuildFormula(double bias, IReadOnlyList<double> coefficients, IReadOnlyList<string> formulas)
    {
        var terms = new List<string> { Format(bias) };
        for (int i = 0; i < coefficients.Count; i++)
        {
            if (Math.Abs(coefficients[i]) <= 1e-14) continue;
            terms.Add($"({Format(coefficients[i])}) * ({formulas[i]})");
        }
        return string.Join(" + ", terms);
    }

    private static string Format(double value) => value.ToString("G17", CultureInfo.InvariantCulture);

    private static void ValidateData(double[][] features, double[] targets, string[]? names)
    {
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(targets);
        if (features.Length != targets.Length)
            throw new ArgumentException("Features and targets must have the same number of rows.");
        if (features.Length == 0 || features[0] is null || features[0].Length == 0)
            throw new ArgumentException("At least one feature is required.", nameof(features));

        int width = features[0].Length;
        for (int row = 0; row < features.Length; row++)
        {
            if (features[row] is null || features[row].Length != width)
                throw new ArgumentException("All feature rows must have the same length.", nameof(features));
            if (features[row].Any(v => !double.IsFinite(v)))
                throw new ArgumentException($"Feature row {row} contains NaN or infinity.", nameof(features));
            if (!double.IsFinite(targets[row]))
                throw new ArgumentException($"Target {row} contains NaN or infinity.", nameof(targets));
        }

        if (names is not null)
        {
            if (names.Length != width)
                throw new ArgumentException("Feature name count must match the feature count.", nameof(names));
            if (names.Any(string.IsNullOrWhiteSpace) || names.Distinct(StringComparer.Ordinal).Count() != names.Length)
                throw new ArgumentException("Feature names must be non-empty and unique.", nameof(names));
        }
    }

    private sealed record LinearBlend(double Bias, double[] Coefficients, int NumericalRank);

    private static class MoorePenroseBlend
    {
        private const double MachineEpsilon = 2.2204460492503131e-16;

        public static LinearBlend Fit(double[,] features, double[] targets)
        {
            int rows = features.GetLength(0);
            int columns = features.GetLength(1);
            if (targets.Length != rows || rows == 0 || columns == 0)
                throw new ArgumentException("LSQ design matrix dimensions are invalid.");

            var means = new double[columns];
            var scales = new double[columns];
            for (int column = 0; column < columns; column++)
            {
                double mean = 0;
                double m2 = 0;
                for (int row = 0; row < rows; row++)
                {
                    double value = features[row, column];
                    double delta = value - mean;
                    mean += delta / (row + 1);
                    m2 += delta * (value - mean);
                }
                means[column] = mean;
                double variance = rows > 1 ? m2 / rows : 0;
                scales[column] = variance > 1e-24 ? Math.Sqrt(variance) : 1.0;
            }

            double targetMean = targets.Average();
            var gram = new double[columns, columns];
            var cross = new double[columns];
            for (int row = 0; row < rows; row++)
            {
                var standardized = new double[columns];
                for (int column = 0; column < columns; column++)
                    standardized[column] = (features[row, column] - means[column]) / scales[column];

                double centeredTarget = targets[row] - targetMean;
                for (int left = 0; left < columns; left++)
                {
                    cross[left] += standardized[left] * centeredTarget;
                    for (int right = left; right < columns; right++)
                        gram[left, right] += standardized[left] * standardized[right];
                }
            }
            for (int left = 0; left < columns; left++)
                for (int right = 0; right < left; right++)
                    gram[left, right] = gram[right, left];

            (double[] eigenvalues, double[,] eigenvectors) = SymmetricEigen.Decompose(gram);
            double maxEigenvalue = eigenvalues.Length == 0 ? 0 : eigenvalues.Max();
            double tolerance = MachineEpsilon * Math.Max(rows, columns) * Math.Max(maxEigenvalue, 1.0);
            var standardizedWeights = new double[columns];
            int rank = 0;

            for (int component = 0; component < columns; component++)
            {
                double eigenvalue = eigenvalues[component];
                if (!(eigenvalue > tolerance)) continue;
                rank++;
                double projection = 0;
                for (int row = 0; row < columns; row++)
                    projection += eigenvectors[row, component] * cross[row];
                double factor = projection / eigenvalue;
                for (int row = 0; row < columns; row++)
                    standardizedWeights[row] += eigenvectors[row, component] * factor;
            }

            var coefficients = new double[columns];
            for (int column = 0; column < columns; column++)
                coefficients[column] = standardizedWeights[column] / scales[column];

            double bias = targetMean;
            for (int column = 0; column < columns; column++)
                bias -= coefficients[column] * means[column];

            return new LinearBlend(bias, coefficients, rank);
        }
    }

    private static class SymmetricEigen
    {
        public static (double[] Eigenvalues, double[,] Eigenvectors) Decompose(double[,] source)
        {
            int n = source.GetLength(0);
            if (n != source.GetLength(1)) throw new ArgumentException("Matrix must be square.", nameof(source));

            var matrix = (double[,])source.Clone();
            var vectors = new double[n, n];
            for (int i = 0; i < n; i++) vectors[i, i] = 1.0;

            int maxIterations = Math.Max(32, 50 * n * n);
            for (int iteration = 0; iteration < maxIterations; iteration++)
            {
                int p = 0, q = 0;
                double largest = 0;
                for (int row = 0; row < n; row++)
                {
                    for (int column = row + 1; column < n; column++)
                    {
                        double magnitude = Math.Abs(matrix[row, column]);
                        if (magnitude <= largest) continue;
                        largest = magnitude;
                        p = row;
                        q = column;
                    }
                }

                double diagonalScale = 0;
                for (int i = 0; i < n; i++) diagonalScale = Math.Max(diagonalScale, Math.Abs(matrix[i, i]));
                if (largest <= 1e-14 * Math.Max(diagonalScale, 1.0)) break;

                double app = matrix[p, p];
                double aqq = matrix[q, q];
                double apq = matrix[p, q];
                double angle = 0.5 * Math.Atan2(2.0 * apq, aqq - app);
                double c = Math.Cos(angle);
                double s = Math.Sin(angle);

                for (int k = 0; k < n; k++)
                {
                    if (k == p || k == q) continue;
                    double mkp = matrix[k, p];
                    double mkq = matrix[k, q];
                    matrix[k, p] = matrix[p, k] = c * mkp - s * mkq;
                    matrix[k, q] = matrix[q, k] = s * mkp + c * mkq;
                }

                matrix[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
                matrix[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
                matrix[p, q] = matrix[q, p] = 0;

                for (int row = 0; row < n; row++)
                {
                    double vip = vectors[row, p];
                    double viq = vectors[row, q];
                    vectors[row, p] = c * vip - s * viq;
                    vectors[row, q] = s * vip + c * viq;
                }
            }

            var order = Enumerable.Range(0, n).OrderByDescending(i => matrix[i, i]).ToArray();
            var values = new double[n];
            var sortedVectors = new double[n, n];
            for (int column = 0; column < n; column++)
            {
                int original = order[column];
                values[column] = matrix[original, original];
                for (int row = 0; row < n; row++) sortedVectors[row, column] = vectors[row, original];
            }
            return (values, sortedVectors);
        }
    }
}
