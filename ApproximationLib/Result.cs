namespace ApproximationLib;

public sealed record ParetoSolution(
    string Formula,
    string FunctionalFormula,
    int Size,
    double TrainingLoss,
    double ValidationLoss);

/// <summary>The fitted symbolic model and diagnostics.</summary>
public sealed class Result
{
    public required string BestFormula { get; init; }
    public required string FunctionalFormula { get; init; }
    public double Fitness { get; init; }
    public double MSE { get; init; }
    public double RMSE { get; init; }
    public double MAE { get; init; }
    public double R2 { get; init; }
    public double ValidationLoss { get; init; }
    public int Size { get; init; }
    public int Generations { get; init; }
    public TimeSpan TrainingTime { get; init; }
    public required Func<double, double> Function { get; init; }
    public required Func<double[], double> MultiFunction { get; init; }
    public required IReadOnlyList<string> FeatureNames { get; init; }
    public required IReadOnlyList<ParetoSolution> ParetoFront { get; init; }
    public required IReadOnlyList<GenerationData> History { get; init; }

    public double Predict(params double[] features) => MultiFunction(features);
    public double[] Predict(IEnumerable<double[]> rows) => rows.Select(MultiFunction).ToArray();
}
