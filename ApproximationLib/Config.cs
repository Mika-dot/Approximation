namespace ApproximationLib;

public enum LossFunction
{
    MeanSquaredError,
    MeanAbsoluteError,
    Huber
}

public enum ParentSelection
{
    EpsilonLexicase,
    Tournament
}

/// <summary>Settings for deterministic, tree-based symbolic regression.</summary>
public sealed class Config
{
    public int PopulationSize { get; set; } = 600;
    public int MaxGenerations { get; set; } = 250;
    public int? Seed { get; set; } = 42;
    public int MaxDepth { get; set; } = 7;
    public int MaxMutationDepth { get; set; } = 3;
    public int MaxNodes { get; set; } = 63;
    public double SurvivalRate { get; set; } = 0.04;
    public double RandomImmigrantRate { get; set; } = 0.03;
    public double CrossoverRate { get; set; } = 0.82;
    public double MutationRate { get; set; } = 0.35;
    public ParentSelection Selection { get; set; } = ParentSelection.EpsilonLexicase;
    public int TournamentSize { get; set; } = 5;
    public int LexicasePoolSize { get; set; } = 32;
    public double ParsimonyCoefficient { get; set; } = 0.001;
    public double ValidationFraction { get; set; } = 0.2;
    public int EarlyStoppingPatience { get; set; } = 35;
    public double TargetError { get; set; } = 1e-10;
    public LossFunction Loss { get; set; } = LossFunction.Huber;
    public double HuberDelta { get; set; } = 1.0;
    public bool EnableLinearScaling { get; set; } = true;
    public int ConstantOptimizationInterval { get; set; } = 10;
    public int ConstantOptimizationCandidates { get; set; } = 8;
    public int ConstantOptimizationPasses { get; set; } = 3;
    public double ConstantMin { get; set; } = -10.0;
    public double ConstantMax { get; set; } = 10.0;
    public bool ParallelEvaluation { get; set; } = true;
    public bool EnableLogging { get; set; }
    public bool GenerateReport { get; set; }
    public string OutputPath { get; set; } = "symbolic-regression-report.html";
    public int TopSolutionsToTrack { get; set; } = 8;

    public string[] Functions { get; set; } =
    {
        "+", "-", "*", "protectedDiv", "sin", "cos", "tanh", "log", "sqrt", "abs", "exp", "min", "max"
    };

    internal void Validate()
    {
        if (PopulationSize < 20) throw new ArgumentOutOfRangeException(nameof(PopulationSize), "PopulationSize must be at least 20.");
        if (MaxGenerations < 1) throw new ArgumentOutOfRangeException(nameof(MaxGenerations));
        if (MaxDepth < 2 || MaxDepth > 30) throw new ArgumentOutOfRangeException(nameof(MaxDepth));
        if (MaxMutationDepth < 1 || MaxMutationDepth > MaxDepth) throw new ArgumentOutOfRangeException(nameof(MaxMutationDepth));
        if (MaxNodes < 3) throw new ArgumentOutOfRangeException(nameof(MaxNodes));
        CheckRate(SurvivalRate, nameof(SurvivalRate));
        CheckRate(RandomImmigrantRate, nameof(RandomImmigrantRate));
        CheckRate(CrossoverRate, nameof(CrossoverRate));
        CheckRate(MutationRate, nameof(MutationRate));
        if (SurvivalRate + RandomImmigrantRate >= 1) throw new ArgumentException("SurvivalRate + RandomImmigrantRate must be below 1.");
        if (TournamentSize < 2) throw new ArgumentOutOfRangeException(nameof(TournamentSize));
        if (LexicasePoolSize < 2) throw new ArgumentOutOfRangeException(nameof(LexicasePoolSize));
        if (ParsimonyCoefficient < 0) throw new ArgumentOutOfRangeException(nameof(ParsimonyCoefficient));
        if (ValidationFraction < 0 || ValidationFraction >= 0.5) throw new ArgumentOutOfRangeException(nameof(ValidationFraction));
        if (EarlyStoppingPatience < 1) throw new ArgumentOutOfRangeException(nameof(EarlyStoppingPatience));
        if (HuberDelta <= 0) throw new ArgumentOutOfRangeException(nameof(HuberDelta));
        if (ConstantMin >= ConstantMax) throw new ArgumentException("ConstantMin must be below ConstantMax.");
        if (Functions is null || Functions.Length == 0) throw new ArgumentException("At least one function is required.", nameof(Functions));
        foreach (string function in Functions)
            if (!ExpressionTree.SupportedFunctions.Contains(function))
                throw new ArgumentException($"Unsupported function '{function}'.", nameof(Functions));
    }

    internal Config Copy() => (Config)MemberwiseClone();

    private static void CheckRate(double value, string name)
    {
        if (value < 0 || value > 1) throw new ArgumentOutOfRangeException(name, "Rate must be in [0, 1].");
    }
}
